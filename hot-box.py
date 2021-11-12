"""Demonstrate acoustic pulse, and adiabatic slip wall."""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import logging
from mirgecom.mpi import mpi_entry_point
import numpy as np
from functools import partial
import pyopencl as cl
import pyopencl.tools as cl_tools
from pytools.obj_array import make_obj_array

from meshmode.array_context import (
    PyOpenCLArrayContext,
    SingleGridWorkBalancingPytatoArrayContext as PytatoPyOpenCLArrayContext
)
from mirgecom.profiling import PyOpenCLProfilingArrayContext
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer

from mirgecom.navierstokes import ns_operator
from mirgecom.simutil import (
    get_sim_timestep,
    generate_and_distribute_mesh
)
from mirgecom.io import make_init_message

from mirgecom.integrators import rk4_step
from mirgecom.fluid import make_conserved
from mirgecom.steppers import advance_state
from mirgecom.boundary import IsothermalNoSlipBoundary
from mirgecom.initializers import (Uniform)
from mirgecom.eos import IdealSingleGas
from mirgecom.transport import SimpleTransport

from logpyle import IntervalTimer, set_dt
from mirgecom.euler import extract_vars_for_logging, units_for_logging
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_many_discretization_quantities,
    logmgr_add_device_name,
    logmgr_add_device_memory_usage,
    set_sim_state,
    LogUserQuantity
)

logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass

class UniformModified:
    r"""Solution initializer for a uniform flow with boundary layer smoothing.

    Similar to the Uniform initializer, except the velocity profile is modified
    so that the velocity goes to zero at y(min, max)

    The smoothing comes from a hyperbolic tangent with weight sigma

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(
            self, *, dim=1, nspecies=0, pressure=1.0, temperature=2.5,
            velocity=None, mass_fracs=None,
            temp_wall, temp_sigma,
            xmin=0., xmax=1.0,
            ymin=0., ymax=1.0
    ):
        r"""Initialize uniform flow parameters.

        Parameters
        ----------
        dim: int
            specify the number of dimensions for the flow
        nspecies: int
            specify the number of species in the flow
        temperature: float
            specifies the temperature
        pressure: float
            specifies the pressure
        velocity: numpy.ndarray
            specifies the flow velocity
        temp_wall: float
            wall temperature
        temp_sigma: float
            near-wall temperature relaxation parameter
        vel_sigma: float
            near-wall velocity relaxation parameter
        xmin: flaot
            minimum y-coordinate for smoothing
        xmax: float
            maximum y-coordinate for smoothing
        ymin: flaot
            minimum y-coordinate for smoothing
        ymax: float
            maximum y-coordinate for smoothing
        """
        if velocity is not None:
            numvel = len(velocity)
            myvel = velocity
            if numvel > dim:
                dim = numvel
            elif numvel < dim:
                myvel = np.zeros(shape=(dim,))
                for i in range(numvel):
                    myvel[i] = velocity[i]
            self._velocity = myvel
        else:
            self._velocity = np.zeros(shape=(dim,))

        if mass_fracs is not None:
            self._nspecies = len(mass_fracs)
            self._mass_fracs = mass_fracs
        else:
            self._nspecies = nspecies
            self._mass_fracs = np.zeros(shape=(nspecies,))

        if self._velocity.shape != (dim,):
            raise ValueError(f"Expected {dim}-dimensional inputs.")

        self._pressure = pressure
        self._temperature = temperature
        self._dim = dim
        self._temp_wall = temp_wall
        self._temp_sigma = temp_sigma
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax

    def __call__(self, x_vec, *, eos, **kwargs):
        """
        Create a uniform flow solution at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: :class:`mirgecom.eos.IdealSingleGas`
            Equation of state class with method to supply gas *gamma*.
        """

        xpos = x_vec[0]
        ypos = x_vec[1]
        actx = ypos.array_context
        xmax = 0.0*x_vec[1] + self._xmax
        xmin = 0.0*x_vec[1] + self._xmin
        ymax = 0.0*x_vec[1] + self._ymax
        ymin = 0.0*x_vec[1] + self._ymin
        ones = (1.0 + x_vec[0]) - x_vec[0]

        pressure = self._pressure * ones
        temperature = self._temperature * ones

        # modify the temperature in the near wall region to match
        # the isothermal boundaries
        sigma = self._temp_sigma
        wall_temperature = self._temp_wall
        smoothing_min_x = actx.np.tanh(sigma*(actx.np.abs(xpos-xmin)))
        smoothing_max_x = actx.np.tanh(sigma*(actx.np.abs(xpos-xmax)))
        smoothing_min_y = actx.np.tanh(sigma*(actx.np.abs(ypos-ymin)))
        smoothing_max_y = actx.np.tanh(sigma*(actx.np.abs(ypos-ymax)))
        temperature = (wall_temperature +
                       (temperature - wall_temperature)*
                       smoothing_min_x*smoothing_max_x*
                       smoothing_min_y*smoothing_max_y)

        velocity = make_obj_array([self._velocity[i] * ones
                                   for i in range(self._dim)])
        y = make_obj_array([self._mass_fracs[i] * ones
                            for i in range(self._nspecies)])
        if self._nspecies:
            mass = eos.get_density(pressure, temperature, y)
        else:
            mass = pressure/temperature/eos.gas_const()
        specmass = mass * y

        mom = mass*velocity
        if self._nspecies:
            internal_energy = eos.get_internal_energy(temperature=temperature,
                                                      species_mass=specmass)
        else:
            internal_energy = pressure/(eos.gamma() - 1)
        kinetic_energy = 0.5 * np.dot(mom, mom)/mass
        energy = internal_energy + kinetic_energy

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=mom, species_mass=specmass)



@mpi_entry_point
def main(ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_profiling=False, casename=None,
         rst_filename=None, actx_class=PyOpenCLArrayContext):
    """Drive the example."""
    cl_ctx = ctx_factory()

    if casename is None:
        casename = "mirgecom"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_parts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(use_logmgr,
        filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(
            cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    actx = actx_class(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    # timestepping control
    current_step = 0
    if use_leap:
        from leap.rk import RK4MethodBuilder
        timestepper = RK4MethodBuilder("state")
    else:
        timestepper = rk4_step

    t_final = 1e-4
    current_cfl = 1.0
    current_dt = 1e-6
    current_t = 0
    constant_cfl = False

    # some i/o frequencies
    nstatus = 1
    nrestart = 100
    nviz = 10
    nhealth = 1

    health_pres_min = 1.0
    health_pres_max = 1e7

    alpha_sc = 0
    s0_sc = 5.0
    kappa_sc = 0.5

    dim = 2
    viz_path = "viz_data/"
    vizname = viz_path + casename

    rst_path = "restart_data/"
    rst_pattern = (
        rst_path + "{cname}-{step:04d}-{rank:04d}.pkl"
    )
    if rst_filename:  # read the grid from restart data
        rst_filename = f"{rst_filename}-{rank:04d}.pkl"
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, rst_filename)
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        assert restart_data["num_parts"] == num_parts
    else:  # generate the grid from scratch
        from meshmode.mesh.generation import generate_regular_rect_mesh
        box_ll = -0.5
        box_ur = 0.5
        nel_1d = 16
        generate_mesh = partial(generate_regular_rect_mesh, a=(box_ll,)*dim,
                                b=(box_ur,) * dim, nelements_per_axis=(nel_1d,)*dim)
        local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                    generate_mesh)
        local_nelements = local_mesh.nelements

    order = 2
    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(actx, discr.nodes())

    vis_timer = None
    log_cfl = LogUserQuantity(name="cfl", value=current_cfl)

    if logmgr:
        logmgr_add_device_name(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)
        logmgr_add_many_discretization_quantities(logmgr, discr, dim,
                             extract_vars_for_logging, units_for_logging)

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)
        logmgr.add_quantity(log_cfl, interval=nstatus)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s, "),
            ("cfl.max", ", cfl = {value:1.4f}\n"),
            ("min_pressure", "------- P (min, max) (Pa) = ({value:1.9e}, "),
            ("max_pressure",    "{value:1.9e})\n"),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s")
        ])

    mu = 1.e-5
    rho = 2.0
    #kappa = rho*mu/0.75
    kappa = 100000
    transport_model = SimpleTransport(viscosity=mu, thermal_conductivity=kappa)
    gamma = 1.4
    gas_constant = 500.0
    pressure = 1e6
    temperature = pressure/gas_constant/rho
    wall_temp = 300.0
    temp_sigma = 10
    eos = IdealSingleGas(gamma=gamma, gas_const=gas_constant, transport_model=transport_model)

    vel = np.zeros(shape=(dim,))
    orig = np.zeros(shape=(dim,))

    xmin=-0.5
    xmax=0.5
    ymin=-0.5
    ymax=0.5

    initializer = UniformModified(dim=dim, pressure=pressure, temperature=temperature,
                                  temp_wall=wall_temp, temp_sigma=temp_sigma,
                                  xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    wall = IsothermalNoSlipBoundary(wall_temperature=wall_temp)
    boundaries = {BTAG_ALL: wall}
    uniform_state = initializer(x_vec=nodes, eos=eos)

    if rst_filename:
        current_t = restart_data["t"]
        current_step = restart_data["step"]
        current_state = restart_data["state"]
        if logmgr:
            from mirgecom.logging_quantities import logmgr_set_time
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        # Set the current state from time 0
        current_state = uniform_state

    visualizer = make_visualizer(discr)

    initname = "pulse"
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=initname,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    def my_write_viz(step, t, state, dv=None, ts_field=None):
        if dv is None:
            dv = eos.dependent_vars(state)
        viz_fields = [("cv", state),
                      ("dv", dv)]
        from mirgecom.simutil import write_visfile
        write_visfile(discr, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True, vis_timer=vis_timer)

    def my_write_restart(step, t, state):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != rst_filename:
            rst_data = {
                "local_mesh": local_mesh,
                "state": state,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": num_parts
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(pressure):
        health_error = False
        from mirgecom.simutil import check_naninf_local, check_range_local
        if check_naninf_local(discr, "vol", pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if check_range_local(discr, "vol", pressure, health_pres_min, health_pres_max):
            health_error = True
            logger.info(f"{rank=}: Invalid pressure data found.")

        return health_error

    def my_get_viscous_timestep(discr, eos, cv):
        """Routine returns the the node-local maximum stable viscous timestep.

        Parameters
        ----------
        discr: grudge.eager.EagerDGDiscretization
            the discretization to use
        eos: :class:`~mirgecom.eos.GasEOS`
            A gas equation of state
        cv: :class:`~mirgecom.fluid.ConservedVars`
            Fluid solution

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The maximum stable timestep at each node.
        """
        from grudge.dt_utils import characteristic_lengthscales
        from mirgecom.fluid import compute_wavespeed

        length_scales = characteristic_lengthscales(cv.array_context, discr)

        mu = 0
        d_alpha_max = 0
        transport = eos.transport_model()
        if transport:
            from mirgecom.viscous import get_local_max_species_diffusivity
            mu = transport.viscosity(eos, cv)
            d_alpha_max = \
                get_local_max_species_diffusivity(
                    cv.array_context, discr,
                    transport.species_diffusivity(eos, cv)
                )

        return(
            length_scales / (compute_wavespeed(eos, cv)
            + ((mu + d_alpha_max + alpha_sc) / length_scales))
        )

    def my_get_viscous_cfl(discr, eos, dt, cv):
        """Calculate and return node-local CFL based on current state and timestep.

        Parameters
        ----------
        discr: :class:`grudge.eager.EagerDGDiscretization`
            the discretization to use
        eos: :class:`~mirgecom.eos.GasEOS`
            A gas equation of state
        dt: float or :class:`~meshmode.dof_array.DOFArray`
            A constant scalar dt or node-local dt
        cv: :class:`~mirgecom.fluid.ConservedVars`
            The fluid conserved variables

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The CFL at each node.
        """
        return dt / my_get_viscous_timestep(discr, eos=eos, cv=cv)

    def my_get_timestep(t, dt, state):
        t_remaining = max(0, t_final - t)
        if constant_cfl:
            ts_field = current_cfl * my_get_viscous_timestep(discr, eos=eos,
                                                             cv=state)
            from grudge.op import nodal_min
            dt = actx.to_numpy(nodal_min(discr, "vol", ts_field))
            cfl = current_cfl
        else:
            ts_field = my_get_viscous_cfl(discr, eos=eos, dt=dt,
                                          cv=state)
            from grudge.op import nodal_max
            cfl = actx.to_numpy(nodal_max(discr, "vol", ts_field))

        return ts_field, cfl, min(t_remaining, dt)

    def my_pre_step(step, t, dt, state):
        try:
            dv = None

            if logmgr:
                logmgr.tick_before()

            ts_field, cfl, dt = my_get_timestep(t, dt, state)
            log_cfl.set_quantity(cfl)

            from mirgecom.simutil import check_step
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            if do_health:
                dv = eos.dependent_vars(state)
                health_errors = global_reduce(my_health_check(dv.pressure), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, state=state)

            if do_viz:
                if dv is None:
                    dv = eos.dependent_vars(state)
                my_write_viz(step=step, t=t, state=state, dv=dv)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, state=state)
            my_write_restart(step=step, t=t, state=state)
            raise

        dt = get_sim_timestep(discr, state, t, dt, current_cfl, eos, t_final,
                              constant_cfl)
        return state, dt

    def my_post_step(step, t, dt, state):
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, state, eos)
            logmgr.tick_after()
        return state, dt

    def my_rhs(t, state):
        return (ns_operator(discr, cv=state, t=t, boundaries=boundaries, eos=eos))

    current_dt = get_sim_timestep(discr, current_state, current_t, current_dt,
                                  current_cfl, eos, t_final, constant_cfl)

    current_step, current_t, current_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step, dt=current_dt,
                      state=current_state, t=current_t, t_final=t_final)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")
    final_dv = eos.dependent_vars(current_state)
    ts_field, cfl, dt = my_get_timestep(t=current_t, dt=current_dt, state=current_state)
    my_write_viz(step=current_step, t=current_t, state=current_state, dv=final_dv,
                 ts_field=ts_field)
    my_write_restart(step=current_step, t=current_t, state=current_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":
    import argparse
    casename = "hotbox"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--log", action="store_true", default=True,
        help="turn on logging")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--restart_file", help="root name of restart file")
    parser.add_argument("--casename", help="casename to use for i/o")
    args = parser.parse_args()
    if args.profiling:
        if args.lazy:
            raise ValueError("Can't use lazy and profiling together.")
        actx_class = PyOpenCLProfilingArrayContext
    else:
        actx_class = PytatoPyOpenCLArrayContext if args.lazy \
            else PyOpenCLArrayContext

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if args.casename:
        casename = args.casename
    rst_filename = None
    if args.restart_file:
        rst_filename = args.restart_file

    main(use_logmgr=args.log, use_leap=args.leap, use_profiling=args.profiling,
         casename=casename, rst_filename=rst_filename, actx_class=actx_class)

# vim: foldmethod=marker
