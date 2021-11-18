"""Demonstrate Sod's 1D shock example."""

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
import numpy as np  # noqa
import pyopencl as cl
import pyopencl.tools as cl_tools
from functools import partial
from pytools.obj_array import make_obj_array

from meshmode.array_context import (
    PyOpenCLArrayContext,
    SingleGridWorkBalancingPytatoArrayContext as PytatoPyOpenCLArrayContext
)
from arraycontext import thaw, freeze, flatten, unflatten, to_numpy, from_numpy
from mirgecom.profiling import PyOpenCLProfilingArrayContext
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer


from mirgecom.euler import euler_operator
from mirgecom.fluid import make_conserved
from mirgecom.simutil import (
    get_sim_timestep,
    generate_and_distribute_mesh
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point

from mirgecom.integrators import rk4_step, euler_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import PrescribedInviscidBoundary
from mirgecom.initializers import SodShock1D
from mirgecom.eos import IdealSingleGas

from logpyle import IntervalTimer, set_dt
from mirgecom.euler import extract_vars_for_logging, units_for_logging
from mirgecom.inviscid import get_inviscid_timestep, get_inviscid_cfl
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

from ToroExact import exactRP

class RiemannExact:
    r"""Generate exact solutions to Riemann problems for the Euler equations

    This class wraps an exact Riemann solver from Toro [ref].

    E.F. Toro. Riemann solvers and numerical methods for fluid dynamics:
    a practical introduction. Springer, Berlin, New York, 2009.

    The solver is a python-based implementation of the fortran implementation
    found in [1]. The solution is generated from an initial left/right discontinuous
    flow state for density, pressure, and velocity at initial location x0.
    The solution is returned at time t.

    Riemann solver Python implementation taken from:
    https://github.com/tahandy/ToroExact

    .. automethod:: __init__
    .. automethod:: __call__
    """
    def __init__(self, *, dim, x0, gamma, rhol, pl, ul, rhor, pr, ur):
        r"""Initialize the problem parameters.

        Parameters
        ----------
        x0: float
            discontinuity starting location
        gamma: float
            ratio of specific heats
        rhol: float
            density left of the discontinuity
        pl: float
            pressure left of the discontinuity
        ul: float
            velocity left of the discontinuity
        rhor: float
            density left of the discontinuity
        pr: float
            pressure left of the discontinuity
        ur: float
            velocity left of the discontinuity
        """

        self._dim = dim
        self._x0 = x0
        #self._rhol = rhol
        #self._pl = pl
        #self._ul = ul
        #self._rhor = rhor
        #self._pr = pr
        #self._ur = ur
        stateL = [rhol, ul, pl]
        stateR = [rhor, ur, pr]

        self._rp = exactRP.exactRP(gamma, stateL, stateR)
        success = self._rp.solve()
        if not success:
            print(f"gamma={gamma}")
            print(f"left state: rho={rhol}, p={pl}, u={ul}")
            print(f"right state: rho={rhor}, p={pr}, u={ur}")
            raise RuntimeError("Unable to solve specified Riemann problem.")


    def __call__(self, x_vec, eos, *, time=0.0):
        """Extract the solution of the Riemann problem on x_vec at a given time.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        eos:
            Equation of state object
        time: float
            Time at which solution is desired. The problem structure is dependent
            on time
        """
        xpos = x_vec[0]
        actx = xpos.array_context
        zeros = 0*xpos
        x0 = zeros + self._x0

        if time < 1e-9:
            s = xpos - x0
        else:
            s = (xpos - x0)/time

        actx = xpos.array_context
        s_flat = to_numpy(flatten(s, actx), actx)

        dens, pres, velx, eint, scpd = self._rp.sample(s_flat)

        rho = unflatten(xpos, from_numpy(np.array(dens), actx), actx)
        u = make_obj_array([unflatten(xpos, from_numpy(np.array(velx), actx), actx)])
        e = unflatten(xpos, from_numpy(np.array(eint), actx), actx)

        return make_conserved(dim=self._dim, mass=rho,
                              momentum=rho*u,
                              energy=rho*(e + 0.5*np.dot(u, u))
                             )


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
        filename=f"{casename}.sqlite", mode="wo", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(
            cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    actx = actx_class(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    # timestepping control
    if use_leap:
        from leap.rk import RK4MethodBuilder
        timestepper = RK4MethodBuilder("state")
    else:
        timestepper = rk4_step
        #timestepper = euler_step
    #t_final = 1.0
    t_final = 0.2
    #t_final = 0.001
    current_cfl = 1.0
    current_dt = .0001
    current_t = 0
    constant_cfl = False
    current_step = 0

    # some i/o frequencies
    nstatus = 100
    nrestart = 500
    nviz = 10
    nhealth = 10

    health_pres_min = 0.0
    health_pres_max = 10

    dim = 1
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
        #nel_1d = 2501
        #nel_1d = 251
        #nel_1d = 20
        nel_1d = 500
        #nel_1d = 500
        #nel_1d = 1000
        box_ll = 0.0
        box_ur = 1.0
        generate_mesh = partial(generate_regular_rect_mesh, a=(box_ll,)*dim,
                                b=(box_ur,) * dim, nelements_per_axis=(nel_1d,)*dim)
        local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                    generate_mesh)
        local_nelements = local_mesh.nelements

    order = 1
    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(discr.nodes(), actx)

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

    # first problem from Toro, modified Sod shock
    gamma = 1.4
    r=300
    initializer = SodShock1D(dim=dim, x0=0.3, rhol=1.0, rhor=0.125,
                             pleft=1.0, pright=0.1, ul=0.75, ur=0.0)
    exact_solution = RiemannExact(dim=dim, x0=0.3, gamma=gamma, rhol=1.0, rhor=0.125,
                                  pl=1.0, pr=0.1, ul=0.75, ur=0.0)
    eos = IdealSingleGas(gamma=gamma, gas_const=r)
    boundaries = {
        BTAG_ALL: PrescribedInviscidBoundary(fluid_solution_func=initializer)
    }
    if rst_filename:
        current_t = restart_data["t"]
        current_step = restart_data["step"]
        current_state = restart_data["state"]
        if logmgr:
            from mirgecom.logging_quantities import logmgr_set_time
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        # Set the current state from time 0
        current_state = initializer(nodes)

    visualizer = make_visualizer(discr)

    initname = initializer.__class__.__name__
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

    def my_write_status(component_errors):
        if rank == 0:
            logger.info(
                "------- errors="
                + ", ".join("%.3g" % en for en in component_errors)
            )

    def my_write_viz(step, t, state, dv=None, exact=None, resid=None, ts_field=None):
        if dv is None:
            dv = eos.dependent_vars(state)
        if exact is None:
            exact = initializer(x_vec=nodes, eos=eos, time=t)
        dv_exact = eos.dependent_vars(exact)
        if resid is None:
            resid = state - exact
        internal_energy = eos.internal_energy(state)/state.mass
        internal_energy_exact = eos.internal_energy(exact)/exact.mass
        viz_fields = [("cv", state),
                      ("dv", dv),
                      ("velocity", state.velocity),
                      ("internal_energy", internal_energy),
                      ("exact", exact),
                      ("dv_exact", dv_exact),
                      ("velocity_exact", exact.velocity),
                      ("internal_energy_exact", internal_energy_exact),
                      ("residual", resid)]
        from mirgecom.simutil import write_visfile
        write_visfile(discr, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True, vis_timer=vis_timer)

    def my_write_restart(state, step, t):
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

    def my_health_check(pressure, component_errors):
        health_error = False
        from mirgecom.simutil import check_naninf_local, check_range_local
        if check_naninf_local(discr, "vol", pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if check_range_local(discr, "vol", pressure, health_pres_min,
                             health_pres_max):
            health_error = True
            logger.info(f"{rank=}: Invalid pressure data found.")

        #exittol = .09
        exittol = 100
        if max(component_errors) > exittol:
            health_error = True
            if rank == 0:
                logger.info("Solution diverged from exact soln.")

        return health_error

    def my_get_timestep(t, dt, state):
        t_remaining = max(0, t_final - t)
        if constant_cfl:
            ts_field = current_cfl * get_inviscid_timestep(discr, eos=eos,
                                                             cv=state)
            from grudge.op import nodal_min
            dt = actx.to_numpy(nodal_min(discr, "vol", ts_field))
            cfl = current_cfl
        else:
            ts_field = get_inviscid_cfl(discr, eos=eos, dt=dt,
                                          cv=state)
            from grudge.op import nodal_max
            cfl = actx.to_numpy(nodal_max(discr, "vol", ts_field))

        return ts_field, cfl, min(t_remaining, dt)

    def my_pre_step(step, t, dt, state):
        try:
            dv = None
            #exact = None
            exact = exact_solution(x_vec=nodes, eos=eos, time=t)
            component_errors = None

            if logmgr:
                logmgr.tick_before()

            ts_field, cfl, dt = my_get_timestep(t, dt, state)
            log_cfl.set_quantity(cfl)

            from mirgecom.simutil import check_step
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)
            do_status = check_step(step=step, interval=nstatus)

            if do_health:
                dv = eos.dependent_vars(state)
                if exact is None:
                    exact = initializer(x_vec=nodes, eos=eos, time=t)
                from mirgecom.simutil import compare_fluid_solutions
                component_errors = compare_fluid_solutions(discr, state, exact)
                health_errors = global_reduce(
                    my_health_check(dv.pressure, component_errors), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, state=state)

            if do_viz:
                if dv is None:
                    dv = eos.dependent_vars(state)
                if exact is None:
                    exact = initializer(x_vec=nodes, eos=eos, time=t)
                resid = state - exact
                my_write_viz(step=step, t=t, state=state, dv=dv, exact=exact,
                             resid=resid)

            if do_status:
                if component_errors is None:
                    from mirgecom.simutil import compare_fluid_solutions
                    if exact is None:
                        exact = initializer(x_vec=nodes, eos=eos, time=t)
                    component_errors = \
                        compare_fluid_solutions(discr, state, exact)
                my_write_status(component_errors)

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
        return euler_operator(discr, cv=state, time=t,
                              boundaries=boundaries, eos=eos)

    current_dt = get_sim_timestep(discr, current_state, current_t, current_dt,
                                  current_cfl, eos, t_final, constant_cfl)

    current_step, current_t, current_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step, dt=current_dt,
                      post_step_callback=my_post_step,
                      state=current_state, t=current_t, t_final=t_final)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    final_dv = eos.dependent_vars(current_state)
    final_exact = initializer(x_vec=nodes, eos=eos, time=current_t)
    final_resid = current_state - final_exact
    my_write_viz(step=current_step, t=current_t, state=current_state, dv=final_dv,
                 exact=final_exact, resid=final_resid)
    my_write_restart(step=current_step, t=current_t, state=current_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":
    import argparse
    casename = "sod-shock"
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
