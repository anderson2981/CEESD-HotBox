####################
# program controls #
####################
t_final = 0.2
#t_final = 0.001
current_cfl = 1.0
current_dt = .0001
current_t = 0
#current_t = 0.1
constant_cfl = False

# some i/o frequencies
nstatus = 1
nrestart = 500
nviz = 1
nhealth = 1

health_pres_min = 0.0
health_pres_max = 100000

####################################
# discretization and model control #
####################################
nel_x = 100
nel_y = 10
nel_z = 10
dim = 3
order = 1
alpha_sc = 0. # no artificial dissipation
#alpha_sc = 0.0001 # stabilize higher order
#alpha_sc = 0.0001 # this is minimum amount to resolve the entropy jump in the rarefaction
#alpha_sc = 0.001 # this amount resolved the wiggles at the shock front
s0_sc = -5.0
kappa_sc = 0.5

###############################
# problem initialization data #
###############################

# first problem from Toro, modified Sod shock
init_type = 0 # for planar discontinuity
init_sigma = 1e-2
#init_type = 1 # for riemann exact solution
gamma = 1.4
r = 300
rhol = 1.0
rhor = 0.125
pl = 1.0
pr = 0.1
ul = 0.75
ur = 0.0
tl = pl/rhol/r
tr = pr/rhor/r
x0 = 0.3
init_sigma = 1e-3
