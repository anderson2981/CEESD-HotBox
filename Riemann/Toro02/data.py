####################
# program controls #
####################
t_final = 0.15
#t_final = 0.001
current_cfl = 1.0
current_dt = .0001
#current_t = 0
current_t = 1.e-2
constant_cfl = False

# some i/o frequencies
nstatus = 1
nrestart = 500
nviz = 10
nhealth = 1

health_pres_min = 0.0
health_pres_max = 10

####################################
# discretization and model control #
####################################
nel_x = 400
order = 1
dim = 1
alpha_sc = 0. # no artificial dissipation
#alpha_sc = 0.001 # stabilize initial solution
s0_sc = -5.0
kappa_sc = 0.5

###############################
# problem initialization data #
###############################

# first problem from Toro, modified Sod shock
#init_type = 0 # for planar discontinuity
init_sigma = 1e-2
init_type = 1 # for riemann exact solution
gamma = 1.4
r = 300
rhol = 1.0
rhor = 1.0
pl = 0.4
pr = 0.4
ul = -2.0
ur = 2.0
tl = pl/rhol/r
tr = pr/rhor/r
x0 = 0.5
