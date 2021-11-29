####################
# program controls #
####################
t_final = 2.0
#t_final = 0.001
current_cfl = 1.0
current_dt = .0025
current_t = 0
#current_t = 5e-3
constant_cfl = False

# some i/o frequencies
nstatus = 1
nrestart = 500
nviz = 50
nhealth = 1

health_pres_min = 0.0
health_pres_max = 100000000

####################################
# discretization and model control #
####################################
nel_x = 101
order = 1
dim = 1
alpha_sc = 0. # no artificial dissipation
#alpha_sc = 0.001 # stabilize solution
s0_sc = -5.0
kappa_sc = 0.5

###############################
# problem initialization data #
###############################

# first problem from Toro, modified Sod shock
init_type = 0 # for planar discontinuity
init_sigma = 1e-6
#init_type = 1 # for riemann exact solution
gamma = 1.4
r = 300
rhol = 1.4
rhor = 1.0
pl = 1.0
pr = 1.0
ul = 0.1
ur = 0.1
tl = pl/rhol/r
tr = pr/rhor/r
x0 = 0.5
