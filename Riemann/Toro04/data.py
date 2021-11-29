####################
# program controls #
####################
t_final = 0.035
#t_final = 0.001
current_cfl = 1.0
current_dt = .00002
current_t = 0
#current_t = 5e-3
constant_cfl = False

# some i/o frequencies
nstatus = 1
nrestart = 500
nviz = 10
nhealth = 1

health_pres_min = 0.0
health_pres_max = 100000000

####################################
# discretization and model control #
####################################
nel_x = 500
order = 1
dim = 1
#alpha_sc = 0. # no artificial dissipation
alpha_sc = 0.01 # stabilize solution
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
rhol = 5.99924
rhor = 5.99242
pl = 460.894
pr = 46.0950
ul = 19.5975
ur = -6.19633
tl = pl/rhol/r
tr = pr/rhor/r
x0 = 0.4
