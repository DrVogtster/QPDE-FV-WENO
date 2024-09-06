import numpy as np
import matplotlib.pyplot as plt
import time
import math
from euler_utils import *
from prettytable import PrettyTable
# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------

# Pre-defined test cases
from euler_tests import *

"""
# Custom job parameters (overwrites pre-defined test case)
flux = LxW # LxF, LxW
inter = DoNone # DoNone, WENO, WENO_Roe
integ = RK1 # RK1, RK3, RK4 <!> RK4 requires smaller time-steps by a factor 2/3 (cf. CFL)
# BC = PeriodicBC # OutgoingBC, PeriodicBC
# u0, pb = Density() # Density(), Riemann(rhoJ,uJ,pJ)
# xlims = np.array([0, 2]) # Physical domain
# Tf = 2 # Final time
"""

# Graphics
plots = 0 # 0, 1, 2, 3

def run(Nx):
	# Mesh size
	Nx_normal=Nx
	Co = 0.6 # CFL
	method="LxF"
	title="WENO"

	# ----------------------------------------------------------------
	# Simulation
	# ----------------------------------------------------------------

	# Data initialisation
	
	n = 0
	dx = np.abs( (xlims[1]-xlims[0])/Nx )
	# x = np.linspace(xlims[0]-3*dx, xlims[1]+2*dx, Nx+6)
	x = np.linspace(xlims[0]-2.5*dx, xlims[1]+2.5*dx, Nx+6)
	n_x_normal = Nx
	Nx = Nx+6


	# Initial condition averaging
	
	u = BC(cellav(u0, x, dx))
	# CFL
	vals, _ = EigA(u)
	amax = np.max(np.max(np.abs(vals)))
	dt = Co * dx/amax

	# Evolution right-hand side
	

	# Graphics initialisation
	# if plots:
	# 	figure, ax = plt.subplots(figsize=(10, 8))
	# 	M = 1.1 * (np.max(np.abs(u[plots-1,:])) + 0.02)
	# 	plt.axis([x[0], x[Nx-1], -M, M])
	# 	line1, = ax.plot(x,u[plots-1,:],'b.-')
	# 	plt.xlabel('x')
	# 	plt.ylabel('u_'+str(plots))
	# 	plt.title('t = '+str(t))
	# 	plt.draw()

	# Main loop
	print('Entering loop (gamma = '+str(gam)+').')
	print("\n "+ str(n_x_normal))
	tStart = time.time()
	tPlot = time.time()


	u = BC(cellav(u0, x, dx))
	t = 0
	flux = LxF # LxF, LxW
	inter = WENO_Roe
	L = RHS(flux, inter, BC, dt, dx)
	while t<Tf:
		# Iteration
		u = integ(L, u, dt)
		t = t + dt
		n = n + 1

		# CFL
		vals, _ = EigA(u)
		amax = np.max(np.max(np.abs(vals)))
		dt = Co * dx/amax

		# Evolution right-hand side
		if flux == LxW:
			L = RHS(flux, inter, BC, dt, dx)

		# Graphics update
		if (plots > 0) and (time.time() - tPlot > 1.5):
			# intermediate solution
			line1.set_ydata(u[plots-1,:])
			figure.canvas.draw()
			figure.canvas.flush_events()
			plt.title('t = '+str(t))
			plt.pause(0.05)
			tPlot = time.time()
	u_WENO = u
	tEnd = time.time()
	print('Elapsed time is '+str(tEnd-tStart)+' seconds.')
	print('Terminated in '+str(n)+' iterations.')

	# Exact solution
	if pb == 'Density':
		utheo = lambda x: u0(x - t)
	elif pb == 'Riemann':
		UJ = np.array([rhoJ, rhoJ*uJ, 0.5*rhoJ*uJ**2 + pJ/(gam-1)])
		utheo = RiemannExact(UJ, gam, t)

	# Plot final solution
	if plots:
		# numerical solution
		line1.set_ydata(u[plots-1,:])
		figure.canvas.draw()
		figure.canvas.flush_events()
		plt.title('t = '+str(t))
		# exact solution
		xplot = np.linspace(x[0], x[Nx-1], math.floor(np.max([2 * Nx, 1e3])))
		uth = utheo(xplot)
		plt.plot(xplot, uth[plots-1, :], 'k-')
		plt.draw()
		plt.show()

	# Error wrt. exact solution
	if test==1:
		uth = cellav(utheo, x, dx)
		ierr = (x>xlims[0])*(x<xlims[1])
		derr = u[:,ierr] - uth[:,ierr]
		one_err = np.array([np.linalg.norm(derr[0,:]*dx,1), np.linalg.norm(derr[1,:]*dx,1), np.linalg.norm(derr[2,:]*dx,1)])
		#two_err = np.array([np.linalg.norm(derr[0,:]*dx,2), np.linalg.norm(derr[1,:]*dx,2), np.linalg.norm(derr[2,:]*dx,2)])
		two_err = np.array([np.linalg.norm(derr[0,:]*math.sqrt(dx),2), np.linalg.norm(derr[1,:]*math.sqrt(dx),2), np.linalg.norm(derr[2,:]*math.sqrt(dx),2)])
		inf_err = np.array([np.linalg.norm(derr[0,:],np.inf), np.linalg.norm(derr[1,:],np.inf), np.linalg.norm(derr[2,:],np.inf)])
		return (one_err, two_err,inf_err,n_x_normal)
		print('L1, L2, Linf errors')
		print(np.array([one_err, two_err, inf_err]))
	if test==2:
		uth = cellav(utheo, x, dx)
		# ierr = (x>xlims[0])*(x<xlims[1])
		# derr = u[:,ierr] - uth[:,ierr]
		# one_err = np.array([np.linalg.norm(derr[0,:]*dx,1), np.linalg.norm(derr[1,:]*dx,1), np.linalg.norm(derr[2,:]*dx,1)])
		# two_err = np.array([np.linalg.norm(derr[0,:]*dx,2), np.linalg.norm(derr[1,:]*dx,2), np.linalg.norm(derr[2,:]*dx,2)])
		# inf_err = np.array([np.linalg.norm(derr[0,:],np.inf), np.linalg.norm(derr[1,:],np.inf), np.linalg.norm(derr[2,:],np.inf)])
		# print('L1, L2, Linf errors')
		# print(np.array([one_err, two_err, inf_err]))
		
	n = 0
	dx = np.abs( (xlims[1]-xlims[0])/Nx )
	# x = np.linspace(xlims[0]-3*dx, xlims[1]+2*dx, Nx+6)
	x = np.linspace(xlims[0]-2.5*dx, xlims[1]+2.5*dx, Nx+6)
	Nx = Nx+6

	# Initial condition averaging
	
	u = BC(cellav(u0, x, dx))
	# CFL
	vals, _ = EigA(u)
	amax = np.max(np.max(np.abs(vals)))
	dt = Co * dx/amax

	# Evolution right-hand side
	

	# Graphics initialisation
	# if plots:
	# 	figure, ax = plt.subplots(figsize=(10, 8))
	# 	M = 1.1 * (np.max(np.abs(u[plots-1,:])) + 0.02)
	# 	plt.axis([x[0], x[Nx-1], -M, M])
	# 	line1, = ax.plot(x,u[plots-1,:],'b.-')
	# 	plt.xlabel('x')
	# 	plt.ylabel('u_'+str(plots))
	# 	plt.title('t = '+str(t))
	# 	plt.draw()

	# Main loop
	print('Entering loop (gamma = '+str(gam)+'). - Nx=' +str(Nx_normal))
	tStart = time.time()
	tPlot = time.time()
	u = BC(cellav(u0, x, dx))
	t = 0
	flux = LxF # LxF, LxW
	inter = DoNone
	L = RHS(flux, inter, BC, dt, dx)
	while t<Tf:
		# Iteration
		u = integ(L, u, dt)
		t = t + dt
		n = n + 1

		# CFL
		vals, _ = EigA(u)
		amax = np.max(np.max(np.abs(vals)))
		dt = Co * dx/amax

		# Evolution right-hand side
		if flux == LxW:
			L = RHS(flux, inter, BC, dt, dx)

		# Graphics update
		if (plots > 0) and (time.time() - tPlot > 1.5):
			# intermediate solution
			line1.set_ydata(u[plots-1,:])
			figure.canvas.draw()
			figure.canvas.flush_events()
			plt.title('t = '+str(t))
			plt.pause(0.05)
			tPlot = time.time()
	tEnd = time.time()
	u_LxF = u
	u = BC(cellav(u0, x, dx))
	t = 0
	flux = LxW # LxF, LxW
	inter = DoNone
	L = RHS(flux, inter, BC, dt, dx)
	while t<Tf:
		# Iteration
		u = integ(L, u, dt)
		t = t + dt
		n = n + 1

		# CFL
		vals, _ = EigA(u)
		amax = np.max(np.max(np.abs(vals)))
		dt = Co * dx/amax

		# Evolution right-hand side
		if flux == LxW:
			L = RHS(flux, inter, BC, dt, dx)

		# Graphics update
		if (plots > 0) and (time.time() - tPlot > 1.5):
			# intermediate solution
			line1.set_ydata(u[plots-1,:])
			figure.canvas.draw()
			figure.canvas.flush_events()
			plt.title('t = '+str(t))
			plt.pause(0.05)
			tPlot = time.time()
	tEnd = time.time()
	u_LxW = u

	u = BC(cellav(u0, x, dx))
	t = 0
	flux = LxF # LxF, LxW
	inter = WENO_Roe
	L = RHS(flux, inter, BC, dt, dx)
	while t<Tf:
		# print("time =: " + str(t), end="\r", flush=True)
		# Iteration
		u = integ(L, u, dt)
		t = t + dt
		n = n + 1

		# CFL
		vals, _ = EigA(u)
		amax = np.max(np.max(np.abs(vals)))
		dt = Co * dx/amax

		# Evolution right-hand side
		if flux == LxW:
			L = RHS(flux, inter, BC, dt, dx)
	u_WENO=u
	print("starting to plot...")
	dx = np.abs( (xlims[1]-xlims[0])/512 )
	x_fine = np.linspace(xlims[0]-3*dx, xlims[1]+2*dx, 512+6)
	plt.plot(x_fine,utheo(x_fine)[0],'k',label="Exact")
	plt.xlabel(r"$x$")
	plt.ylabel(r"$\rho(x,T)$")
	
	plt.plot(x,u_LxF[0,:],'b--',label="LxF")
	plt.plot(x,u_LxW[0,:],'g--',label="LxW")
	plt.plot(x,u_WENO[0,:],'r--',label="WENO")
	plt.legend()
		# utheo = lambda x: u0(x - (t+dt))
		# print(dt)
		# print(u[0,:][::5])
	
	plt.savefig(title+"allononeplot-" + str(Nx_normal) +".pdf")
	plt.clf()
		#utheo = lambda x: u0(x - (t+dt))



def generate_convergence_history(error_list, n_list, file_name, title):
	x = PrettyTable()
	x.field_names = ["N", "Max Error", "Convergence Order"]
	x.add_row([n_list[0], error_list[0], "-"])

	con_list = np.zeros(len(error_list) - 1)

	for i in range(0, len(error_list) - 1):
		con_order = np.log2(error_list[i] / error_list[i + 1])
		x.add_row([n_list[i + 1], error_list[i + 1], con_order])
		con_list[i] = con_order

	print(title)
	print(x)
	avg_con = np.mean(con_list)
	print(f"Average Convergence Order: {avg_con}")

	f = open(file_name, "a")
	f.write(title + str("\n"))
	f.write(x.get_string() + str("\n"))
	f.write(f"Average Convergence Order: {avg_con}")
	f.close()
def gen_error(n_vals,error_vals,time_list,second_list,file_name="errorout.txt"):
	f = open(file_name, "a")
	
	for k in range(0,len(n_vals)):
		
		f.write("N=" +str(n_vals[k])+ str("\n"))
		f.write(str(error_vals[k])+ str("\n"))
	f.write(str(time_list)+ str("\n"))
	f.write(str(second_list)+ str("\n"))
	f.close()

nx_list_main=[16,32,64]

rho_err_l1=[]
rho_err_l2 = []
rho_err_linf = []

u_err_l1=[]
u_err_l2 = []
u_err_linf = []

p_err_l1=[]
p_err_l2 = []
p_err_linf = []
print("run starting.....")
full_error=[]
time_list=[]
secondary_list=[]
nx_list=[]
file_name="classicalweno_nature_run_smooth_updated"
for nx in nx_list_main:
	
	out = run(nx)
	if(test==1):
		(one_err,two_err,inf_err,nx) = out
		full_error.append(np.array([one_err, two_err, inf_err]))
		
		nx_list.append(nx)

		rho_err_l1.append(one_err[0])
		u_err_l1.append(one_err[1])
		p_err_l1.append(one_err[2])

		rho_err_l2.append(two_err[0])
		u_err_l2.append(two_err[1])
		p_err_l2.append(two_err[2])
		
		rho_err_linf.append(inf_err[0])
		u_err_linf.append(inf_err[1])
		p_err_linf.append(inf_err[2])  



if(test==1):
	generate_convergence_history(rho_err_l1, nx_list, file_name, " rho L1")
	generate_convergence_history(u_err_l1, nx_list, file_name, " rho u L1")
	generate_convergence_history(p_err_l1, nx_list, file_name, " E L1")
	generate_convergence_history(rho_err_l2, nx_list, file_name, " rho L2")
	generate_convergence_history(u_err_l2, nx_list, file_name, " rho u L2")
	generate_convergence_history(p_err_l2, nx_list, file_name, " E L2")
	generate_convergence_history(rho_err_linf, nx_list, file_name, " rho Linf")
	generate_convergence_history(u_err_linf, nx_list, file_name, " rho u Linf")
	generate_convergence_history(p_err_linf, nx_list, file_name, " E Linf")
	gen_error(nx_list,full_error,time_list,secondary_list,"classicalwenosmoothnature.txt")
