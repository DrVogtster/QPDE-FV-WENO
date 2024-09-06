import numpy as np

import matplotlib.pyplot as plt
# plt.switch_backend('agg')

import time
import math
# from euler_utils import *
from euler_utils_new import *
from prettytable import PrettyTable
import kacewicz_v21_B as kacewicz

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------

# Pre-defined test cases
from euler_tests import *

"""
# Custom job parameters
flux = LxF # LxF
inter = WENO_Roe # DoNone, WENO, WENO_Roe
integ = RK3 # RK1, RK3, RK4 <!> RK4 requires smaller time-steps by a factor 2/3 (cf. CFL)
BC = PeriodicBC # OutgoingBC, PeriodicBC
u0, pb = Density() # Density(), Riemann(rhoJ,uJ,pJ)
xlims = np.array([0, 2]) # Physical domain
Tf = 2 # Final time
"""

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

def run(Nx,title):
	# Graphics
	plots = 0 # 0, 1, 2, 3

	# Mesh size
   
	Co = 0.6 # CFL

	# ----------------------------------------------------------------
	# Simulation
	# ----------------------------------------------------------------

	# Data initialisation
	t = 0
	n = 0
	dx = np.abs( (xlims[1]-xlims[0])/Nx )
	# x = np.linspace(xlims[0]-3*dx, xlims[1]+2*dx, Nx+6)
	x = np.linspace(xlims[0]-2.5*dx, xlims[1]+2.5*dx, Nx+6)
	Nx_normal = Nx
	Nx = Nx+6

	# Evolution right-hand side
	L = RHS(flux, inter, BC, dx)

	# Initial condition averaging
	u=cellav(u0, x, dx)
	u = BC_og(u)
	ic_kac = np.ravel(u)
	
	# CFL
	vals, _ = EigA(u)
	amax = np.max(np.max(np.abs(vals)))
	if pb == 'Density':
		# dt = Co * dx/amax
		dt =  1.0/(8.0)*Co * dx/amax
	else:
		# dt = 1.0/(2**(Nx/16))* Co * dx/amax
		dt = 1.0/(4.0)* Co * dx/amax
	time_steps = math.ceil(Tf/dt)
	t_list = np.linspace(0,Tf,time_steps)
	@jax.jit
	def driver_jax(z):
		#return  np.ravel(L(np.reshape(z, (3, Nx))))
		z = z.astype(jnp.float64)
		return  jnp.ravel(L(jnp.reshape(z, (3, Nx))))
	# def driver(z):
	# 	#return  np.ravel(L(np.reshape(z, (3, Nx))))

	# 	return  np.ravel(L(np.reshape(z, (3, Nx))))
	d = np.size(u)  # ODE dimension
	n_samples = 2 #order of time steper is 2*n_samples
	epsilon_1 = 0.005
	delta = 0.001
	# Main loop
	print('Entering loop (gamma = '+str(gam)+').')
	tStart = time.time()
	tPlot = time.time()
	print("starting run for Nx=" +str(Nx_normal))
	# approx = kacewicz.odeint(
	# 	d, driver, ic_kac, t_list, "quad", n_samples, epsilon_1=epsilon_1, delta=delta, JAX=False, display_progress = True)
	approx,params = kacewicz.odeint(
		d, driver_jax, ic_kac, t_list, "quad", n_samples, epsilon_1=epsilon_1, delta=delta, JAX=True, display_progress = True, return_parameters=True)
	# print(params)
	# quit()
	final_euler_sol = approx[-1, :]
	u = np.reshape(final_euler_sol, (3, Nx))
	# Graphics initialisation
	if plots:
		figure, ax = plt.subplots(figsize=(10, 8))
		M = 1.1 * (np.max(np.abs(u[plots-1,:])) + 0.02)
		plt.axis([x[0], x[Nx-1], -M, M])
		line1, = ax.plot(x,u[plots-1,:],'b.-')
		plt.xlabel('x')
		plt.ylabel('u_'+str(plots))
		plt.title('t = '+str(t))
		plt.draw()

	
	
	tEnd = time.time()
	print("done with run for Nx=" + str(Nx_normal))
	print('Elapsed time is '+str(tEnd-tStart)+' seconds.')
	print('Terminated in '+str(n)+' iterations.')

	# Exact solution
	if pb == 'Density':
		utheo = lambda x: u0(x - Tf)
	elif pb == 'Riemann':
		UJ = np.array([rhoJ, rhoJ*uJ, 0.5*rhoJ*uJ**2 + pJ/(gam-1)])
		utheo = RiemannExact(UJ, gam, Tf)

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

	# print("starting to plot...")
	# plt.plot(x[::5],u[0,:][::5],"o",label="Numerical")
	# # utheo = lambda x: u0(x - (t+dt))
	# print(dt)
	# print(u[0,:][::5])
	# plt.plot(x[::5],utheo(x)[0][::5],label="Exact")
	# plt.xlabel(r"$x$")
	# plt.ylabel(r"$\rho(x,T)$")
	# plt.legend()
	# plt.savefig(title+"quantum1-" + str(Nx_normal) + "instance"+ str(test) + ".pdf")
	# plt.clf()
	# plt.plot(x[::5],u[1,:][::5],"o",label="Numerical")
	# #utheo = lambda x: u0(x - (t+dt))
	# plt.plot(x[::5],utheo(x)[1][::5],label="Exact")
	# plt.xlabel(r"$x$")
	# plt.ylabel(r"$\rho(x,T)u(x,T)$")
	# plt.legend()
	# plt.savefig(title+"quantum2-" + str(  Nx_normal) +"instance"+ str(test) + ".pdf")
	# plt.clf()
	# plt.plot(x[::5],u[2,:][::5],"o",label="Numerical")
  
	# plt.plot(x[::5],utheo(x)[2][::5],label="Exact")
	# plt.xlabel(r"$x$")
	# plt.ylabel(r"$E(x,T)$")
	# plt.legend()
	# # plt.savefig("classicalquantumwenop" + str(Nx) + ".pdf")
	# plt.savefig(title+"quantum3-" + str(  Nx_normal) +"instance"+ str(test) +".pdf")
	# plt.clf()
	print("starting to plot...")
	plt.plot(x,u[0,:],'ro',label="Numerical")
	# utheo = lambda x: u0(x - (t+dt))
	print(dt)
	print(u[0,:][::5])
	plt.plot(x,utheo(x)[0],'k',label="Exact")
	plt.xlabel(r"$x$")
	plt.ylabel(r"$\rho(x,T)$")
	plt.legend()
	plt.savefig(title+"quantum1-" + str(Nx_normal) + "instance"+ str(test) + ".pdf")
	plt.clf()
	plt.plot(x,u[1,:],'ro',label="Numerical")
	#utheo = lambda x: u0(x - (t+dt))
	plt.plot(x,utheo(x)[1],'k',label="Exact")
	plt.xlabel(r"$x$")
	plt.ylabel(r"$\rho(x,T)u(x,T)$")
	plt.legend()
	plt.savefig(title+"quantum2-" + str(  Nx_normal) +"instance"+ str(test) + ".pdf")
	plt.clf()
	plt.plot(x,u[2,:],'ro',label="Numerical")
  
	plt.plot(x,utheo(x)[2],'k',label="Exact")
	plt.xlabel(r"$x$")
	plt.ylabel(r"$E(x,T)$")
	plt.legend()
	# plt.savefig("classicalquantumwenop" + str(Nx) + ".pdf")
	plt.savefig(title+"quantum3-" + str(  Nx_normal) +"instance"+ str(test) +".pdf")
	plt.clf()

	# Error wrt. exact solution
	if pb == 'Density':
		uth = cellav(utheo, x, dx)
		ierr = (x>xlims[0])*(x<xlims[1])
		derr = u[:,ierr] - uth[:,ierr]
		one_err = np.array([np.linalg.norm(derr[0,:]*dx,1), np.linalg.norm(derr[1,:]*dx,1), np.linalg.norm(derr[2,:]*dx,1)])
		# two_err = np.array([np.linalg.norm(derr[0,:]*dx,2), np.linalg.norm(derr[1,:]*dx,2), np.linalg.norm(derr[2,:]*dx,2)])
		two_err = np.array([np.linalg.norm(derr[0,:]*math.sqrt(dx),2), np.linalg.norm(derr[1,:]*math.sqrt(dx),2), np.linalg.norm(derr[2,:]*math.sqrt(dx),2)])
		inf_err = np.array([np.linalg.norm(derr[0,:],np.inf), np.linalg.norm(derr[1,:],np.inf), np.linalg.norm(derr[2,:],np.inf)])
		print('L1, L2, Linf errors')
		print(np.array([one_err, two_err, inf_err]))
		return (one_err, two_err,inf_err, Nx_normal,tEnd-tStart,params['n_intervals'])

	return (None, None,None, None,tEnd-tStart,params['n_intervals'])



if __name__ == '__main__':
	k_min=4
	k_max=11
	rho_err_l1=[]
	rho_err_l2 = []
	rho_err_linf =[]

	u_err_l1=[]
	u_err_l2 = []
	u_err_linf =[]

	p_err_l1=[]
	p_err_l2 = []
	p_err_linf = []
	print("run starting.....")
	nx_list=[]
	full_error=[]
	time_list=[]
	secondary_list=[]
	if(test==1):
			file_name="weno_nature_run_smooth_average_quantum"
			title = "smooth"
			run_algorithm=1
			for k in range(k_min,k_max):
					# run(2**k)
					full_error.append(np.zeros(3))
					rho_err_l1.append(0.0)
					u_err_l1.append(0.0)
					p_err_l1.append(0.0)

					rho_err_l2.append(0.0)
					u_err_l2.append(0.0)
					p_err_l2.append(0.0)
					
					rho_err_linf.append(0.0)
					u_err_linf.append(0.0)
					p_err_linf.append(0.0) 
					
					for p in range(0,run_algorithm):
						out = run(2**k,title)
						(one_err,two_err,inf_err, nx,timedur,secondary) = out
						full_error[-1] = full_error[-1] + (np.array([one_err, two_err, inf_err]))
						time_list.append(timedur)
						secondary_list.append(secondary)
						nx_list.append(nx)

						rho_err_l1[-1] = rho_err_l1[-1] +one_err[0]
						u_err_l1[-1] = u_err_l1[-1] +one_err[1]
						p_err_l1[-1] = p_err_l1[-1] +one_err[2]


						rho_err_l2[-1] = rho_err_l2[-1] +two_err[0]
						u_err_l2[-1] = u_err_l2[-1] +two_err[1]
						p_err_l2[-1] = p_err_l2[-1] +two_err[2]
						


						rho_err_linf[-1] = rho_err_linf[-1] +inf_err[0]
						u_err_linf[-1] = u_err_linf[-1] +inf_err[1]
						p_err_linf[-1] = p_err_linf[-1] +inf_err[2]
						# rho_err_linf.append(inf_err[0])
						# u_err_linf.append(inf_err[1])
						# p_err_linf.append(inf_err[2])        
					full_error[-1] = (1.0/run_algorithm)*full_error[-1]
					rho_err_l1[-1] = (1.0/run_algorithm)*rho_err_l1[-1]
					u_err_l1[-1] = (1.0/run_algorithm)*u_err_l1[-1] 
					p_err_l1[-1] = (1.0/run_algorithm)*p_err_l1[-1] 


					rho_err_l2[-1] = (1.0/run_algorithm)*rho_err_l2[-1] 
					u_err_l2[-1] = (1.0/run_algorithm)*u_err_l2[-1] 
					p_err_l2[-1] = (1.0/run_algorithm)*p_err_l2[-1] 
					


					rho_err_linf[-1] = (1.0/run_algorithm)*rho_err_linf[-1]
					u_err_linf[-1] = (1.0/run_algorithm)*u_err_linf[-1] 
					p_err_linf[-1] = (1.0/run_algorithm)*p_err_linf[-1] 
			generate_convergence_history(rho_err_l1, nx_list, file_name, " rho L1")
			generate_convergence_history(u_err_l1, nx_list, file_name, " rhou L1")
			generate_convergence_history(p_err_l1, nx_list, file_name, " E L1")
			generate_convergence_history(rho_err_l2, nx_list, file_name, " rho L2")
			generate_convergence_history(u_err_l2, nx_list, file_name, " rhou L2")
			generate_convergence_history(p_err_l2, nx_list, file_name, " E L2")
			generate_convergence_history(rho_err_linf, nx_list, file_name, " rho Linf")
			generate_convergence_history(u_err_linf, nx_list, file_name, " rhou Linf")
			generate_convergence_history(p_err_linf, nx_list, file_name, " E Linf")
			gen_error(nx_list,full_error,time_list,secondary_list,"wenosmoothnature.txt")

	else:
			file_name="weno_nature_riemann"
			title = "riemann"
			for k in range(k_min,k_max):
					# run(2**k)
					print("solving riemann problem for Nx:" + str(2**k))
					out =run(2**k,title)
					(one_err,two_err,inf_err, nx,timedur,secondary) =out
					time_list.append(timedur)
					secondary_list.append(secondary)

			gen_error([],[],time_list,secondary_list,"wenoriemannnature.txt")

