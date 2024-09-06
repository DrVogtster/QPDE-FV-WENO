from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from functools import partial

import numpy as np

import os

import dill as pickle

from kacewicz_util_core import FORMAT

####################################################################################################

def jacobian(f, x, h = 1e-6, debug = False):
	"""
	Calculates the explicit jacobian of an arbitrary vector valued function using a second order
	central difference finite difference method.

	Arguments
	---------
	f : Callable
		A vector valued function that takes a vector argument 'x'
	x : Array
		The explicit vector argument to evaluate the jacobian at
	h : float, (Optional)
		Step sized used in the second order finite difference approximation of the partial
		derivatives
	debug : bool, (Optional)
		Toggles debugging print statements. Default value is 'False'.

	Returns
	-------
	jac : Array
		The explicit jacobian evaluated at the given point 'n'
	"""

	if (debug):
		pid = os.getpid()
		print(f"jacobain: - PID: {pid}",
			  f"({pid}) {FORMAT.bold_equality('f', f)}",
			  f"({pid}) {FORMAT.bold_equality('x', x)}",
			  f"({pid}) {FORMAT.bold_equality('h', h)}",
			  sep = "\n\t")

	#==============================================================================================#

	n = x.shape[0]
	J = np.zeros((n, n))

	for i in range(n):
		x_plus = np.copy(x)
		x_minus = np.copy(x)
		x_plus[i] += h
		x_minus[i] -= h
		f_plus = f(x_plus)
		f_minus = f(x_minus)
		J[:, i] = (f_plus - f_minus) / (2 * h)

	return J

def jacobian_part(pickled_f, x, i, mode = "multiprocessing", h = 1e-6, debug = False):
	"""
	Helper function evaluating the 'ith' column of the jacobian matrix of vector valued function
	'f'.

	Arguments
	---------
	f : Callable
		A callable vector valued function that takes a vector argument, 'x'. 'f' must be
		serializable i.e. it has to be able to be pickled.
	x : Array
		Specific array the jacobian is being evaluated at.
	i : int
		Index of the column of the jacobian matrix being calculated.
	h : float, (Optional)
		Step sized used in the second order central difference approximation of partial derivatives.
		Default value is '1E-6'.
	debug : bool, (Optional)
		Toggles debugging print statements. Default value is 'False'.

	Returns
	-------
	jac_part : Array
		'ith' column of the Jacobian matrix of 'f' evaluated at 'x'
	"""

	if (debug):
		if (mode == "MPI"):
			comm = MPI.COMM_WORLD
			size = comm.Get_size()
			rank = comm.Get_rank()
		else:
			current_process = multiprocessing.current_process()
		pid = os.getpid()
		
		print("Jacobian Part - " + \
			  (f"MPI_rank: {rank}, MPI_size: {size}, " if (mode == "MPI") else f"Name: {current_process.name}, ") + \
			  f"PID: {pid}",
			  f"({pid}) {FORMAT.bold_equality('pickled_f', type(pickled_f))}",
			  f"({pid}) {FORMAT.bold_equality('i', i)}",
			  f"({pid}) {FORMAT.bold_equality('x', x)}",
			  f"({pid}) {FORMAT.bold_equality('h', h)}",
			  sep = '\n')

	#==============================================================================================#

	f = pickle.loads(pickled_f)

	x_plus = x.copy()
	x_minus = x.copy()
	x_plus[i] += h
	x_minus[i] -= h
	f_plus = f(x_plus)
	f_minus = f(x_minus)

	return (f_plus - f_minus) / (2 * h)

def jacobian_pool(f, x, mode = "multiprocessing", h = 1e-6, debug = False):
	"""
	Calculates the jacobian matrix of 'f' using non-daemonic processes.

	Arguments
	---------
	f : Callable
		A vector valued function that is a function of a vector argument, 'x'. It is assumed that
		'f' is serializable, i.e. it can be pickled.
	x : Array
		The specific vector to evaluate the jacobian at.
	h : float, (Optional)
		Step size used in the second order central difference approximation of partial derivatives.
		By default, 'h = 1E-6'.
	debug : bool, (Optional)
		Toggles debuggig print statements. Default value is 'False'.

	Returns
	-------
	jac : Array
		Jacobian of 'f' evaluated at the given vector.
	"""

	if (debug):
		print("Jacobian Pool:",
			  f"{FORMAT.bold_equality('f', f)}",
			  f"{FORMAT.bold_equality('x', x)}",
			  f"{FORMAT.bold_equality('mode', mode)}",
			  f"{FORMAT.bold_equality('h', h)}",
			  sep = '\n\t')
	
	#==============================================================================================#

	pickled_f = pickle.dumps(f)
	jacobian_part_ = partial(jacobian_part, pickled_f, x, mode = mode, h = h, debug = debug)
	
	if (mode == "MPI"):
		executor = MPIPoolExecutor()
	elif (mode == "multiprocessing" or mode == "multiprocess"):
		executor = ProcessPoolExecutor()
	else: raise ValueError("Unrecognized parallelization method")

	indices = np.arange(len(x))

	results = executor.map(jacobian_part_, indices)
	jacobian_buffer = np.asarray(list(results)).T
	
	return jacobian_buffer

def Jacobian(f, mode, h = 1e-6, debug = False):
	if (debug):
		print("Jacobian",
			  f"{FORMAT.bold_equality('f', f)}",
			  f"{FORMAT.bold_equality('mode', mode)}",
			  f"{FORMAT.bold_equality('h', h)}",
			  sep = "\n\t")
	
	#==============================================================================================#

	if (mode == "serial"):
		return lambda x: jacobian(f, x, h = h, debug = debug)	
	else:
		return lambda x: jacobian_pool(f, x, mode = "MPI", h = h, debug = debug)

####################################################################################################

def derivative(A, t, h = 1e-6):
	"""
	Compute the first derivative of a function A(t) at a fixed t value using a second-order
	central difference approximation.

	Arguments
	---------
	A : Callable
		A callable function of scalar argument 't'.
	t : float
		The fixed value of t at which to compute the derivative at.
	h : float, (Optional)
		The time step size for the approximation.
    
	Returns
	-------
	dA_dt : Array
		The first derivative of matrix function A(t) at the fixed value of t.
    """

	# Compute A(t + h) and A(t - h)
	A_plus = A(t + h)
	A_minus = A(t - h)

	# Compute the first derivative using a second-order accurate approximation
	dA_dt = (A_plus - A_minus) / (2 * h)

	return dA_dt

def Derivative(A, h = 1e-6):
	return lambda t: derivative(A, t, h = h)

####################################################################################################

def Vectorize(f, in_axis = 0):
	return lambda x: np.apply_along_axis(f, in_axis, x)

####################################################################################################

def instantiate_integrand_function(driver_function, matrix = None, vectorize = True, JIT = None,
debug = False):
	if (debug):
		print("Instantiating Kaceicz Model Integrand Function:",
			  f"{FORMAT.bold_equality('driver_function', driver_function)}",
			  f"{FORMAT.bold_equality('matrix', matrix)}",
			  f"{FORMAT.bold_equality('vectorize', vectorize)}",
			  sep = "\n\t")

	#==============================================================================================#

	# Case 3: Time dependent matrix function
	if (callable(matrix)):
		matrix_derivative = Derivative(matrix)

		def integrand_function_(z):
			return np.dot(matrix_derivative(z[0]), z) + driver_function(z)

	# Cases 1 or 2: Identity matrix or non-trivial constant matrix
	else:
		def integrand_function_(z):
			return driver_function(z)

	#==============================================================================================#

	# Vectorize integrand function to be evaluated over a batch of 'z' values
	if (vectorize):
		integrand_function = Vectorize(integrand_function_, in_axis = 1)
		#integrand_function = np.vectorize(integrand_function_, signature = "(n)->(n)")
	else:
		integrand_function = integrand_function_

	if (debug):
		print(f"Instantiating Integrand Function {FORMAT.green('COMPLETE')}")

	return integrand_function

def instantiate_coefficient_functions(r, driver_function, matrix = None, mode = "serial",
JIT = None, debug = False):
	if (debug):
		print("Instantiating Coefficient Functions",
			  f"{FORMAT.bold_equality('r', r)}",
			  f"{FORMAT.bold_equality('driver_function', driver_function)}",
			  f"{FORMAT.bold_equality('matrix', matrix)}",
			  f"{FORMAT.bold_equality('mode', mode)}",
			  sep = "\n\t")
	
	#==============================================================================================#
	
	# Case 1: Time dependent matrix function
	if (callable(matrix)):
		def derivative_function(z):
			return np.linalg.solve(matrix(z[0]), driver_function(z))

	# Casees 2 and 3: constant matrix
	else:
		def derivative_function(z):
			return driver_function(z)

	#==============================================================================================#

	coefficient_functions = (derivative_function,)

	for i in range(1, r + 1):
		previous_coefficient_function = coefficient_functions[-1]
		next_coefficient_function = instantiate_next_coefficient_function(previous_coefficient_function,
																		  derivative_function, i + 1,
																		  mode = mode, debug = debug)

		coefficient_functions += (next_coefficient_function,)

	#==============================================================================================#

	if (debug):
		print(f"Instantiating Coefficient Functions {FORMAT.green('COMPLETE')}")

	return coefficient_functions

def instantiate_next_coefficient_function(previous_coefficient_function, derivative_function, n,
mode = "serial", h = 1e-6, debug = False):
	if (debug):
		print(f"Instantiating Next Coefficient Function {n}",
			  f"{FORMAT.bold_equality('previous_coefficient_function', previous_coefficient_function)}",
			  f"{FORMAT.bold_equality('derivative_function', derivative_function)}",
			  f"{FORMAT.bold_equality('n', n)}",
			  f"{FORMAT.bold_equality('mode', mode)}",
			  f"{FORMAT.bold_equality('h', h)}",
			  sep = "\n\t")

	#==============================================================================================#

	previous_coefficient_jacobian = Jacobian(previous_coefficient_function, mode, h = h, debug = debug)

	def next_coefficient_function(z):
		return previous_coefficient_jacobian(z) @ derivative_function(z) / n

	#==============================================================================================#

	if (debug):
		print(f"Instantiating Coefficient Function {n} {FORMAT.green('COMPLETE')}")
	
	return next_coefficient_function

def compose_coefficients(coefficient_functions, seed, JIT = None):
	return np.asarray([seed] + [f(seed) for f in coefficient_functions])

def taylor_(t, t0, C):
	powers = (t - t0) ** np.arange(len(C))

	return np.dot(powers, C)
taylor_vec_ = np.vectorize(taylor_, excluded = {1, 2}, signature = "()->(n)")
def taylor(t, t0, C, JIT = None):
	if (hasattr(t, "__len__")): return taylor_vec_(t, t0, C)

	return taylor_(t, t0, C)


def test_ODE(matrix = True, N = 2, r = 2, test_functions = False):
	problem = {"dimension": N + 1}
	
	#==============================================================================================#

	# Manufactrued Solution Function
	def sol_fun(t):
		return np.asarray([t] + [np.sin(i * t)
								  if (i % 2 == 1)
								  else np.cos(i * t)
								  for i in range(1, N + 1)])

	problem["solution"] = sol_fun
	problem["solution_function"] = sol_fun
	
	#==============================================================================================#

	# ODE Matrix
	if (matrix):
		def matrix_fun(t):
			D = np.asarray([[np.sin(i * t) * np.cos(j * t) + 1
							 if (i == j)
							 else np.sin(i * t) * np.cos(j * t)
							 for j in range(1, N + 1)]
							for i in range(1, N + 1)])
			
			return np.block([[np.ones(1), np.zeros(N)], [np.zeros((N, 1)), D]])
	else: matrix_fun = None
	problem["matrix"] = matrix_fun

	#==============================================================================================#
	
	# Non-source component
	def non_source_fun(z):
		return np.asarray([0.] + [z[i] ** 2 for i in range(1, N + 1)])

	#==============================================================================================#
	
	# Derivative Function
	def derivative_fun(t):
		return np.asarray([1.] + [i * np.cos(i * t)
								  if (i % 2 == 1)
								  else -i * np.sin(i * t)
								  for i in range(1, N + 1)])

	#==============================================================================================#

	# Source Function
	if (matrix):
		def source_fun(t):
			return matrix_fun(t) @ derivative_fun(t) - non_source_fun(sol_fun(t))
	else:
		def source_fun(t):
			return derivative_fun(t) - non_source_fun(sol_fun(t))
	
	#==============================================================================================#

	def driver_fun(z):
		return non_source_fun(z) + source_fun(z[0])
	problem["driver"] = driver_fun

	#==============================================================================================#

	if (test_functions):
		# Test Integrand Function
		def true_int_fun_(t):
			return np.asarray([1.] + [np.sin(i * t) ** 2
									  if (i % 2 == 1)
									  else np.cos(i * t) ** 2
									  for i in range(1, N + 1)])

		if (matrix):
			def matrix_deriv(t):
				D = np.asarray([[i * np.cos(i * t) - j * np.sin(j * t)
								 for j in range(1, N + 1)]
								for i in range(1, N + 1)])

				return np.block([[np.zeros(1), np.zeros(N)], [np.zeros((N, 1)), D]])

			def true_int_fun(t):
				return matrix_deriv(t) @ sol_fun(t) + true_int_fun_(t)
			
			problem["true_integrand_function"] = true_int_function
		else:
			problem["true_integrand_function"] = true_int_fun_
		
		##----------------------------------------------------------------------------------------##

		# Coefficient Functions
		factorial = [1]
		for i in range(r + 1): factorial.append(factorial[-1] * (i + 1))

		true_coefficient_functions = [lambda t: np.asarray([1.] + [i * np.cos(i * t)
																   if (i % 2 == 1)
																   else -i * np.sin(i * t)
																   for i in range(1, N + 1)])]
		for m in range(2, r + 1):
			if (m % 4 == 0):
				true_coefficient_functions.append(lambda t: np.asarray([0.] + [i ** m * np.sin(i * t)
																			   if (i % 2 == 1)
																			   else i ** m * np.cos(i * t)
																			   for i in range(1, N + 1)]) / factorial[m])
		
			elif (m % 4 == 1):
				true_coefficient_functions.append(lambda t: np.asarray([0.] + [i ** m * np.cos(i * t)
																			   if (i % 2 == 1)
																			   else i ** m * -np.sin(i * t)
																			   for i in range(1, N + 1)]) / factorial[m])
			
			elif (m % 4 == 2):
				true_coefficient_functions.append(lambda t: np.asarray([0.] + [i ** m * -np.sin(i * t)
																			   if (i % 2 == 1)
																			   else i ** m * -np.cos(i * t)
																			   for i in range(1, N + 1)]) / factorial[m])
			
			else:
				true_coefficient_functions.append(lambda t: jnp.asarray([0.] + [i ** m * -np.cos(i * t)
																				if (i % 2 == 1)
																				else i ** m * np.sin(i * t)
																				for i in range(1, N + 1)]) / factorial[m])

		problem["true_coefficient_functions"] = true_coefficient_functions

	return problem

####################################################################################################

if (__name__ == "__main__"):
	def driver(x): return np.asarray([x[0] ** 2 + x[1], 2 * x[0] ** 2 + x[1] ** 2])
	x = np.array([1., 1.])
	h = 1e-7
	alpha = 10
	debug = True

	#==============================================================================================#

	# Jacobians using serial evaluation
	j_driver = Jacobian(driver, "serial", debug = debug)
	print("> Jacobian Driver - Serial:", j_driver(x))

	def first_jvp(x):
		return j_driver(x) @ driver(x)
	print("> First JVP - Serial:", first_jvp(x))

	j_first_jvp = Jacobian(first_jvp, "serial", debug = debug)
	print("> Jacobian First JVP - Serial:", j_first_jvp(x))

	def second_jvp(x):
		return j_first_jvp(x) @ driver(x)
	print("> Second JVP - Serial:", second_jvp(x))

	#==============================================================================================#

	# Jacobians using multiprocessing pools
	j_driver = Jacobian(driver, "multiprocessing", h = h, debug = debug)
	print("> Jacobian Driver - Pool:", j_driver(x))
	
	def first_jvp(x):
		return j_driver(x) @ driver(x)
	print("> First JVP - Pool:", first_jvp(x))

	j_first_jvp = Jacobian(first_jvp, "multiprocessing", h = h * alpha, debug = debug)
	print("> Jacobian First JVP - Pool:", j_first_jvp(x))

	def second_jvp(x):
		return j_first_jvp(x) @ driver(x)
	print("> Second JPV - Pool:", second_jvp(x))
	
	#==============================================================================================#

	# Jacobians using MPI process pools
	j_driver = Jacobian(driver, "MPI", h = h, debug = debug)
	print("> Jacobian Driver - MPI:", j_driver(x))

	def first_jvp(x):
		return j_driver(x) @ driver(x)
	print("> First JVP - MPI:", first_jvp(x))

	j_first_jvp = Jacobian(first_jvp, "MPI", h = h * alpha, debug = debug)
	print("> Jacobian First JVP - MPI:", j_first_jvp(x))

	def second_jvp(x):
		return j_first_jvp(x) @ driver(x)
	print("> Second JVP - MPI:", second_jvp(x))

	#==============================================================================================#

	def A(t):
		return np.array([[np.sin(np.pi * t / 2.0), t ** 5],
						 [t ** 10, np.cos(np.pi * t / 2.0)]])
	t = 1.0
	dA = Derivative(A)
	print("> Matrix Derivative:", dA(t))

	print("COMPLETE")
