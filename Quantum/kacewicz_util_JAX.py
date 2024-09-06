import jax
from jax import numpy as jnp
jax.config.update("jax_enable_x64", True)

from functools import partial

from kacewicz_util_core import FORMAT

####################################################################################################

def Derivative(A):
	return jax.jacobian(A)

def JIT(f, static_argnums = None, static_argnames = None):
	return jax.jit(f, static_argnums = static_argnums, static_argnames = static_argnames)

####################################################################################################

def instantiate_integrand_function(driver_function, matrix = None, vectorize = True, JIT = True,
debug = False):
	"""
	Instantiates the integrand function used in the Kacewicz integral step. The procedure of how
	the integrand function is instantiated will depend on the values of keyword arguments. See
	Notes section.

	Notes
	-----
	JIT : When True, the integrand function is wrapped with the JAX JIT transformation
	vectorize: 

	Arguments
	---------
	driver_function : Callable[[jax.Array], jax.Array]
		Callable driver function of the autonomous system of ODEs. Expected to be JAX
		compatible.
	matrix : Union[Callable[[jax.Array], jax.Array], Array, None], (Optional)
		If the leading matrix in the autonomous system is the identity matrix, this should be
		None. If the leading matrix is a non-trivial constant matrix, this should be the matrix
		represented by an Array. If the leading matrix is time dependent, is expectd that this
		is JAX compatible. Default value is None, indicating the identity matrix.
	JIT : bool, (Optional)
		Toggles whether or not the integrand function is JIT compiled using the JAX JIT function.
		Default value is True.
	debug : bool, (Optional)
		Boolean flag that toggles debug print statements. Default value is False.
	
	Returns
	-------
	F : Callable[[jax.Array], jax.Array]
		Integrand function used in the Kaceiwcz integral step.
	"""

	if (debug):
		print("Instantiating Integrand Function:",
			  f"{FORMAT.bold_equality('driver_function', driver_function)}",
			  f"{FORMAT.bold_equality('matrix', matrix)}",
			  f"{FORMAT.bold_equality('vectorize', vectorize)}",			  
			  f"{FORMAT.bold_equality('JIT', JIT)}",
			  sep = "\n\t")

	#==============================================================================================#

	# Case 3: Time dependent matrix function
	if (callable(matrix)):
		matrix_derivative = jax.jacobian(matrix)
		
		def integrand_function_(z):
			return jnp.dot(matrix_derivative(z[0]), z) + driver_function(z)

	# Cases 1 or 2: Identity matrix or non-trivial constant matrix
	else:
		def integrand_function_(z):
			return driver_function(z)

	#==============================================================================================#

	if (vectorize):
		integrand_function = jax.vmap(integrand_function_, in_axes = 0, out_axes = 0)

	#==============================================================================================#

	if (JIT):
		if (vectorize): integrand_function = jax.jit(integrand_function)
		else: integrand_function = jax.jit(integrand_function_)
		
	#==============================================================================================#

	if (debug):
		print(f"Instantiating Integrand Function: {FORMAT.green('COMPLETE')}")

	return integrand_function

def instantiate_next_coefficient_function(previous_coefficient_function, derivative_function, n,
debug = False):
	if (debug):
		print(f"Instantiating Next Coefficient Function {n}:",
			  f"{FORMAT.bold_equality('previous_coefficient_function', previous_coefficient_function)}",
			  f"{FORMAT.bold_equality('derivative_function', derivative_function)}",
			  f"{FORMAT.bold_equality('n', n)}",
			  sep = "\n\t")

	def next_coefficient_function(z):
		return jax.jvp(previous_coefficient_function, (z,), (derivative_function(z),))[1] / n

	if (debug):
		print(f"Instantiating Next Coefficient Function {n}: {FORMAT.green('COMPLETE')}")
	
	return next_coefficient_function

def instantiate_coefficient_functions(r, driver_function, matrix = None, mode = "serial",
JIT = False, debug = False):
	if (debug):
		print("Instantiating Coefficient Functions:",
			  f"{FORMAT.bold_equality('r', r)}",
			  f"{FORMAT.bold_equality('driver_function', driver_function)}",
			  f"{FORMAT.bold_equality('matrix', matrix)}",
			  f"{FORMAT.bold_equality('mode', mode)}",
			  f"{FORMAT.bold_equality('JIT', JIT)}",
			  sep = "\n\t")

	#==============================================================================================#

	# Case 1: Callable time dependent matrix
	if (callable(matrix)):
		def derivative_function(z):
			return jnp.linalg.solve(matrix(z[0]), driver_function(z))

	# Cases 2 and 3: Constant matrix
	else: derivative_function = driver_function

	#==============================================================================================#

	coefficient_functions = (derivative_function,)

	for i in range(1, r + 1):
		previous_coefficient_function = coefficient_functions[-1]
		next_coefficient_function = instantiate_next_coefficient_function(previous_coefficient_function,
																		  derivative_function,
																		  i + 1,
																		  debug = debug)
		coefficient_functions += (next_coefficient_function,)

	if (JIT): coefficient_functions = tuple(jax.jit(f) for f in coefficient_functions)

	if (debug): print(f"Instantiating Coefficient Functions: {FORMAT.green('COMPLETE')}")

	return coefficient_functions

def compose_coefficients_(coefficient_functions, seed):
	return jnp.asarray([seed] + [f(seed) for f in coefficient_functions])
compose_coefficients_JIT_ = jax.jit(compose_coefficients_, static_argnums = (0,))
def compose_coefficients(coefficient_functions, seed, JIT = False):
	if (JIT): return compose_coefficients_JIT_(coefficient_functions, seed)

	else: return compose_coefficients_(coefficient_functions, seed)

def taylor_(t, t0, C):
	powers = (t - t0) ** jnp.array(range(len(C)))

	return jnp.dot(powers, C)
taylor_JIT_ = jax.jit(taylor_)
taylor_vec_ = jax.jit(jax.vmap(taylor_, in_axes = (0, None, None), out_axes = 0))
def taylor(t, t0, C, JIT = False):
	if (hasattr(t, "__len__")): return taylor_vec_(t, t0, C)

	if (JIT): return taylor_JIT_(t, t0, C)
	return taylor_(t, t0, C)

####################################################################################################

def pdf(x, M, omega):
	angle = M * omega - x

	return jnp.sin(jnp.pi * angle) ** 2 / jnp.sin(jnp.pi * angle / M) ** 2 / M ** 2
pdf_JIT = jax.jit(pdf)
pdf_vec = jax.jit(jax.vmap(pdf, in_axes = (0, None, None), out_axes = 0))

# Test Problems
def test_ODE(matrix = True, N = 2, r = 2, test_functions = False):
	problem = {"dimension": N + 1}
	
	#==============================================================================================#

	# Manufactrued Solution Function
	def sol_fun(t):
		return jnp.asarray([t] + [jnp.sin(i * t)
								  if (i % 2 == 1)
								  else jnp.cos(i * t)
								  for i in range(1, N + 1)])

	problem["solution"] = sol_fun
	problem["solution_function"] = sol_fun
	
	#==============================================================================================#

	# ODE Matrix
	if (matrix):
		def matrix_fun(t):
			D = jnp.asarray([[jnp.sin(i * t) * jnp.cos(j * t) + 1
							  if (i == j)
							  else jnp.sin(i * t) * jnp.cos(j * t) \
							  for j in range(1, N + 1)]
							 for i in range(1, N + 1)])
			
			return jnp.block([[jnp.ones(1), jnp.zeros(N)], [jnp.zeros((N, 1)), D]])
	else: matrix_fun = None
	problem["matrix"] = matrix_fun

	#==============================================================================================#
	
	# Non-source component
	def non_source_fun(z):
		return jnp.asarray([0.] + [z[i] ** 2 for i in range(1, N + 1)])

	#==============================================================================================#
	
	# Derivative Function
	def derivative_fun(t):
		return jnp.asarray([1.] + [i * jnp.cos(i * t)
								   if (i % 2 == 1)
								   else -i * jnp.sin(i * t)
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
		def proto_true_int_fun(t):
			return jnp.asarray([1.] + [jnp.sin(i * t) ** 2
									   if (i % 2 == 1)
									   else jnp.cos(i * t) ** 2
									   for i in range(1, N + 1)])

		if (matrix):
			def matrix_deriv(t):
				D = jnp.asarray([[i * jnp.cos(i * t) - j * jnp.sin(j * t)
								  for j in range(1, N + 1)]
								 for i in range(1, N + 1)])

				return jnp.block([[jnp.zeros(1), jnp.zeros(N)], [jnp.zeros((N, 1)), D]])

			def true_int_fun(t):
				return matrix_deriv(t) @ sol_fun(t) + proto_true_int_fun(t)
			
			problem["true_integrand_function"] = true_int_function
		else:
			problem["true_integrand_function"] = proto_true_int_fun
		
		##----------------------------------------------------------------------------------------##

		# Coefficient Functions
		factorial = [1]
		for i in range(r + 1): factorial.append(factorial[-1] * (i + 1))
		
		true_coefficient_functions = [lambda t: jnp.asarray([1.] + [i * jnp.cos(i * t)
																	if (i % 2 == 1)
																	else -i * jnp.sin(i * t)
																	for i in range(1, N + 1)])]
		for m in range(2, r + 1):
			if (m % 4 == 0):
				true_coefficient_functions.append(lambda t: jnp.asarray([0.] + [i ** m * jnp.sin(i * t)
																				if (i % 2 == 1)
																				else i ** m * jnp.cos(i * t)
																				for i in range(1, N + 1)]) / factorial[m])
		
			elif (m % 4 == 1):
				true_coefficient_functions.append(lambda t: jnp.asarray([0.] + [i ** m * jnp.cos(i * t)
																				if (i % 2 == 1)
																				else i ** m * -jnp.sin(i * t)
																				for i in range(1, N + 1)]) / factorial[m])
			
			elif (m % 4 == 2):
				true_coefficient_functions.append(lambda t: jnp.asarray([0.] + [i ** m * -jnp.sin(i * t)
																				if (i % 2 == 1)
																				else i ** m * -jnp.cos(i * t)
																				for i in range(1, N + 1)]) / factorial[m])
			
			else:
				true_coefficient_functions.append(lambda t: jnp.asarray([0.] + [i ** m * -jnp.cos(i * t)
																				if (i % 2 == 1)
																				else i ** m * jnp.sin(i * t)
																				for i in range(1, N + 1)]) / factorial[m])

		problem["true_coefficient_functions"] = true_coefficient_functions

	return problem

####################################################################################################

if (__name__ == "__main__"):
	print("COMPLETE")