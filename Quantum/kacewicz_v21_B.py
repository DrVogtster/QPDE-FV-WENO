import os # Operatin System module to print colored console outputs and make directories
os.system("") # Used to display colored console text
import numpy as np
from json import load, dump # Kacewicz model parameters are saved from and loaded to .json files
from pandas import read_csv # Kacewicz model coefficients can be saved to .csv files
import time # Time module used to determine how long processes take
import psutil # Process Utility for checking memory size
import dill as pickle

from matplotlib import pyplot as plt # Graph plotting module for debugging and visualization
from matplotlib.widgets import Button

from kacewicz_util_core import *

####################################################################################################

"""
The Kacewicz module is designed to operate with two distinct methods of differentiation. The default
method is using JAX autodifferentiation.
"""

# Boolean flag determining which method of differentiation to use, JAX Autodifferentiation or Finite
# Difference differentiation
JAX_FLAG = None

def use_JAX_util():
	"""
	Swtiches the kacewicz module into using JAX Autodifferentiation
	"""
	global JAX_FLAG
	global util	

	# Kacewicz module is already using JAX
	if (JAX_FLAG == True): return

	print("Kacewicz module switching to JAX Autodifferentiation mode...")
	import kacewicz_util_JAX as util
	JAX_FLAG = True

def use_FD_util():
	"""
	Switches the Kaceiwcz Module into using Finite Difference differentiation
	"""
	global JAX_FLAG
	global util	

	# Kacewicz module is already using Finite Difference
	if (JAX_FLAG == False): return
	
	print("Kacewicz module switching to Finite Difference differentiation mode...")
	import kacewicz_util_FD as util
	JAX_FLAG = False

# When module is being loaded, attempt to initialize using JAX Autodifferentiation
try: use_JAX_util()
except ImportError:
	print("Failed to load JAX module(s). Falling back on Finite Difference mode.")
	use_FD_util()

####################################################################################################

class kacewicz():
	"""
	The Kacewicz model is a Class that contains all the functionality and features for solving an
	Autonomous System of Ordinary Differential Equations (ASODEs) using a simulation of the Kacewicz
	algorithm. This includes functionality to load/save the parameters and approximation
	coefficients of a fully instantiated model to/from files.

	Notes
	-----
	The Kacewicz Class can solve ASODEs of the following form.

	A(z[0]) * dz/dt = f(z)
	z_0 = z(0)
		
	z(t) : State Vector
		-This is a vector valued function that is the solution to the ASODEs.
		-The first componnet of the State Vector, 'z[0]', MUST be the time parameter 't'.
	A(z[0]) : leading Matrix
		-This can be the identity matrix, an arbitrary constant, square matrix, or a time dependent,
		square matrix.
		-Because the first component of the State Vector, 'z[0]', is the time parameter, 't', the
		upper left matrix element must be '1.0' with zeros along the remaining first row and column.
	f(z) : Driver Function
		-This is the vector valued ASODEs driver function
		-Becuase the first component of the State Vector, 'z[0]', must be the time paramter, 't',
		the first component of the driver function, 'f(z)[0]', must always be one.
	z_0 : Initial Condition
		-The autonomous system of ODEs initial condition
		-Because the first component of the State Vector, 'z[0]', must be the time parameter, 't',
		the first component of the Initial Condition, 'z_0[0]', must be the initial time.

	Kacewicz solves this ASODEs by partitioning the time domain into primary and secondry intervals,
	and then generating a Taylor series approximation over each secondary interval.

	Each Taylor series approximation is of degree 'r + 2' where 'r' is an integer parameter. To
	facilitate the continuity of the approximation over a primary interval, 'r + 1' coefficient
	functions are instantiated and used to calculate the Taylor series coefficients.

	At the end of each primary interval, an integral is used to calculate the value of the State
	Vector at the start of the next primary interval. This is the Kacewciz inegral step, and is
	where a quantum integral algorithm would be implemented.

	In this model, the quantum component comes from the use of a Quantum Amplitude Estimation
	Algorithm (QAEA) to find the approxiamte integral of a derived Integrand Function. The QAEA uses
	a Monte Carlo algorithm to simulate the probabilistic behavior of a true quantum algorithm.
	"""

	# Kaceiwcz model supported integral modes
	supported_integrals = ["riemann", "quad"]

	def model(r, d, driver_function, matrix = None, JAX = True, JIT = True, vectorize = True,
	parallelize = True, n = None, k = None, epsilon_1 = None, min_n_intervals = None, debug = False):
		"""
		Initializes an instance of the Kacewicz model, instantiating all Kaceiwcz functions and the
		number of primary and secondary intervals.

		Notes
		-----
		Leading Matrix:
			-See Kacewicz.set_functions() Docstring.
		
		JAX:
			-See Kacewicz.set_functions() Docstring.
		JIT:
			-See Kacewicz.set_functions() Docstring.

		Epsilon Condition:
			-Given a parameter 'epsilon_1', a number strictly between zero and one, this condition
			bounds the normalized integral error of the the Kacewicz integral step to be less than
			or equal to 'epsilon_1'.
			-The normalzed integral error is the error in the Kacewicz integral step where the
			domain and co-domain of the integrand function have been normalized to be between zero
			and one.

		Arguments
		---------
		r : int
			-Kacewicz order parameter used to determine the number of terms in the Taylor series
			approximations.
			-See Kacewicz Class Docstring.
			-Expected to be a positive integer.
		d : int
			-This is the dimensionality of the ASODEs, also interpreted as the number of equations
			in the ASODEs.
			-The state vector, 'z', output of the driver function, 'f(z)', and initial condition,
			'z_0', should all have 'd' components.
			-Currently, there is no explicit way of enforcing this constraint directly on the ASODEs
			driver and matrix functions.
			-Just make sure it is correct or else you will get an Array boradcasting error due to
			mismatched shapes.
		driver_function : Union[Callable[[numpy.ndarray], numpy.ndarray],
								Callable[[jax.Array], jax.Array]]
			-ASODEs driver function with the State Vector, 'z', as its only argument.
			-See Kacewicz Class Docstring.
		matrix : Union[None,
					   Union[jax.Array, numpy.ndarray],
					   Union[Callable[[float], numpy.ndarray], Callable[[float], jax.Array]]],
				 (Optional)
			-Representation of the ASODEs Leading Matrix.
			-See notes section.
			-Default value is 'None'.
		
		JAX : bool, (Optional)
			-Toggles the use of JAX Autodifferentiation.
			-See notes section.
			-Default value is 'True'.
		JIT : bool, (Optional)
			-Toggles JIT compiling transformation.
			-See notes section.
			-Default value is 'True'.
		vectorize : bool, (Optional)

		The number of primary and secondary intervals can be determined in one of two ways.
		Case 1:
			-Specifying the number of primary intervals, 'n', and the secondary interval
			parameter, 'k', directly.
			-The Kacewics model will have 'n ** (k - 1)' secondary intervals PER primary interval
			with a total of 'n ** k' total secondary intervals.
		Case 2:
			-Calculating the number of primary and secondry intervals by satisfying a normalized
			bounded erorr condition called the Epsilon Condition.
			-See notes section.

		n : int, (Optional)
			-The number of primary intervals.
			-Expected to be a positive, non-zero integer.
			-Default value is '0'.
		k : int, (Optional)
			-The secondary interval parameter.
			-Expected to be a positive non-zero integer.
			-Default value is '0'.
		
		If 'n' and 'k' are not specified, or are invalid values, the kacewicz epsilon condition will
		be used to determine the number of primary and secondary inervals.
		
		epsilon_1 : float, (Optional)
			-This is the error parameter in the Epsilon Condition.
			-See notes section.
			-Expected to be a number strictly between zero and one.
			-Default value is '0.0'.
		min_n_intervals : int, (Optional)
			-The minimum number of secondary intervals needed to satisfy any ASODEs stability
			conditions.
			-Expected to be a positive non-zero integer.
			-Default value is '0'.
		
		Returns
		-------
		model : kacewicz model
			Returns a new instnace of a kacewicz model
		"""

		##----------------------------------------------------------------------------------------##

		# Each new model instance can change the Kaceiwcz differentiation mode
		if (JAX and not JAX_FLAG): use_JAX_util()
		if (not JAX and JAX_FLAG): use_FD_util()

		##----------------------------------------------------------------------------------------##

		model = kacewicz()
		setattr(model, "debug", debug)

		# Instantiate Kacewicz integrand function and the 'r + 1' coefficient functions
		model.set_functions(r, d, driver_function, matrix = matrix, vectorize = vectorize,
							parallelize = parallelize, JAX = JAX, JIT = JIT)

		# Set Kacewicz primary and secondary intervals
		model.set_intervals(n = n, k = k, epsilon_1 = epsilon_1, min_n_intervals = min_n_intervals)

		return model
	
	#==============================================================================================#

	def get_parameters(self):
		"""
		Collects all important Kacewicz model parameters into a dictionary. The model instance is
		assumed to have been fully initialized and run. This is primarily used for saving the model
		parameters to file, allowing the model to be loaded from file in the future.

		Returns
		-------
		x : dict[str, Any]
			A dictionary of kacewicz model parameters with string keys.
		"""

		parameters_dict = dict()
		
		parameters_keys = ["r", "d", "n", "k", "N_k", "n_intervals", "interval_size", "epsilon_1",
						   "min_n_intervals", "tlims", "integral_mode", "n_samples", "quantum",
						   "hilbert_dimension", "delta"]

		for key in parameters_keys:
			# For easier parsing and storage in parameters files, we separate the limits of the time
			# domain
			if (key == "tlims"):
				tlims = getattr(self, "tlims")
				parameters_dict["t0"] = tlims[0]
				parameters_dict["t1"] = tlims[1]
				continue
			
			parameters_dict[key] = getattr(self, key)

		return parameters_dict

	#==============================================================================================#

	def load_model(parameters_file_name, debug = False):
		"""
		Instantiates new kacewicz model from parameters and coefficients files.

		Arguments
		---------
		parameters_file_name : str
			-Path to the parameters file.
			-Assumed to be a '.json' file with all appropriate key-value pairs.
			-The path to the coefficients file should be one of the parameters.
		debug : bool, (Optional)
			-Toggles debugging print statements.
			-Default value is 'False'.

		Returns
		-------
		model : kacewicz model
			Returns a newly instantiated kacewicz model instance with parameters and coefficients
			loaded from file.
		"""
		
		model = kacewicz()

		model.load(parameters_file_name, debug = debug)

		return model

	def load(self, parameters_file_name, debug = False):
		"""
		Loads kacewicz model parameters and coefficients from files. This will replace any existing
		model parameters and coefficients. The kacewicz model parameters must be stored in a .json
		parameters file with appropriately named keys-value pairs.

		Arguments
		---------
		paramters_file_name : str
			-Path to parameters file with .json file extension.
		debug : bool, (Optional)
			-Toggles debugging print statements.
			-Default value is 'False'.
		"""

		if (debug):
			print("Loading Kacewicz Model:",
				  f"{FORMAT.bold_equality('parameters_file_name', parameters_file_name)}",
				  sep = "\n\t")

		parameters = kacewicz.load_parameters(parameters_file_name, debug = debug)

		# Recombine the t0 and t1 keys into the kacewicz model time domain tuple
		self.tlims = (parameters.pop("t0"), parameters.pop("t1"))

		# The coefficients file name is the only key in the parameters file that is not loaded into the model
		coefficients_file_name = parameters.pop("coefficients_file_name")

		for key in list(parameters.keys()):
			setattr(self, key, parameters.pop(key))

		coefficients = kacewicz.load_coefficients(coefficients_file_name,
												  self.n_intervals,
												  self.r,
												  self.d,
												  debug = debug)

		self.coefficients = coefficients

		if (debug):
			print(f"Loading Kacewicz Model: {FORMAT.green('COMPLETE')}")

	def load_parameters(parameters_file_name, debug = False):
		"""
		Loads kacewicz model parameters from a .json file to a dictionary. The parameters file must
		have appropriately named key-value pairs.
		
		Arguments
		---------
		parameters_file_name : str
			-Path to parameters file with a .json file extension.
		debug : bool, (Optional)
			-Toggles debugging print statements.
			-Default value is 'False'.

		Returns
		-------
		x : dict[str, Any]
			Dictionary of kacewicz model parameters loaded from file.
		"""
		
		if (debug):
			print("Loading Kacewicz Model Parameters from File:",
				  f"{FORMAT.bold_equality('parameters_file_name', parameters_file_name)}",
				  sep = "\n\t")

		parameters_file = open(parameters_file_name, "r")
		parameters = load(parameters_file)
		parameters_file.close()

		if (debug):
			print(f"Loading Kacewicz Model Parameters from File: {FORMAT.green('COMPLETE')}")

		return parameters

	def load_coefficients(coefficients_file_name, n_intervals, r, d, debug = False):
		"""
		Loads the approximation coefficients of a previously existing kacewicz model from a
		coefficients file. It is assumed that the coefficients file contains all necessary data to
		generate the coefficients array of shape ('n_intervals', 'r + 2', 'd').

		Arguments
		---------
		coefficients_file_name : str
			-Path to the coefficients file.
			-The coefficients file can be one of three accepted file formats: .txt, .npy or .npz
		n_intervals : int
			-Number of total secondary intervals, i.e. the number of Taylor series approximations.
			-Expected to be a positive, non-zero integer.
		r : int
			-Kacewicz order parameter determining the number of terms in each Taylor series.
			-Expecting 'r + 2' terms in each Taylor series approximation.
			-Should be a positive integer.
		d : int
			-The dimensionalit of the ASODEs.
			-Expected to be a positive, non-zero integer.
		debug : bool, (Optional)
			-Boolean flag that toggles debugging print statements.
			-Default value is 'False'.
		
		Returns
		-------
		C : Array[float]
			-Numerical Array of kacewicz model coefficients in the shape of 
			('n_samples', 'r + 2', 'd')
		"""
		
		if (debug):
			print("Loading Kacewicz Model Coefficients from File",
				  f"{FORMAT.bold_equality('coefficients_file_name', coefficients_file_name)}",
				  f"{FORMAT.bold_equality('n_intervals', n_intervals)}",
				  f"{FORMAT.bold_equality('r', r)}",
				  f"{FORMAT.bold_equality('d', d)}",
				  sep = "\n\t")

		# Save to Numpy binary file
		if (coefficients_file_name[-3:] == "npy"):
			coefficients_file = open(coefficients_file_name, "rb")
			coefficients = np.reshape(np.load(coefficients_file), (n_intervals, r + 2, d))
		
		# Save to plain text file
		elif (coefficients_file_name[-3:] == "txt"):
			coefficients_file = open(coefficients_file_name, "r")
			coefficients = read_csv(coefficients_file, delimiter = ' ', header = None).to_numpy().reshape((n_intervals, r + 2, d))
			#self.coefficients = np.reshape(np.genfromtxt(self.coefficients_file_name), (self.n_intervals, self.r + 2, self.d))
		
		# Save to compressed Numpy binary file
		elif (coefficients_file_name[-3:] == "npz"):
			coefficients_file = open(coefficients_file_name, "r")
			coefficients = np.load(coefficients_file)
		
		else:
			raise ValueError(f"Unrecognized file extension in coefficinets file name \"{coefficients_file_name}\"")	
		
		coefficients_file.close()
		
		if (debug):
			print(f"Loading Kacewicz Model Coefficients from File {FORMAT.green('COMPLETE')}")
		
		return coefficients

	#==============================================================================================#

	def set_functions(self, r, d, driver_function, matrix = None, JAX = True, JIT = True,
	vectorize = True, parallelize = True,  debug = False):
		"""
		Instantiates and sets the Kacewicz model integrand function and all 'r + 1' coefficient
		functions.

		Notes
		-----
		Leading Matrix:
			-If the Leading Matrix of the ASODEs is the identity matrix, 'matrix' should be 'None'.
			This shortcuts matrix multiplication.
			-If the Leading Matrix is a constant, square matrix, 'matrix' should be represented as
			an Array.
			-If the Leading Matrix is time dependent, 'matrix' should be a callable function that
			accepts a scalar argument and returns a 2D 'd' by 'd' Array. If using JAX
			Autodifferentiation, this should be a JAX traceable function.
		
		JAX:
			-When using JAX Autodifferentation, automatic jacobians are used to calculate the
			derivatives of the driver function. This is used to generate the coefficient functions
			that calculate the coefficients of the Taylor series approximations.
			-This requries that all input functions are JAX traceable.
		JIT:
			-JAX traceable functions in JAX Autodifferentiation mode can be Just In Time (JIT)
			compiled to allow for faster run-time evaluation.

		vectorize : When 'True' the integrand function is wrapped in the JAX vectorization
			transformation allowing the time parameter argument 't' or approximation argument
			'alpha' to be batched vectors.
			When 'False' the integrand function only allows either scalar values of time or single
			vector values of 'alpha'.
		
		Arguments
		---------
		r : int
			-Kacewicz order parameter determining the number of coefficient functions to
			instantiate, i.e. 'r + 1'.
			-This also represents the highest order derivative of the ASODEs driver function.
			-Expected to be a positive integer.
		d : int
			-Dimensionality of the ASODEs.
			-Expected to be positive non-zero integer.
		driver_function : Union[Callable[[numpy.ndarray], numpy.ndarray],
								Callable[[jax.Array], jax.Array]]
			-ASODEs driver function.
			-When using JAX Autodifferentiation, this is expected to be JAX compatible.
			-When using JAX Just in Time (JIT) compiler, this is expected to be traceable for JIT
		matrix : Union[None,
					   Union[jax.Array, numpy.ndarray],
					   Union[Callable[[float], numpy.ndarray], Callable[[float], jax.Array]]],
				 (Optional)
			-ASODEs Leading matrix.
			-If the leading matrix of the autnomous system is the identity matrix, this should be
			'None'.
			-If the leading matrix is a non-trivial, constant matrix, this should be an array
			representation of the matrix.
			-If this is a callable function, it is expected to be function of a scalar parameter
			that returns a square matrix.
				-If using JAX Autodifferentiation, this is expected to be a JAX compatible function.
			-Default value is 'None'.
		JIT : bool, (Optional)
			-Toggles whether or not all instantiated kacewicz functions are wrapped with JAX JIT
			transformations. Default value is True.
		debug : bool, (Optional)
			-Toggles boolean
		"""

		##----------------------------------------------------------------------------------------##

		debug = self.debug or debug
		if (debug):
			print("Setting Kaceiwcz Model Functions:",
				  f"{FORMAT.bold_equality('r', r)}",
				  f"{FORMAT.bold_equality('d', d)}",
				  f"{FORMAT.bold_equality('driver_function', driver_function)}",
				  f"{FORMAT.bold_equality('matrix', matrix)}",
				  f"{FORMAT.bold_equality('JAX', JAX)}",
				  f"{FORMAT.bold_equality('JIT', JIT)}",
				  f"{FORMAT.bold_equality('vectorize', vectorize)}",
				  f"{FORMAT.bold_equality('parallelize', parallelize)}",
				  sep = "\n\t")
		
		##----------------------------------------------------------------------------------------##

		if (d < 1):
			raise ValueError(f"kacewicz - Number of equations in the autonomou system {FORMAT.bold('d')} must be positive non-zero")
		self.d = d

		if (r < 0):
			raise ValueError(f"kacewicz - Order parameter {FORMAT.bold('r')} must be positive")
		self.r = r

		self.JAX = JAX
		self.vectorize = vectorize
		self.parallelize = parallelize
		self.JIT = JAX and JIT

		##----------------------------------------------------------------------------------------##

		if (parallelize): mode = "multiprocessing"
		else: mode = "serial"

		coefficient_functions = util.instantiate_coefficient_functions(r, driver_function,
																	   matrix = matrix,
																	   mode = mode,
																	   JIT = self.JIT,
																	   debug = debug)
		self.coefficient_functions = coefficient_functions

		integrand_function = util.instantiate_integrand_function(driver_function,
																 matrix = matrix,
																 vectorize = self.vectorize,
																 JIT = self.JIT,
																 debug = debug)
		self.integrand_function = integrand_function

		if (callable(matrix) and self.JIT): matrix = util.JIT(matrix)
		self.matrix = matrix
		
		##----------------------------------------------------------------------------------------##

		if (debug):
			print(f"Setting Kacewicz Model Functions: {FORMAT.green('COMPLETE')}")
	
	#==============================================================================================#

	def set_intervals(self, n = None, k = None, epsilon_1 = None, min_n_intervals = None, debug = False):
		"""
		Sets the number of primry and secondary intervals of the Kacewicz model given two potential
		sets of keyword arguments.

		SET 1:
			n - Number of primary intervals
			k - Kaceiwcz secondary interval parameter
		
		SET 2:
			epsilon_1 - Kacewicz epsilon parameter
			min_n_intervals - Minimum number of secondary intervals


		Arguments
		---------
		n : Union[None, int], (Optional)
			-Number of primary intervals.
			-When provided, this is expected to be a positive non-zero integer.
			-Default value is 'None', indicating it is not in use.
		k : Union[None, int], (Optional)
			-Kacewicz secondary interval parameter used to set the number of secondary intervals.
			-When provided, this is expected to be a positive non-zero integer.s
			-The default value is 'None', indicating it is not in use.
		epsilon_1 : Union[None, float], (Optional)
			-Kaceiwcz epsilon paramter used in the Epsilon Condition.
			-When provided, expected to be a number strictly between zero and one.
			Default value is 'None', indicating it is not used.
		min_n_intervals : Union[None, int], (Optional)
			-Minimum number of secondary intervals obtained from a stability condition on the ASODEs.
			-When provided, expected to be a positive non-zero integer.
			-Default value is 'None', indicating it is not used.
		"""
		
		debug = self.debug or debug
		if (debug):
			print("Setting Kacewicz Model Intervals:",
				  f"{FORMAT.bold_equality('n', n)}",
				  f"{FORMAT.bold_equality('k', k)}",
				  f"{FORMAT.bold_equality('epsilon_1', epsilon_1)}",
				  f"{FORMAT.bold_equality('min_n_intervals', min_n_intervals)}",
				  sep = "\n\t")

		##----------------------------------------------------------------------------------------##

		# Define the number of primary and secondary intervals explicitly
		if (n is not None and k is not None):
			if (not isinstance(n, int) or n < 1):
				raise ValueError(f"Kacewicz - number of primary intervals, {FORMAT.bold('n')}, must be a positive non-zero")
			if (not isinstance(k, int) or k < 1):
				raise ValueError(f"Kacewicz - secondary interval parameter, {FORMAT.bold('k')}, must be a positive non-zero integer")
			epsilon_1 = 1 / (n ** (k - 1))
			min_n_intervals = None
	
		# Calculate number of primary and secondary intervals from Epsilon Condition
		elif (epsilon_1 is not None and min_n_intervals is not None):
			if (not isinstance(epsilon_1, float) or epsilon_1 <= 0.0 or epsilon_1 >= 1.0):
				raise ValueError(f"Kacewicz - epsilon paramter, {FORMAT.bold('epsilon_1')}, must be strictly between zero and one")
			if (not isinstance(min_n_intervals, int) or min_n_intervals < 1):
				raise ValueError(f"Kacewicz - minimum number of secondary intervals, {FORMAT.bold('min_n_intervals')}, must be a positive non-zero integer")
			n, k = kacewicz.calculate_partition_parameters(epsilon_1, min_n_intervals)
		
		# Invalid Arguments
		else:
			raise ValueError("Kacewicz - unable to set number of primary and secondary intervals due to invalid arguments")
		
		##----------------------------------------------------------------------------------------##

		self.n = n
		self.k = k
		self.N_k = n ** (k - 1)
		self.n_intervals = n ** k
		self.epsilon_1 = epsilon_1
		self.min_n_intervals = min_n_intervals

		if (self.debug or debug):
			print(f"Setting Kacewicz Model Intervals: {FORMAT.green('COMPLETE')}")

	def calculate_partition_parameters(epsilon_1, min_n_intervals, n0 = 2):
		"""
		Calculates the values of the kacewicz primary and secondary interval parameters such that
		the kacewicz epsilon condition is satisfied, and the total number of secondary intervals is
		at least the minimum number of intervals, 'min_n_intervals'.

		Arguments
		---------
		epsilon_1 : float
			Kacewicz epsilon_1 parameter. Expected to be a floating point value strictly between
			zero and one.
		min_n_intervals : int
			Minimum total number of secondary intervals. This should be derived from any stability
			conditions imposed on the autonomous system of ODEs. Expected to be positive non-zero.
		n0 : int, (Optional)
			Initial guess for the number of primary intervals. The default value is two.

		Returns
		-------
		p : tuple[int]
			A tuple contaning the calculated Kacewicz interval parameters 'n' and 'k'
		"""

		n = n0
		k = kacewicz.calculate_k(epsilon_1, n)

		while (n ** k < min_n_intervals):
			n += 1
			k = kacewicz.calculate_k(epsilon_1, n)

		return (n, k)

	def calculate_k(epsilon_1, n):
		"""
		Given a values for kacewicz epsilon parameter 'epsilon_1' and the number of primary
		intervals 'n', this function calculates the value of kacewicz secondary interval parameter
		'k' such that the Kacewicz epsilon condition is satisfied.

		Arguments
		---------
		epsilon_1 : float
			Kacewicz epsilon parameter. Expected to be a floating point value strictly between zero
			and one.
		n : int
			Number of primary intervals. Expected to be positive non-zero.
		
		Returns
		-------
		k : int
			Kacewicz secondary interval parameter
		"""
		
		return int(np.ceil(1 - np.log(epsilon_1) / np.log(n)))

	#==============================================================================================#

	def run(self, tlims, ic, integral_mode, n_samples, quantum = True , hilbert_dimension = None,
	delta = None, dynamic_save = False, parameters_file_name = "", coefficients_file_name = "",
	directory = "", display_progress = True, profile_memory = True, debug = False):
		"""
		Runs the Kacewicz time marching algorithm to calculate all Taylor series coefficients for
		all secondary intervals. Assumes that all the necessary functions and interval parameters
		have been properly initialized.

		Notes
		-----
		Quantum:
			-The Kacewciz integral step is SIMULATED using a Quantum Amplitude Estimation Algorithm
			(QAEA) that accurately simulates the probabilistic behavior of a truly Quantum Integral
			Algorithm implemented on a quantum computer.
			-See randQAEA() and QAmpEst() in Kacewicz's core utilities.
		Dynamic Save:
			-When in dynamic save mode, after each Kacewicz integration step, the coefficients from
			the previous primary interval are saved to a coefficients file and are then "forgotten".
			-This mode uses less memory at the cost of the time it takes to write to files.
			-In order for the coefficients file to be used later, the Kacewciz model parameters are
			also saved to a parameters file.
			-Kacewicz model parameters and coefficients can be saved to specified coefficients file
			and parameter file names.
			-If these file names are not provided, time stamped files will names are used.

		Arguments
		---------
		tlims : tuple[float]
			-Boundaries of the time domain of the ASODEs.
			-Kacewicz model is able to solve both the normal problem, where 'tlims[0] < tlims[1]'
			and the terminal problem where 'tlims[0] > tlims[1]'.
			-Behavior is undefined if the boundry points are the same.
		ic : Union[jax.Array, numpy.ndarray]
			-ASODEs initial condition.
			-This is expected to be the state vector evaluated at the initial time, 'z(tlims[0])'.
			-The initial condition is expected to have 'd' components.
			-Because 
		integral_mode : str
			-String identifying the classical integral method used to approximate the integral in
			the Kacewicz integral step.
			-This must be one of the supported integral modes.
		n_samples : int
			-The number of samples used in the classical integral used to approximate the integral
			in the Kacewciz integral step.
			-Expected to be positive non-zero integer.
		
		quantum : bool, (Optional)
			-Toggles the use of the quantum simulation for calculating the integral estimation of
			the Kacewicz integral step.
			-See notes section.
			-When toggled on, the user is expected to provide values for the 'hilbert_dimension' and
			Kacewicz probability parameter, 'delta'. 
			-Default value is 'True'.
		hilbert_dimension : Union[int, None], (Optional)
			-Number of dimensions in the Hilbert Space used in the QAEA.
			-If in quantum mode, and is 'None', the number of secondary intervals in one primary
			interval, 'N_k', is used as the hilbert dimension.
			-If a non-None hilbert dimension is provided, it is expected to be a positive non-zero
			integer.
			-Defualt value is 'None'.
		delta : float, (Optional)
			-Kacewicz probability parameter used in the QAEA.
			-If running in quantum mode, it is expected to be a number strictly between zero and
			one.
			-The default value is '0.0'.
		
		dynamic_save : bool, (Optional)
			-Toggles whether or not Kacewicz model is run in Dynamic Save mode.
			-See notes section.
			-Default value is 'False'.
		parameters_file_name : str, (Optional)
			-If running in Dynamic Save mode, the Kacewicz model parameters are saved to a file.
			-The name of this file can be specified by this parameter.
			-If no 'parameters_file_name' is provided, a time stamped file name is generated.
			-The default value is the empty string.
		coefficients_file_name : {str, None}, (Optional)
			If using dynamic save, the Kaceicz model coefficients are saved to this file. If no coefficients file name is
			provided, an arbitrary file name is generated. Defualt value is None.
		directory : str, (Optional)
			If using dynamic save, all arbitrary files are stored to this directory. If this is the empty string, all
			arbitrary files are stored in the current working directory. Default value is "".
		debug : bool, (Optional)
			Toggles debugging print statements. Default value is False
		display_time_march : bool, (Optional)
			Toggles time march progress print statements. Default value is True.
		"""	

		##----------------------------------------------------------------------------------------##

		if (debug):
			if (directory == ""): directory_str = "\"\""
			else: directory_str = directory
			if (parameters_file_name == ""): parameters_file_name_str = "\"\""
			else: parameters_file_name_str = parameters_file_name
			if (coefficients_file_name == ""): coefficients_file_name_str = "\"\""
			else: coefficients_file_name_str = coefficients_file_name

			print("Running Time Marching Algorithm:",
				  f"{FORMAT.bold_equality('tlims', tlims)}",
				  f"{FORMAT.bold_equality('ic', ic)}",
				  f"{FORMAT.bold_equality('integral_mode', integral_mode)}",
				  f"{FORMAT.bold_equality('n_samples', n_samples)}",
				  f"{FORMAT.bold_equality('quantum', quantum)}",
				  f"{FORMAT.bold_equality('hilbert_dimension', hilbert_dimension)}",
				  f"{FORMAT.bold_equality('delta', delta)}",
				  f"{FORMAT.bold_equality('dynamic_save', dynamic_save)}",
				  f"{FORMAT.bold_equality('parameters_file_name', parameters_file_name_str)}",
				  f"{FORMAT.bold_equality('coefficients_file_name', coefficients_file_name_str)}",
				  f"{FORMAT.bold_equality('directory', directory_str)}",
				  f"{FORMAT.bold_equality('display_progress', display_progress)}",
				  sep = "\n\t")
		
		##----------------------------------------------------------------------------------------##

		if (tlims[0] == tlims[1]):
			raise ValueError("kacewicz - Time domain bounds must not be the same value")
		self.tlims = tlims

		self.interval_size = np.abs(tlims[1] - tlims[0]) / self.n_intervals
		
		if (np.shape(ic) != (self.d,)):
			raise ValueError("kacewicz - Initial condition must have the same components as the number of equations in the system")

		if (not integral_mode in kacewicz.supported_integrals):
			raise ValueError(f"kacewicz - Integral mode {FORMAT.bold_equality('integral_mode', integral_mode)} not supported")
		self.integral_mode = integral_mode
		
		if (n_samples < 1):
			raise ValueError("kacewicz - Number of integral samples must be positive non-zero")
		self.n_samples = n_samples

		self.quantum = quantum
		if (quantum):
			if (hilbert_dimension is None): hilbert_dimension = self.N_k
			elif (hilbert_dimension < 1):
				raise TypeError(f"Kacewicz - hilbert dimension {FORMAT.bold('hilbert_dimension')} must be positive non-zero")
		if (delta <= 0.0 or delta >= 1.0):
				raise TypeError(f"Kacewicz - probability parameter {FORMAT.bold('delta')} must be striclty between zero and one")
		self.hilbert_dimension = hilbert_dimension
		self.delta = delta

		##----------------------------------------------------------------------------------------##

		# Time marching algorithm can still dynamically save even if the file names are not given
		# Set up dynamic save files
		if (dynamic_save):
			time_name = time.ctime().replace(":", "-").replace(" ", "_")
			
			if (parameters_file_name == ""):
				parameters_file_name = directory + "parameters_" + time_name + ".json"
			
			if (coefficients_file_name == ""):
				coefficients_file_name = directory + "coefficients_" + time_name + ".txt"
				bytes_file = False
			else:
				if (coefficients_file_name[-3:] in ["npy", "npz"]): bytes_file = True
				elif (coefficients_file_name[-3:] == "txt"): bytes_file = False
				else: raise ValueError(f"kacewicz - Cannot time march kacewicz model in dynamic saving mode due to unrecognized file extension in coefficients file name, {FORMAT.bold_equality('coefficinets_file_name', coefficients_file_name)}")
					
			self.save_parameters(parameters_file_name, coefficients_file_name, debug = debug)
			
			if (bytes_file): coefficients_file = open(coefficients_file_name, "wb")
			else: coefficients_file = open(coefficients_file_name, "w+")

		##----------------------------------------------------------------------------------------##

		# Print Initial statements of time marching algorithm
		if (display_progress):
			print(self)

			if (dynamic_save):
				print(f"Coefficient array will use {self.N_k * (self.r + 2) * self.d * 64 / 8 / 1000} kB")
				print("To conserve memory, kacewicz will save and then forget coefficients of primary intervals that are no longer needed for time marching")
				print(f"Kacewicz coefficents will be saved to coefficients file - \"{coefficients_file_name}\"")		
			else:
				print(f"Coefficient array will use {self.n_intervals * (self.r + 2) * self.d * 64 / 8 / 1000} kB")

			print(f"\rProgress: (000 %) [{'.' * 100}] 0/{self.n_intervals}", end = '\r', flush = True)

		# Junp is the directed interval size used for iteration
		if (tlims[1] > tlims[0]): jump = self.interval_size
		else: jump = -self.interval_size

		# The "seed" value for the first primary interval is always the initial condition
		old_seed = ic

		##----------------------------------------------------------------------------------------##

		if (profile_memory):
			MemoryProfiler.clear()
			MemoryProfiler.record_data_point()

		##----------------------------------------------------------------------------------------##

		# Initialize coefficients array by calculating the coefficients of the first secondary interval
		# This also compiles most JIT functions for the first time
		# In JIT mode, display the memory consumption
		if (self.JIT and display_progress):
			print(f"\nMemory Before JIT Compiling: {psutil.Process(os.getpid()).memory_info().rss / 1000} kB", end = "")
			time_JIT_0 = time.time()

		self.coefficients = [util.compose_coefficients(self.coefficient_functions, old_seed, self.JIT)]
		
		if (self.JIT and display_progress):
			time_JIT_1 = time.time()
			time_JIT = time_JIT_1 - time_JIT_0
			print(f"\nCoefficient Function Compiling Time: {time_JIT} s")
			print(f"Memory After JIT Compiling: {psutil.Process(os.getpid()).memory_info().rss / 1000} kB")

		if (profile_memory): MemoryProfiler.record_stamp("First Coefficients")

		##----------------------------------------------------------------------------------------##

		if (display_progress):
			percent = int(1 / self.n_intervals * 100)
			print(f"\rProgress: ({percent:03} %) [{FORMAT.cyan('#' * percent)}{'.' * (100 - percent)}] 1/{self.n_intervals}", end = '\r', flush = True)

		# Time marching over all remaining secondary intervals
		for m in range(1, self.n_intervals):
			# Center of the Taylor series of the CURRENT secondary interval
			current_center = self.tlims[0] + m * jump
			
			###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###

			# Kacewicz Integral Step
			# Integration is taken over the domain of a primary interval
			if (m % self.N_k == 0):
				if (display_progress): start_integral_time = time.time()

				# Index of which integral step we are on
				integral_index = (m - 1) // self.N_k

				integral = self.integrate_primary_interval(integral_index, jump,
														   dynamic_save = dynamic_save,
														   display_progress = display_progress,
														   debug = debug)

				# Dynamically save coefficients of the previous primary interval to file
				# We don't need these values in memory anymore, so we can remove them
				if (dynamic_save):
					if (bytes_file): np.savez(coefficients_file, np.array(self.coefficients))

					else: np.savetxt(coefficients_file,
									 np.reshape(np.array(self.coefficients),
									 			(self.N_k, (self.r + 2) * self.d)))
					self.coefficients.clear()

				# Calculate the value of the next seed from the old seed value
				# Calculate using a non-trivial matrix
				if (callable(self.matrix)):
					old_center = self.tlims[0] + integral_index * self.N_k * jump
					b = np.dot(self.matrix(old_center), old_seed) + integral
					new_seed = np.linalg.solve(self.matrix(current_center), b)
				elif (isinstance(self.matrix, np.ndarray)):
					b = np.dot(self.matrix, old_seed) + integral
					new_seed = np.linalg.solve(self.matrix, b)
				# Calculate with a trivial identity matrix
				else: new_seed = old_seed + integral

				# Set the old seed as the current seed, setup for the next integration step
				old_seed = new_seed

				if (display_progress):
					integral_time = time.time() - start_integral_time
					integration_step = m // self.N_k

					if (integration_step == 1 and self.JIT):
						print(f"\nIntegration Step 1 + JIT Compiling: {integral_time}")
					else:
						print(f"\nIntegration Step {integration_step}: {integral_time} s")
					
					print(f"Kacewicz Model Memory: {psutil.Process(os.getpid()).memory_info().rss / 1000} kB")
				
				if (profile_memory): MemoryProfiler.record_stamp(f"Integral Step {m // self.N_k}")

				current_seed = old_seed
			
			###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###

			# The boundaries between adjascent secondary intervals within a primary interval must equal
			else:
				old_center = current_center - jump
				if (self.JIT): current_seed = util.taylor(current_center, old_center, self.coefficients[-1])
				else:
					current_seed = util.taylor(current_center, old_center, self.coefficients[-1])

			###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###

			# From the secondary interval seed value, generate the coefficients of the secondary intervals:
			self.coefficients.append(util.compose_coefficients(self.coefficient_functions,
															   current_seed,
															   JIT = self.JIT))

			###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###

			if (display_progress):
				percent = int((m + 1) / self.n_intervals * 100)
				print(f"\rProgress: ({percent:03} %) [{FORMAT.cyan('#' * percent)}{'.' * (100 - percent)}] {m + 1}/{self.n_intervals}", end = '\r', flush = True)

			if (profile_memory): MemoryProfiler.record_data_point()

		##----------------------------------------------------------------------------------------##

		# Save coefficients of last secondary interval
		if (dynamic_save):
			if (bytes_file): np.save(coefficients_file, np.array(self.coefficients))
			else: np.savetxt(coefficients_file, np.reshape(np.array(self.coefficients),
														   (self.N_k, (self.r + 2) * self.d)))
			self.coefficients.clear()
			coefficients_file.close()
		else: self.coefficients = np.array(self.coefficients)

		##----------------------------------------------------------------------------------------##

		if (display_progress):
			print("")

		if (profile_memory):
			MemoryProfiler.record_data_point()
			MemoryProfiler.plot()
		
		if (debug):
			print(f"Kacewicz time marching algorithm {FORMAT.GREEN}COMPLETE{FORMAT.END}")
	
	def integrate_primary_interval(self, integral_index, jump, dynamic_save = False, 
	display_progress = False, debug = False):
		"""
		Calculates the definite integral of the integrand function over the time domain of a primary
		interval.
		
		Arguments
		---------
		integral_index : int
		dynamic_save : bool, (Optional)
		debug : bool, (Optional)
		display_progress : bool, (Optional)

		Returns
		-------
		I : Array
			The definite integral
		"""

		if (debug):
			print(f"\nKaciewcz integral step {integral_index + 1}",
				  f"{FORMAT.bold_equality('interval_index', integral_index)}",
				  f"{FORMAT.bold_equality('dynamic_save', dynamic_save)}",
				  f"{FORMAT.bold_equality('display_progress', display_progress)}",
				  sep = "\n\t")

		##----------------------------------------------------------------------------------------##

		integral = np.zeros(self.d)
		
		# Index of the starting secondary interval
		start_index = self.N_k * integral_index

		if (display_progress):
			if (not debug): print()
			print(f"Integral Progress: (000 %) [{'.' * 100}] 0/{self.N_k}", end = '\r', flush = True)
		
		for j in range(self.N_k):
			# Define the integration bounds
			lower_limit = self.tlims[0] + (start_index + j) * jump
			upper_limit = self.tlims[0] + (start_index + j + 1) * jump
			limits = (lower_limit, upper_limit)

			# Taylor series coefficients of the secondary interval
			if (dynamic_save): C = self.coefficients[j]
			else: C = self.coefficients[start_index + j]

			integrand_function_ = lambda t: self.integrand_function(util.taylor(t, lower_limit, C))

			# Integral over a secondary interval
			sub_integral = integrate(integrand_function_, limits, self.integral_mode,
									 self.n_samples, self.quantum,
									 hilbert_dimension = self.hilbert_dimension, delta = self.delta)

			integral += sub_integral

			if (display_progress):
				percent = int((j + 1) / self.N_k * 100)
				print(f"\rIntegral Progress: ({percent:03} %) [{FORMAT.blue('O' * percent)}{'.' * (100 - percent)}] {j + 1}/{self.N_k}",
					  end = '\r', flush = True)

		return integral

	#==============================================================================================#

	def save(self, parameters_file_name = "", coefficients_file_name = "", directory = "",
	debug = False):
		"""
		Saves all kacewicz parameters and coefficients to a parameter file and coefficients file respectivley. If no
		parameters file name is given, a default parameters file name containing a time stamp will be used. In the event
		the coefficient file name is None, the same time stamp will be used to set and write the coefficient file.

		Arguments
		---------
		parameters_file_name : {str, None}, (Optional)
			Path of parameters file
		directory : str, (Optional)
			Relative path to a directory. If no parameters file name is given, the default parameters file and possible
			default coefficients file will be saved to this directory. Defualt value is the empty string, thus saving to 
			the current directory.
		debug : bool, (Optional)
			Toggles debugging print statements
		
		Returns
		-------
		(pf, cf) : tuple[str]
			A tuple containing the paths to the parameters file and coefficients file used
		"""

		if (debug):
			print("Saving Kacewicz Model:",
				  f"{FORMAT.bold_equality('parameters_file_name', parameters_file_name)}",
				  f"{FORMAT.bold_equality('coefficients_file_name', coefficients_file_name)}",
				  f"{FORMAT.bold_equality('directory', directory)}",
				  sep = "\n\t")

		time_name = time.ctime().replace(":", "-").replace(" ", "_")

		if (coefficients_file_name == ""):
			coefficients_file_name = directory + "coefficients_" + time_name + ".txt"
		self.save_coefficients(coefficients_file_name, debug = debug)
		
		if (parameters_file_name == ""):
			parameters_file_name = directory + "paramters_" + time_name + ".json"
		self.save_parameters(parameters_file_name, coefficients_file_name, debug = debug)

		if (debug):
			print(f"Saveing Kacewicz Modeol: {FORMAT.green('COMPLETE')}")

		return (parameters_file_name, coefficients_file_name)

	def save_parameters(self, parameters_file_name, coefficients_file_name, debug = False):
		"""
		Save all kacewicz model parameters to .json file. It is assumed that the model has been properly initialized.

		Arguments
		---------
		parameters_file_name : str
			Path to parameters file that we want to save to.
		coefficients_file_name : str
			Path to the coefficients file where we will save the coefficients of the kacewicz mode. This is saved to the 
			parameters file so that we can load the coefficients from the parameters file.
		debug : bool, (Optional)
			Toggle debugging print statements
		"""
		
		if (debug):
			print("Saving Kacewicz Model Parameters",
				  f"{FORMAT.bold_equality('parameters_file_name', parameters_file_name)}",
				  f"{FORMAT.bold_equality('coefficients_file_name', coefficients_file_name)}",
				  sep = "\n\t")
		
		parameters_file = open(parameters_file_name, "w")
		
		parameters = self.get_parameters()
		
		# JSON does not recognize numpy data types
		for key in parameters:
			if (type(parameters[key]) is np.int64): parameters[key] = int(parameters[key])
			if (type(parameters[key]) is np.float64): parameters[key] = float(parameters[key])

		parameters["coefficients_file_name"] = coefficients_file_name

		dump(parameters, parameters_file, indent = 1)
		
		parameters_file.close()

		if (debug):
			print(f"Saving Kacewicz Model Parameters: {FORMAT.green('COMPLETE')}")

	def save_coefficients(self, coefficients_file_name, debug = False):
		"""
		Saves the Taylor series coefficients of the kacewicz model approximation to file

		Arguments
		---------
		coefficients_file_name : str
			Path to coefficients file to save coefficients to
		debug : bool, (Optional)
			Toggle debugging print statements
		"""
		
		if (debug):
			print("Saving Kacewicz Model Coefficients:",
				  f"{FORMAT.bold_equality('coefficients_file_name', coefficients_file_name)}",
				  sep = "\n\t")
		
		if (coefficients_file_name == ""):
			raise ValueError("Cannot save kacewicz model coefficients without coefficients file name")
		
		# Saving coefficinets as bytes
		if (coefficients_file_name[-3:] == "npy" or coefficients_file_name[-3:] == "npz"):
			coefficients_file = open(coefficients_file_name, "wb")
			np.save(coefficients_file, self.coefficients)
		# Saving coefficients as text
		elif (coefficients_file_name[-3:] == "txt"):
			coefficients_file = open(coefficients_file_name, "w")
			np.savetxt(coefficients_file, np.reshape(self.coefficients, (self.n_intervals, (self.r + 2) * self.d)))
		else:
			raise ValueError(f"Cannot save kacewicz model coefficients due to an unrecognized coefficients file extension; \"{coefficients_file_name}\"")
		
		coefficients_file.close()

		if (debug):
			print(f"Saving Kacewicz Model Coefficients: {FORMAT.green('COMPLETE')}")

	#==============================================================================================#

	def evaluate(self, t):
		"""
		Evaluates the approximation function on a single time point or an array of time points. The
		model must have all parameters and coefficinets loaded.

		Arguments
		---------
		t : Union[float, numpy.ndarray, jax.Array]
			-Input time point or Array of time points to evaluate the approximation at.

		Returns
		-------
		x : Union[numpy.ndarray, jax.Array]
			-Approximation evaluated at the input time point or set of time points.
			-Output is batched on the first axis, i.e. axis zero.
		"""

		##----------------------------------------------------------------------------------------##

		# Kacewicz CANNOT be evaluated at time points outside of its defined time domain
		if (np.min(t) < np.min(self.tlims) or np.max(t) > np.max(self.tlims)):
			raise ValueError("Cannot evaluate time point(s) outside the time domain of the model")
		
		# Kacewciz CANNOT be evaluated without fully loaded coefficients
		if (self.coefficients is None or np.shape(self.coefficients) != (self.n_intervals, self.r + 2, self.d)):
			raise RuntimeError("Cannot evaluate kacewicz model with non-existant or incomplete set of coefficients")
		
		##----------------------------------------------------------------------------------------##

		# Centers for each Taylor series, i.e. the starting point of each secondary interval
		centers = np.linspace(self.tlims[0], self.tlims[1], self.n_intervals, endpoint = False)

		indices = np.digitize(t, centers, right = self.tlims[1] < self.tlims[0]) - 1
		unique_indices = np.unique(indices)

		output = np.concatenate([util.taylor(t[indices == m], centers[m], self.coefficients[m], JIT = self.JIT)
								 for m in unique_indices], axis = 0)

		return output

	#==============================================================================================#

	def plot(self, t = None, H = 1000, show = True, solution_function = None):
		"""
		Plot the approximate solution of the autonomous system of ODEs. Each component of the
		approximation is plot on its own figure. No more than 10 component figures can be plotted.
		
		Arguments
		---------
		t : {Array[float], None}, (Optional)
		H : int, (Optional)
		show : bool, (Optional)
		solution_function : Callable, (Optional)

		"""
		
		if (t is None):
			t = np.linspace(self.tlims[0], self.tlims[1], H, endpoint = True)
		
		approximation = self.evaluate(t)
		
		if (solution_function is not None):
			solution = solution_function(t)
		
		fig = plt.figure()
		sub = fig.add_subplot(111)

		def plot_component(d):
			sub.set(xlabel = "t", ylabel = f"z_{d}")
			
			sub.plot(t, approximation[:, d])
			
			if (solution_function is not None):
				sub.plot(t, solution[d], "--", color = "k")

		plot_component(0)

		##----------------------------------------------------------------------------------------##

		class Index():
			def __init__(self, limit):
				self.limit = limit
				self.ind = 0

			def next(self, event):
				self.ind = (self.ind + 1) % self.limit
				sub.cla()
				plot_component(self.ind)
				plt.draw()

			def prev(self, event):
				self.ind = (self.ind - 1) % self.limit
				sub.cla()
				plot_component(self.ind)
				plt.draw()

		##----------------------------------------------------------------------------------------##

		callback = Index(self.d)
		axprev = fig.add_axes([0.7, 0.05, 0.1, 0.075])
		axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
		bnext = Button(axnext, 'Next')
		bnext.on_clicked(callback.next)
		bprev = Button(axprev, 'Previous')
		bprev.on_clicked(callback.prev)

		if (show): plt.show()

		return (fig, approximation)

####################################################################################################

def odeint(d, driver_function, ic, t, integral_mode, n_samples, **kwargs):
	"""
	User facing wrapper function for easily instantiating, running, and saving a Kacewicze model
	parameters and coefficients to file.

	See Kaceiwcz Class Docstring.
	
	Arguments
	---------
	d : int
		-Dimensionality of the ASODEs, i.e. the number of equations in the ASODEs.
		-Expected to be a positive, non-zero integer.
	f : Union[Callable[[jax.Array], jax.Array], Callable[np.ndarray], np.ndarray]]
		-ASODEs driver function.
		-'f' should only be a function of the State Vector 'z'.
		-Because the first component of the State Vector, 'z[0]', is the time parameter, 't', the
		first component of the driver function should be 1.0 for any input state 'z'.
		-When using JAX Autodifferentiation, 'f' must be JAX traceable.
	ic : Union[jax.Array, np.ndarray]
		-ASODEs initial condition.
		-Expected to have 'd' components
	integral_mode : str
		-The classical integral method used in the Kacewicz integral step.
		-This can be either "riemann" or "quad" specifying a Left-Reimann Sum or Gauss-Legendre
		respectively.
	t : np.ndarray
		-An array of monotonic time values.
		-The upper and lower bounds of the array are used as the boundaries of the time domain.
		-The total number of time points is interpreted as the minimum number of secondary intervals
		constrained by any stability conditions on the ASODEs.
	n_samples : int
		-The number of samples used in the Kacewicz integral step.
		-Combined with the integral mode, this determines the value of 'r' the Kacewicz order
		parameter.
		-This is done to preserver the convergence order of the integral method throughout the rest
		of the algorithm.
		-Expected to be a positive non-zero integer.

	Keyword Arguments
	-----------------
	quantum : bool, (Optional)
		-Toggles Quantum mode, i.e. the use of the Quantum Amplitude Estimation Algorithm (QAEA) to
		simulate calculating the Kacewicz integral step on a quantum computer.
		-Defualt vlalue is 'True'.
	delta : Union[float, None], (Optional)
		-Kacewicz probability parameter used in the QAEA.
		-It is only used when model is operating in Quantum mode.
		-When given, expected to be a number between zero and one.
		-Default value is 'None'.
	JAX : bool, (Optional)
		-Control flag toggling the use of JAX Automatic differentiation.
		-Default value is 'True'.

	n : int, (Optional)
	k : int, (Optional)
	epsilon_1 : float, (Optional)

	JIT : bool, (Optional)
	parallelize : bool, (Optional)
	matrix : Union[Union[Callable[], Callable[]], Union[], None]

	save : bool, (Optional)	
	dynamic_save : bool, (Optional)
	display_progress : bool, (Optional)
	parameters_file_name : str, (Optional)
	coefficients_file_name : str, (Optional)
	directory : str, (Optional)

	return_parameters : bool, (Optional)
	return_mode : bool, (Optional)
	debug : bool, (Optional)
	"""

	#==============================================================================================#

	# Control flags and keyword arguments are numerous and can be expansive
	df_kwargs = {"matrix":None, "JAX":True, "JIT":True, "parallelize":True, "vectorize":True,
				 "n":None, "k":None, "epsilon_1":None, "min_n_intervals":None,
				 "hilbert_dimension":None, "delta":None, "quantum":True, "dynamic_save":None,
				 "display_progress":True, "profile_memory":False, "parameters_file_name":"",
				 "coefficients_file_name":"", "directory":"", "save":False,
				 "return_parameters":False, "return_model":False, "plot_output":False,
				 "show_output":False, "debug":False}

	key_intersection = kwargs.keys() - df_kwargs.keys()
	if (len(key_intersection) > 1): raise ValueError(f"Kacewicz - Unrecognized keyword argument  {key_intersection.pop()}")
	
	df_kwargs.update(kwargs)

	# The time domain bounds and the minimum number of secondary intervals is taken from the input
	# time domain
	tlims = (t[0], t[-1])
	min_n_intervals = len(t)
	f = driver_function
	debug = df_kwargs["debug"]

	# The kacewicz order parameter, 'r', is calculated from the integral mode and potentially the
	# number of integral samples. This is done to preserver the convergence order of the integral
	# mode
	if (integral_mode == "riemann"):
		if (debug):
			print(f"kacewicz - To recover convergence order in Riemann integral mode, {FORMAT.bold('r')} is being set to 1")
		r = 2
#	elif (integral_mode == "trapz"):
#		if (debug): print(f"kacewicz - To recover convergence order in Trapezoid integral mode, {FORMAT.bold('r')} is being set to 3")
#		r = 3
	elif (integral_mode == "quad"):
		if (debug): print(f"kacewicz - To recover convergence order in Gauss-Quadrature integral mode, {FORMAT.bold('r')} is being set to 2 * {FORMAT.bold('n_samples')} - 1")
		r = 2 * n_samples - 1
	else:
		raise ValueError(f"kaceiwcz - Unrecognized integral mode {FORMAT.bold_equality('integral_mode', integral_mode)}")

	#==============================================================================================#

	# Instantiate Kacewicz model
	model = kacewicz.model(r, d, f, matrix = df_kwargs["matrix"], JAX = df_kwargs["JAX"], JIT = df_kwargs["JIT"],
						   vectorize = df_kwargs["vectorize"], parallelize = df_kwargs["parallelize"],
						   n = df_kwargs["n"], k = df_kwargs["k"], epsilon_1 = df_kwargs["epsilon_1"],
						   min_n_intervals = min_n_intervals, debug = debug)
	
	#==============================================================================================#

	# Run Kacewicz Model
	model.run(tlims, ic, integral_mode, n_samples, quantum = df_kwargs["quantum"],
			  hilbert_dimension = df_kwargs["hilbert_dimension"], delta = df_kwargs["delta"],
			  dynamic_save = df_kwargs["dynamic_save"], parameters_file_name = df_kwargs["parameters_file_name"],
			  coefficients_file_name = df_kwargs["coefficients_file_name"], 
			  directory = df_kwargs["directory"], display_progress = df_kwargs["display_progress"],
			  profile_memory = df_kwargs["profile_memory"], debug = debug)

	#==============================================================================================#

	# If Dynamic Save is turned OFF, but save is turned ON, then save parameters and coefficients
	if (not df_kwargs["dynamic_save"] and df_kwargs["save"]):
		(parameters_file_name,
		 coefficients_file_name) = model.save(parameters_file_name = df_kwargs["parameters_file_name"],
											  coefficients_file_name = df_kwargs["coefficients_file_name"],
											  directory = df_kwargs["directory"], debug = debug)

	#==============================================================================================#

	# If Dynamic Save is turned ON, load all coefficients to prepare for approximation evaluation
	if (df_kwargs["dynamic_save"]):
		model.coefficients = kacewicz.load_coefficients(coefficients_file_name, model.n_intervals,
														model.r, model.d, debug = debug)

	#==============================================================================================#

	approx = model.evaluate(t)

	#==============================================================================================#

	if (not df_kwargs["return_parameters"] and not df_kwargs["return_model"]): return approx

	output = (approx,)
	if (df_kwargs["return_parameters"]):
		params = model.get_parameters()
		output += (params,)
	if (df_kwargs["return_model"]): output += (model,)
	
	return output

####################################################################################################

if (__name__ == "__main__"):
	JAX = True
	if (JAX and not JAX_FLAG): use_JAX_util()
	elif (not JAX and JAX_FLAG): use_FD_util()

	problem = util.test_ODE(matrix = True, N = 2)

	#==============================================================================================#

	d = problem["dimension"]
	driver_function = problem["driver"]
	tlims = (0., 1.)
	ic = problem["solution_function"](tlims[0])
	integral_mode = "quad"

	n_samples = 2
	#r = 2 * n_samples - 1
	quantum = True

	matrix = problem["matrix"]
	vectorize = True
	parallelize = False
	JIT = True

	n = 2
	k = 4
	min_n_intervals = 1000
	epsilon_1 = 0.005

	hilbert_dimension = None
	delta = 0.001

	dynamic_save = True
	parameters_file_name = "parameters.json"
	coefficients_file_name = "coefficients.txt"

	display_progress = True
	debug = True
	profile_memory = True

	#==============================================================================================#

	t = np.linspace(tlims[0], tlims[1], min_n_intervals, endpoint = True)


	model = kacewicz.model(r, d, driver_function, matrix = matrix, vectorize = vectorize,
						   parallelize = parallelize, JAX = JAX, JIT = JIT, n = n, k = k,
						   epsilon_1 = epsilon_1, min_n_intervals = min_n_intervals, debug = debug)
	model.run(tlims, ic, integral_mode, n_samples, quantum = quantum,
			  hilbert_dimension = hilbert_dimension, delta = delta, dynamic_save = dynamic_save,
			  parameters_file_name = parameters_file_name,
			  coefficients_file_name = coefficients_file_name,
			  display_progress = display_progress, profile_memory = profile_memory, debug = debug)

	if (dynamic_save):
		model.load(parameters_file_name, debug = True)

	model.plot(solution_function = problem["solution_function"])

	
	f = driver_function
	(approx, model) = odeint(d, f, ic, t, integral_mode, n_samples,
							 epsilon_1 = epsilon_1, delta = delta, matrix = matrix,
							 dynamic_save = dynamic_save, save = True, display_progress = True,
							 parameters_file_name = parameters_file_name,
							 coefficients_file_name = coefficients_file_name, return_model = True,
							 debug = True)
	
	model.plot(solution_function = problem["solution_function"])

	plt.show()
	print(f"{FORMAT.green('COMPLETE')}")