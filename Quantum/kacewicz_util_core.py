import numpy as np

from scipy.special import roots_legendre

import psutil

import time

from matplotlib import pyplot as plt

import os

####################################################################################################

# Linux Terminal Color-Coding Formatting Class
class FORMAT():
	END = "\033[0m" # Ends special formatting
	RED = "\033[91m" # Red Text
	GREEN = "\033[92m" # Green Text
	BLUE = "\033[94m" # Blue Text
	CYAN = "\033[96m" # Cyan Text
	BOLD = "\033[1m" # Bold Text

	#==============================================================================================#

	def bold(value):
		# Returns a string with bold value
		return f"{FORMAT.BOLD}{value}{FORMAT.END}"
	
	def bold_equality(string, value):
		# Returns a the string '<string> = <value>' with a bold string formatting
		return f"{FORMAT.bold(string)} = {value}"

	def green(value):
		# Returns a string with green value
		return f"{FORMAT.GREEN}{value}{FORMAT.END}"

	def blue(value):
		# Returns a strng with blue value
		return f"{FORMAT.BLUE}{value}{FORMAT.END}"
	
	def cyan(value):
		# Returns a string with cyan value
		return f"{FORMAT.CYAN}{value}{FORMAT.END}"

####################################################################################################

# Memory Profiling
class MemoryProfiler():
	data = [[], []]
	n_points = 0
	stamps = []

	def plot():
		fig = plt.figure()
		sub = fig.add_subplot(111)
		sub.set(title = "Kacewicz Memory Profile", xlabel = "Time %", ylabel = "Memory (kB)")
		
		max_t = max(MemoryProfiler.data[0])
		min_t = min(MemoryProfiler.data[0])
		a = 1 / (max_t - min_t)
		b = -min_t * a
		time_points = np.asarray(MemoryProfiler.data[0]) * a + b
		sub.plot(time_points, MemoryProfiler.data[1])

		plt.show()

	def log():
		file = open("memory_profile_log.txt", "w")
		for stamp in MemoryProfiler.stamps:
			file.write(f"{stamp[1]}, {MemoryProfiler.data[0][stamp[0]]}, {MemoryProfiler.data[1][stamp[0]]}")
		file.close()

	def append_data_point(time_point, memory):
		"""
		Appends a new time-memory data piont to the memory profiler. It is assumed that the time
		argument is larger than all the time values in the existing data list.
		"""
		
		MemoryProfiler.data[0].append(time_point)
		MemoryProfiler.data[1].append(memory)
		
		MemoryProfiler.n_points += 1

	def record_data_point():
		time_point = time.time()
		memory = psutil.Process(os.getpid()).memory_info().rss / 1000
		MemoryProfiler.append_data_point(time_point, memory)

	def append_stamp(time_point, memory, label):
		"""
		Appends a new time-memory data point to the memory profiler. It is assumed that the time
		argument is greater than all the time values in the exsiting data list.
		
		Arguments
		---------
		time : int
			-Linux time in seconds.
		memory : float
			-Memroy usage at the specified Linux time in kB.
		label : str
			-Label associated with the data stamp.
		"""

		MemoryProfiler.append_data_point(time_point, memory)
		MemoryProfiler.stamps.append((MemoryProfiler.n_points - 1, label))

	def record_stamp(label):
		time_point = time.time()
		memory = psutil.Process(os.getpid()).memory_info().rss / 1000
		MemoryProfiler.append_stamp(time_point, memory, label)

	def clear():
		MemoryProfiler.data[0].clear()
		MemoryProfiler.data[1].clear()

		MemoryProfiler.stamps.clear()
		
		MemoryProfiler.n_points = 0

####################################################################################################

def integrate_riemann(integrand_function, tlims, n_samples, quantum, delta = 0.0,
hilbert_dimension = 0):
	"""
	Simulates a quantum algorithm that calculates the definite Left Riemann integral of the given
	integrand function using a quantum amplitude estimation algorithm.

	Arguments
	---------
	integrand_function : Callable
		This is the integrand function of the integral. It must be a vectorized function that
		accepts an array of time values as an input. If the integrand function is vector valued, the
		output vectors must be batched along the zero axis.
	tlims : tuple[float, float]
		Lower and upper bounds of the integral. These can be any floating point values
		so long as the are not identical.
	n_samples : int
		Number of samples in the left riemann integral. This should be a positive, non-zero integer.
	quantum : bool
		This is a boolean toggle used to invoke the simulation of the quantum amplitude estimation
		algorithm. If this is 'False', the classical left riemann integral will be returned.
	delta : float, (Optional)
		This is the Kacewicz algorithm's probability parameter, used in the quantum amplitude
		estimation algorithm. It is a requried argument when operating in quantum mode, and should
		be strictly between zero and one. Default value is zero.
	hilbert_dimension : int, (Optional)
		THis is the number of Hilbert dimensions to use in the quantum amplitude estimation
		algorithm. It is a requried argument when operating in quantum mode, and should be a
		positive non-zero integer. Default value is '0'.
	
	Returns
	-------
	integral : Array
		Returns the left reimann integral approximation
	"""

	#==============================================================================================#

	# Scale the integral domain to be between 0 and 1
	a = tlims[1] - tlims[0]
	b = tlims[0]
	u = np.linspace(0, 1, n_samples, endpoint = False)

	t = a * u + b

	f = integrand_function(t)

	#==============================================================================================#

	# Return classical left riemann integral
	if (not quantum): return np.mean(a * f, axis = 0)

	#==============================================================================================#

	# Simulate quantum riemann integral
	if (delta <= 0.0 or delta >= 1.0):
		raise ValueError(f"Cannot simulate quantum riemann integral without Kacewicz probability parameter")	
	if (hilbert_dimension < 1):
		raise ValueError(f"Cannot simulate quantum reimann integral without Hilbert dimension parameter")

	# The integrand function co-domain must be scaled to be between 0 and 1 so the integral 
	# approximation can be interpreted as a robability
	f_max = np.max(f, axis = 0)
	f_min = np.min(f, axis = 0)
	f_diff = f_max - f_min
	f_scaled = (f - f_min) / f_diff

	# We use the average as a probability
	f_avg = np.mean(f_scaled, axis = 0)

	# The probability is converted into a quantum aplitude
	omega = np.arcsin(np.sqrt(f_avg)) / np.pi
	
	estimates = QAmpEst3(hilbert_dimension, omega, delta)

	integral = (tlims[1] - tlims[0]) * (estimates * f_diff + f_min)

	return integral

def integrate_quad(integrand_function, tlims, legendre_order, quantum, hilbert_dimension = 0,
delta = 0.0):
	"""
	Simulates a quantum algorithm that calculates the definite Gauss-Legendre quadrature integral of
	the given integrand function using a quantum amplitude estimation algorithm.

	Arguments
	---------
	integrand_function : Callable
		This is the integrand function of the integral. It must be a vectorized function that
		accepts an array of time values as an input. If the integrand function is vector valued, the
		output vectors must be batched along the zero axis.
	tlims : tuple[float, float]
		Lower and upper bounds of the integral. These can be any floating point values
		so long as the are not identical.
	legendre_order : int
		Highest polynomial order used in the Gauss-Legendre integral. This is also the number of
		samples in the Gauss-Legendre quadrature integral. This should be a positive, non-zero
		integer.
	quantum : bool
		This is a boolean toggle used to invoke the simulation of the quantum amplitude estimation
		algorithm. If this is 'False', the classical Gauss-Legendre quadrature integral will be
		returned.
	delta : float, (Optional)
		This is the Kacewicz algorithm's probability parameter, used in the quantum amplitude
		estimation algorithm. It is a requried argument when operating in quantum mode, and should
		be strictly between zero and one. Default value is zero.
	hilbert_dimension : int, (Optional)
		THis is the number of Hilbert dimensions to use in the quantum amplitude estimation
		algorithm. It is a requried argument when operating in quantum mode, and should be a
		positive non-zero integer. Default value is '0.0'.

	Returns
	-------
	integral : Array
		Returns the Gauss-Legendre Quadrature integral approximation
	"""

	#==============================================================================================#

	# Find the roots and weights for the Gauss-Legendre integral approximation
	roots, weights = roots_legendre(legendre_order)
	
	# Linear scale parameters from the domain [-1, 1] to the integral domain
	a = (tlims[1] - tlims[0]) / 2
	b = (tlims[1] + tlims[0]) / 2

	f = (integrand_function(a * roots + b).T * weights).T * a

	if (not quantum): return np.sum(f, axis = 0)

	if (delta <= 0.0 or delta >= 1.0):
		raise ValueError("kacewicz integrate_quad() - Cannot calculate quantum integral without probability parameter")
	
	if (hilbert_dimension < 1):
		raise ValueError("kacewicz integrate_quad) - Cannot calculate quantum integral without M parameter of the quantum amplitude estimation algorithm")

	#==============================================================================================#

	# Scale the integrand function co-domain to between 0 and 1 so the integral approximation can be
	# interpreted as a probability
	f_max = np.max(f, axis = 0)
	f_min = np.min(f, axis = 0)
	f_diff = f_max - f_min
	f_scaled = (f - f_min) / f_diff

	# We use the average as a probability
	f_avg = np.mean(f_scaled, axis = 0)

	#==============================================================================================#

	omega = np.arcsin(np.sqrt(f_avg)) / np.pi

	#==============================================================================================#
	
	estimates = QAmpEst3(hilbert_dimension, omega, delta)

	#==============================================================================================#

	integral = legendre_order * (estimates * f_diff + f_min)

	return integral

def integrate(integrand_function, tlims, integral_mode, n_samples, quantum, hilbert_dimension = 0,
delta = 0.0):
	"""
	Calculates the definite integral of a time valued function over a finite domain using a
	simulation of a quantum integral estimation algorithm.

	Arguments
	---------
	integrand_function : Callable
		Callable integrand function that accepts a 1D array of values as input.
	tlims: tuple[float]
		Lower and upper bounds of the the finite time domain.
	integral_mode : str
		Classical numerical ntegral mode being called
	n_samples : int
		Number of samples in the in time the domain
	quantum : bool
		Boolean flag used to toggle the use of the quantum amplitude estimation algorithm that
		simulates the quantum integral algorithm.
	hilbert_dimension : int, (Optional)
		Hilbert dimension used in the quantum amplitude estimation algorithm. This is a requried
		argument when running in 'quantum' mode. Default value is '0'.
	delta : float, (Optional)
		Kaceiwcz probability parameter used in the quantum amplitude estimation algorithm. This is
		a requried argument when running in 'quantum' mode. Default value is '0.0'
	
	Returns
	-------
	integral : Array
		Returns the finite integral of the integrand function using the specified quantum
		integration method.
	"""

	if (integral_mode == "riemann"):
		return integrate_riemann(integrand_function, tlims, n_samples, quantum,
								 hilbert_dimension = hilbert_dimension, delta = delta)
	
	elif (integral_mode == "quad"):
		return integrate_quad(integrand_function, tlims, n_samples, quantum,
							  hilbert_dimension = hilbert_dimension, delta = delta)

	else: raise ValueError("kaceiwcz integrate() - Invalid integral_mode")

####################################################################################################

def pdf(x, M, omega):
	"""
	This is the probability distribution function of our quantum simulator. Given, a point in the
	hilbert space, the size of the hilbert space, and the true phase of the amplitude we want to
	estimate, it gives a probability density.

	Arguments
	---------
	x : int
		A point in the hilbert space
	M : int
		This is the hilbert-dimension of the computational basis you want to simulate.
	omega : float
		This is the true phase of the quantum amplitude we want to estimate
	"""

	angle = M * omega - x

	return np.sin(np.pi * angle) ** 2 / np.sin(np.pi * angle / M) ** 2 / M ** 2

####################################################################################################

def randQAEA(M, omega):
	"""
	Generates random deviate for Quantum Amplitude Estimation. Samples random deviate from the
	probability distribution function over discrete values of 'x'. See Bassard et al. quant-ph
	0005055. Code originally written by F. Gaitan.
	
	THIS FUNCTION IS DEPRICATED and only exists as reference. When implementing, use randQAEA3()
	
	Arguments
	---------
	M :	 int
		Is the dimension of the Hilbert Space
	omega :	float
		Is the unknown scalar used for deriving the quantum amplitude. It is assumed that ``omega`` is a scalar quantity
		between 0 and 1 inclusive
	
	Returns
	-------
	x : int
		Returns a random deviate

	Examples
	--------
	TODO
	"""
	
	Momega = M * omega

	# SubIntCounter identifies which subinterval is currently being processed in the while loop below
	# Subintervals are numbered 1 -> M, and subinterval j corresponds to QAEA probability p[j - 1]
	SubIntCounter = 0

	# Stores QAEA discrete probability distribution. p(j - 1), with 1 <= j <= M
	Probs = []

	# Stores partial sums of p(j - 1) from 1 to M terms
	p_sums = []

	# Integer values of discrete domain
	y = np.array(range(M))
	x = y - Momega # Shifted domain appearing in p(y)

	# Calculate QAEA probabilities p(j - 1), store in Probs(j) with 1 <= j <= M
	# Note that probs[1] = p(0), probs(j) = p(j - 1), and Probs(M) = p(M - 1)
	tempProb = np.sin(np.pi * x) / np.sin(np.pi * x / M)
	Probs = (1 / M ** 2) * (tempProb ** 2)

	# Accumulate partial sums of p(j - 1), 1 <= j <= M, store in p_sums, where p_sums(j) = p(0) + ... + p(j - 1)
	p_sums.append(Probs[0])
	for i in range(1, M):
		p_sums.append(p_sums[i - 1] + Probs[i])

	# Sample uniform deviate
	u = np.random.random()

	# Determine which subinterval contains u
	# 1. Loop through subintervals, with SubIntCounter tracking which subinterval is currently being processed
	# 2. Exit loop when u > A(SubIntCounter) first occurs. Subinterval labeled by SubIntCounter at exit contains u
	while (p_sums[SubIntCounter] < u):
		SubIntCounter += 1

	return SubIntCounter 
def randQAEA2(M, omega, shape = ()):
	"""
	Calculates random quantum deviate(s) for a single value of phase 'omega' and hilber dimension
	'M'. The output deviates 

	THIS FUNCTION IS DEPRICATED and only exists as reference. When implementing, use randQAEA3()

	Arguments
	---------
	M : int
		Hilbert dimension. Expected to be positive non-zero.
	omega : float
		Quantum phase. Expected to be 
	shape : {int, tuple[int]}, optional
		int - Calculates ``shape`` random quantum deviates
		tuple[int] - Calculates an Array of random quantum deviates with shape ``shape``
		None - Calculates one random quantum deviate
		Default value is None.

	Returns
	-------
	x : {int, Array[int]}
		Single random quantum deviate or an array of random quantum deviates
	
	Examples
	--------
	"""

	# Calculate discrete commulative distribution function values
	cdfs = np.cumsum(pdf(np.arange(0, M, 1), M, omega))

	return np.searchsorted(cdfs, np.random.random(shape))
def randQAEA3(M, omega, length = 0):
	"""
	Calculates random quantum deviate(s)

	Random quantum deviates are calculated by sampling the discrete quantum probability distribution function
		
	Arguments
	---------
	M : int
		Dimension of the Hilbert space of the computational basis. For a quantum computer, this should be two to the power of
		the number of qubits used. In Kacewicz algorithm, the value of ``M`` is closely tied to the number of primary and
		sceondary intervals, thus determining the accuracy of the approximation.
	omega : {float, Array[float]}
		The phase of the quantum amplitude. In Kacewicz, this is calculated from an approximation of the an integral. To
		speed up computations where multiple values of the ``omega`` are involved, ``omega`` can be an array
	length : {int}, optional
		This is the number of deviates calculated per phase value. If ``omega`` is an Array, an array random quantum deviates
		is calculated where each value in ``omega`` is associated with ``length`` random quantum deviates.The default value is
		None.
	
	Returns
	-------
	x : {int or Array[int]}
		Random quantum deviate or an Array of random quantum deviates

	Examples
	--------
	TODO
	"""

	# Generate discrete probability values
	discrete = np.moveaxis(np.tile(np.arange(0, M, 1), np.shape(omega) + (1,)), np.ndim(omega), 0)
	probs = pdf(discrete, M, omega)

	# Calculate discrete cumaltive distribution values
	cdfs = np.cumsum(probs, axis = 0)

	# Calculate one random deviate for each phase value
	if (length == 0):
		samples = np.random.random(np.shape(omega))
		deviates = np.sum(cdfs < samples.reshape((1,) + np.shape(samples)), axis = 0)
	
	# Calculate multiple random deviate values for each phase value
	else:
		samples = np.random.random(np.shape(omega) + (length,))
		deviates = np.sum(cdfs.reshape(np.shape(cdfs) + (1,)) < samples.reshape((1,) + np.shape(samples)), axis = 0)

	return deviates

def QAmpEst(M, omega, delta):
	"""
	Estimates unknown quantum amplitude using Quantum Amplitude Estiamtion Algorithm. Code written by F. Gaitan. See 
	(Brassard et al., quant-ph/0005055)
	
	THIS FUNCTION IS DEPRICATED and only exists as reference. For implementation, use randQAEA3()

	Arguments
	---------
	M : int
		Is the dimension of the Hilbert space
	omega : float
		Is the unkown scalar used for deriving the quantum amplitude
	delta : float
		Is one minus the probability that our amplitude estimation is within the error bounds of the true amplitude

	Returns
	-------
	 x : tuple[float, float, float, str, int, float]
	 	Returns tuple with the following named components (a_estimate, a, error, message, SucccessFlag, upper_bound)
		a_estimate - Estiamte of the quantum amplitude
		a - Actual value of the quantum amplitude
		error - Error between the estimate and the actual amplitude
		message - Printable message to console as a string
		SuccessFlag - Integer indicating whether the error violated the upper bound
			SuccessFlag = 0 if upper bound was violated
			SuccessFlag = 1 if upper bound  was NOT violated
		upper_bound - Upper bound of the error between the estiamte and true value of the amplitude
	
	Examples
	--------
	TODO
	"""
	
	true_amplitude = np.sin(np.pi * omega) ** 2
	
	upper_bound = (2 * np.pi / M) * np.sqrt(true_amplitude * (1 - true_amplitude)) + (np.pi / M) ** 2

	# Calculate the total number of random deviates needed
	total_runs = int(np.ceil(-8 * np.log(delta)))
	if (total_runs % 2 == 0): total_runs += 1

	estimates = np.array([randQAEA(M, omega) for i in range(total_runs)])
	
	median_estimate = np.median(estimates)
	
	estimate_amplitude = np.sin(np.pi * median_estimate / M) ** 2

	# Check upper bound on the error
	error = np.abs(estimate_amplitude - true_amplitude)
	if (error <= upper_bound):
		message = "Estimate error upper bound satisfied!"
		SuccessFlag = 1
	else:
		message = "Estimate error upper bound violated!"
		SuccessFlag = 0

	return (estimate_amplitude, true_amplitude, error, message, SuccessFlag, upper_bound)
def QAmpEst2(M, omega, delta, statistics = False):
	"""
	Calculates an estimate for a quantum amplitude. The optional statistics flag lets the function return a tuple contaning
	the estiamte and a pass/fail flag

	THIS FUNCTION IS DEPRICATED and only exists as reference. For implementations, use QAmpEst3()

	Arguments
	---------
	M : int
		Integer number of dimensions in the quantum Hilbert space
	omega : float
		Phase of the quantum amplitude that is being estimated
	delta : float
		Kacewicz probability parameter
	statistics - bool, optional
		This flag determines if pass/fail statistics flag should be included in the function return value. The default value
		is False

	Returns
	-------
	x : {float, tuple[float, bool]}
		float - Estimate of the quantum amplitude
		tuple[float, bool] - Estimate of the quantum amplitude with pass/fail statistic flag

	Examples
	--------
	TODO
	"""

	true_amplitude = QAmpTrue(omega)
	
	upper_bound = QAmpBound(M, true_amplitude)

	total_runs = np.ceil(-8 * np.log(delta)).astype(np.int64)
	if (total_runs % 2 == 0): total_runs += 1

	estimates = randQAEA2(M, omega, shape = (total_runs,))

	estimate_amplitude = np.sin(np.pi * np.median(estimates) / M) ** 2

	if (not statistics): return estimate_amplitude

	# Calculate the pass/fail statistics
	error = np.abs(true_amplitude - estimate_amplitude)
	if (error > upper_bound): return (estimate_amplitude, False)
	
	return (estimate_amplitude, True)
def QAmpEst3(M, omega, delta, statistics = False):
	"""
	Calculates an estimate for the quantum amplitude using a Monte Carlo simulation 

	Arguments
	---------
	M : int
		Dimension of the Hilbert space of the computational basis. For a quantum computer, this should be two to
		the power of the number of qubits used. In Kacewicz algorithm, the value of ``M`` is closely tied to the 
		number of primary and sceondary intervals, thus determining the accuracy of the approximation.
	omega : {float, Array[float]}
		Phase of the unknown quantum amplitude that is being estimated. This can either be a single scalar value, or an array
		of phases. An estimate for all amplitudes will be calculated.
	delta : float
		Kacewicz probability parameter influencing the number of deviates to calculate
	statistics : bool, optional
		Optional flag that when set to True, QAmpEst3() returns a tuple containing the estimate(s) and a boolean pass/fail
		statistic for whether the estimate was within the error bounds

	Returns
	-------
	x : {float, tuple[float, bool]} or {Array[float], tuple[Array[float], Array[bool]]}
		The return type depends on the ``statistics`` flag and whether ``omega`` is a scalar or an array. When the 
		``statistics`` flag is True, tuple is returned, containing the quantum estimate(s) and pass/fail statistics

	Examples
	--------
	TODO
	"""
	
	true_amplitude = QAmpTrue(omega)
	upper_bound = QAmpBound(M, true_amplitude)

	#==============================================================================================#

	total_runs = np.ceil(-8 * np.log(delta)).astype(np.int64)

	if (total_runs % 2 == 0): total_runs += 1

	#==============================================================================================#

	estimates = randQAEA3(M, omega, length = total_runs)
	median_estimate = np.median(estimates, axis = -1)

	#==============================================================================================#

	estimate_amplitude = np.sin(np.pi * median_estimate / M) ** 2

	#==============================================================================================#

	if (not statistics): return estimate_amplitude

	error = np.abs(true_amplitude - estimate_amplitude)
	success = error <= upper_bound

	return (estimate_amplitude, success)

def QAmpTrue(omega):
	return np.sin(np.pi * omega) ** 2

def QAmpBound(M, true_amplitude):
	return (2 * np.pi / M) * np.sqrt(true_amplitude * (1 - true_amplitude)) + (np.pi / M) ** 2

####################################################################################################

if (__name__ == "__main__"):
	# Basic Test of Memory

	for i in range(100):
		MemoryProfiler.record_data_point()

	MemoryProfiler.plot()

	print("COMPLETE")