# Setting up an environment for running the codes & running the codes: 

If you're on a Linux machine, to keep things straightforward, we have added a 'fvweno.yml' file that can be imported via conda: 
	conda env create -f fvweno.yaml 
	conda activate fvweno In this conda environment
 you can then execute the codes. 
 
 NOTE: The 'fvweno.yaml' is configured specifically for Linux operating system. If you don't use Linux or conda, you need to manually download the following: 
 - jax
 - scipy
 - numpy
 - dill
 - prettytable
 - psutil
 - wcwidth 
 
 You might need a few additional libraries (don't worry, the program will alert you about any missing libraries so you can download them promptly). There are two folders: Classical and Quantum. The Classical folder contains files that solve the 1-D Euler gas law in three different scenarios (problem resulting in smooth solution, Lax problem (Riemann problem), and Sod problem (Riemann problem) using an RK4 (fourth order) time integration method and a fifth-order WENO spatial discretization. The Quantum folder houses code that employs a fourth-order quantum ODE time integration method and a fifth-order WENO spatial discretization.
In either folder:

A. How to use the Python scripts: 
- Open 'euler_hb.py', the primary file.
- In the Configuration section, you have the option to load a pre-defined test. Alternatively, you can configure custom parameters by uncommenting the specified blocks.
- Change the mesh size 'Nx' and Courant number 'Co' as needed.
-  Executing a pre-defined test: Open 'euler_tests.py', the configuration file.
-   Keep the shared parameters as default unless necessary.
-   Choose the appropriate 'test' configuration from the provided list (parameters should remain unchanged unless required).
-   For test 2, configure the Riemann data according to the examples for Lax & Sod (parameters should remain unchanged unless required).
-  Conducting error measurements: 1. Run the first test in 'euler_tests.py'.
-   Errors will be shown in the terminal (and text files will be generated containing the same information)
-   For invoking the quantum algorithm in euler_hb within the Quantum directory: 1) Adjust epsilon (the upper limit on numerical error between the numerical and exact solution), and delta (the probability that this upper limit is met) (refer to quantum algorithm details for more information). 2) Modify n_samples (the degree of the Legendre polynomial used in Gauss quadrature) - we use n_samples=2 for a 4th-order time-stepping method (2*n_samples).,
If using test=2, choose which instance of the Riemann problem to solve (Lax or Sod) by commenting/uncommenting the relevant initial condition sections in euler_tests.py.


For every problem you attempt to solve, plots will be automatically generated. Currently, the program is set to tackle a series of problems with finite volume cells Nx=16,32,64,...,1024. The plot names will specify the corresponding solution and Nx value they pertain to.

Figures/tables in paper and what code generated them:
- Fig 1 -  output from running euler.hb in Classical folder with test=2
- Table 2, 4 - output from running euler.hb in Classical folder with test=1
- Fig 2,3 / Table 1,3 - output from running euler.hb in Quantum folder with test = 1
- Fig 4, 5, 6 - output from running euler.hb in Quantum folder with test = 2 (and editing euler_tests.py to set Sod initial conditions - see below)
- Fig 7, 8, 9 - output from running euler.hb in Quantum folder with test = 2 (and editing euler_tests.py to set Lax initial conditions - see below)


For example to run Lax modify euler_tests.py,:

	# Lax
	rhoJ = np.array([0.445, 0.5]) # rhoJ = np.array([0.445, 0.5])
	uJ = np.array([0.698, 0])     # uJ = np.array([0.698, 0])
	pJ = np.array([3.528, 0.571]) # pJ = np.array([3.528, 0.571])
	# # # Sod
	# rhoJ = np.array([1, 0.125]) # rhoJ = np.array([1, 0.125])
	# uJ = np.array([0, 0])       # uJ = np.array([0, 0])
	# pJ = np.array([1, 0.1])     # pJ = np.array([1, 0.1])

To run Sod, instead comment the Lax initial conditions and uncomment Sod initial conditions. 
 # Cite out work
 If you found these codes to useful in academic research, please cite:
- include Bibtex + google scholar reference later (after publication).
 



