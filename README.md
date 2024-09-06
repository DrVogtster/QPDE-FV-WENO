Getting an environment to run the codes:

If you are running on a linux machine, for  simplicity, we have added a file 'fvweno.yml' that can be imported via condo:

	conda env create -f fvweno.yaml
	conda activate fvweno

In this conda environment you can then run the codes

NOTE: the fvweno.yaml is confirgured for a linux machine. If you don't have linux or you don't use conda one just needs to download:

-jax
-scipy
-numpy
-dill
-prettytable
-psutil
-wcwidth

You might need a few more libraries (don't worry the program will error out and say you don't have a library that you can then promptly download.)


Classical runs:

See folder named "Classical"

A. To use the python code:
1. Open 'euler_hb.py', the main file.
2. In the Configuration section, you can import a pre-defined test
   (see section B. below). Otherwise, you can set custom parameters
   in the dedicated section (to uncomment blockwise).
3. You can set the graphics output 'plots', where
	0 does not produce graphics
	1 displays density
	2 displays momentum
	3 displays energy
4. You can set the mesh size 'Nx' and Courant number 'Co' below.


B. To run a pre-defined test
1. Open 'euler_tests.py', which is a configuration file.
2. Unless needed, leave the common parameters unchanged.
3. Select the desired configuration 'test' whose parameters are shown below
   (to be left unchanged unless needed).
4. For the selection of the Riemann data (test 2),
   set the custom Riemann data similarly to the Lax & Sod examples above
   (to be left unchanged unless needed).


C. To perform error measurements
1. Select first test in 'euler_tests.py' and run the simulation.
2. Errors are displayed in the terminal.


Quantum runs:
See folder named "Quantum"

Very similar workflow to the "Classical" instructions

1)One can change epsilon (upper bound on numerical error between the numerical and exact solution, delta (1-delta is probability this upper bound is satisfied)  (see quantum algorithm for details))
2) One can also change n_samples (degree of legendre polynomial used in gauss-quadature) - but we set n_samples=2 to recover a 2*n_samples=4th order time stepping method.

If running test=1 then a error/convergence study is eventually reported to the terminal and a text file is generated in the directory with the same error/convergence study.

If test=2 you can pick which instance of the Riemann problem is to be solved (Lax or Sod) by commenting/uncommenting the section that specifices the initial conditions associated with each problem in euler_tests.py.

For example:

	# Lax
	rhoJ = np.array([0.445, 0.5]) # rhoJ = np.array([0.445, 0.5])
	uJ = np.array([0.698, 0])     # uJ = np.array([0.698, 0])
	pJ = np.array([3.528, 0.571]) # pJ = np.array([3.528, 0.571])
	# # # Sod
	# rhoJ = np.array([1, 0.125]) # rhoJ = np.array([1, 0.125])
	# uJ = np.array([0, 0])       # uJ = np.array([0, 0])
	# pJ = np.array([1, 0.1])     # pJ = np.array([1, 0.1])

