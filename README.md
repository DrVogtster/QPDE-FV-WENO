This github contains the code associated with generating the figures/tables for the paper "Incorporating high-accuracy, finite-volume shock stabilization methods into a quantum algorithm
for nonlinear partial differential equations" in Physical Review A : https://journals.aps.org/pra/pdf/10.1103/9phc-5m7b
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

- run_tests.py - run this first - it will run all classical and quantum tests. This will generate text files that hold  the convergence tables for the smooth example (both using the quantum and classical solver) that appear in the paper, as well as data files that can be used for plotting.
- plot_data.py - after generating the data, running plot_data.py will make all the figures which appear in the paper.


 # Cite our work
 If you found these codes to useful in academic research, please consider citing our work:
@article{vogt2025incorporating,
  title={Incorporating high-accuracy, finite-volume shock stabilization methods into a quantum algorithm for nonlinear partial differential equations},
  author={Vogt, Ryan and Berjamin, Harold and Rouillard, Hunter and Raff, Edward and Ranade, Priyanka and Collett, Torstein and D'Angelo-Cosme, Roberto and Holt, James and Gaitan, Frank},
  journal={Physical Review A},
  volume={112},
  number={5},
  pages={052431},
  year={2025},
  publisher={APS}
}
 


