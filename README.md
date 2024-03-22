# Non-ideal-gas-simulation

A computational model of a non-ideal gas with volumetric particles naturally yielding the Maxwell-Boltzmann distribution.

## Contents:
- `model_for_N_collisions.py`: Contains the `model_2D_N_particles` class, which implements the mechanism used to produce the simulation.
- `simulation_for_N_collisions.py`: This Python script imports the `model_2D_N_particles` class to run the simulation and specify its parameters. It also contains Matplotlib code to produce the Maxwell-Boltzmann distribution after completing the simulation.

## Usage:
1. Ensure that both `model_for_N_collisions.py` and `simulation_for_N_collisions.py` are in the same directory.
2. Execute `simulation_for_N_collisions.py` to start the simulation. You will see a progress bar indicating the simulation's progress, and another window will open to display the simulation.
3. Once the simulation has completed, the simulation window will close automatically, and a histogram will be displayed showing the resulting velocity distribution of the particles.

