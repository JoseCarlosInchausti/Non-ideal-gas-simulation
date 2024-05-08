from model_for_N_collisions import model_2d_N_particles
import matplotlib.pyplot as plt
import numpy as np
L=1
t_total=30
v_o=0
n_particles=500
m=1

model=model_2d_N_particles(v_o=v_o,m=m,L=L,t_total=t_total,frames_per_sec=100,n_particles=n_particles,simulation_name='Test')
x,y,t,v=model.initialize_model(LJ_potential=True,animation=True,save_animation=False,save_history=False,save_last_only=False,history_path='LJ_1_sample.pkl',epsilon=0.001,sigma=0.001)


'''
Uncomment for Monte Carlo sampling
'''
# n_samples=100
# for n in range(n_samples):
#     print('Sampling {}/{}'.format(n,n_samples))
#     try:
#         model=model_2d_N_particles(v_o=v_o,m=m,L=L,t_total=t_total,frames_per_sec=100,n_particles=n_particles,simulation_name='Test')
#         x,y,t,v=model.initialize_model(LJ_potential=True,animation=False,save_animation=False,save_history=True,epsilon=0.0001,sigma=0.001,history_path='100_iterations_LJ/iteration_{}.pkl'.format(n))
#     except:
#         print('Unexpected error, re-starting sample')
#         model=model_2d_N_particles(v_o=v_o,m=m,L=L,t_total=t_total,frames_per_sec=100,n_particles=n_particles,simulation_name='Test')
#         x,y,t,v=model.initialize_model(LJ_potential=True,animation=False,save_animation=False,save_history=True,epsilon=0.0001,sigma=0.001,history_path='100_iterations_LJ/iteration_{}.pkl'.format(n))
        
#Histogram plotting
fs=15
num_bins=35#50

plt.figure()
counts, bins, _ = plt.hist(v,bins=num_bins,density=True)
plt.title('Distribution of Speeds After\nt={}s of {} Particles Colliding'.format(t_total,n_particles),size=fs)
plt.xlabel(r'Speed $[ms^{-1}]$',size=fs)
plt.ylabel('Number of particles',size=fs)
plt.show()