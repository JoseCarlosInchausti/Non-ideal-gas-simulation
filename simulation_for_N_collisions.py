from model_for_N_collisions import model_2d_N_particles
import matplotlib.pyplot as plt
import numpy as np

L=1
t_total=60
v_o=2
n_particles=500
m=1.67e-27

model=model_2d_N_particles(v_o=v_o,m=m,L=L,t_total=t_total,frames_per_sec=100,n_particles=n_particles,simulation_name='Test')
x,y,t,v,T,E_f,E_o=model.initialize_model(animation=True,save_animation=False)

fs=15
num_bins=50

plt.figure()
counts, bins, _ = plt.hist(v,bins=num_bins,density=True)
plt.title('Distribution of Speeds After\nt={}s of {} Particles Colliding'.format(t_total,n_particles),size=fs)
plt.xlabel(r'Speed $[ms^{-1}]$',size=fs)
plt.ylabel('Number of particles',size=fs)
plt.show()

print('Initial energy: {} J'.format(E_o[0]))
print('Final energy: {} J'.format(E_f))
