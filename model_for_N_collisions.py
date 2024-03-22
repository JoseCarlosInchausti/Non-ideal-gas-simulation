import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import display
from matplotlib.animation import FuncAnimation
import pygame
from scipy.spatial.distance import pdist, squareform
import imageio
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class model_2d_N_particles:
    def __init__(self,L=1,v_o=0.2,m=1,t_total=10,frames_per_sec=1000,verbose=False,n_particles=2,simulation_name='Test'):
        self.n_particles=n_particles
        self.L=L
        self.v_o=v_o
        self.positions=np.random.uniform(low=-self.L/2, high=self.L/2, size=(self.n_particles,2))
        self.v=np.full((self.n_particles,2),0.0001)#All particles are static
        self.v[0]=np.array([self.v_o,self.v_o])#One highly energetic particle
        self.m=np.full((n_particles,1),m)
        self.t_total=t_total
        self.frames_per_sec=frames_per_sec
        self.simulation_name=simulation_name
        self.x_positions_record=[]
        self.y_positions_record=[]
        
        self.sigma_x=self.v[:,0]*(1/self.frames_per_sec)
        
        self.sigma_y=self.v[:,1]*(1/self.frames_per_sec)
        
        self.t=0
        self.t_record=[]

        self.PARTICLE_RADIUS=1
        self.PARTICLE_DIAMETER=2*self.PARTICLE_RADIUS

        self.WIDTH, self.HEIGHT = 600,600
        self.PARTICLE_COLOR = (255, 255, 255)

        self.PARTICLE_COLLISION_COLOR = (255, 0, 0)
        self.R=L*self.PARTICLE_RADIUS/600 #Radius in "SI"
        self.D=2*self.R #Diameter in "SI"
        self.BACKGROUND_COLOR = (0, 0, 0)
        
        #constants:
        self.k_b=1.380649e-23
        
    def cartesian_to_pixel(self,cartesean_positions):
        translated_positions=cartesean_positions+np.full(cartesean_positions.shape,self.L/2)
        scaled_positoons=translated_positions*self.WIDTH/self.L
        return scaled_positoons

    def initialize_model(self,animation=True,save_animation=False):
        frames=[]
        if animation:
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))#pygame.FULLSCREEN
            pygame.display.set_caption("Ideal Gas Simulation")
            clock=pygame.time.Clock()

        for i in tqdm(range(int(self.frames_per_sec*self.t_total)),unit='frames',desc='Running 2D simulation'):
            self.update_location()
            self.check_collision_box()
            colliding_paritcle_indices=self.check_particle_collision()
            self.update_velocity_for_colliding_particles(colliding_paritcle_indices)
            if animation:
                self.screen.fill(self.BACKGROUND_COLOR)
                for i,location in enumerate(self.cartesian_to_pixel(self.positions)):
                    if i == 0:
                        pygame.draw.circle(self.screen, self.PARTICLE_COLLISION_COLOR,(location[0],location[1]),self.PARTICLE_RADIUS)
                    else:
                        pygame.draw.circle(self.screen, self.PARTICLE_COLOR,(location[0],location[1]),self.PARTICLE_RADIUS)
                pygame.display.flip()
                if save_animation:
                    frames.append(pygame.surfarray.array3d(self.screen.copy()))
            self.t+=1/self.frames_per_sec
            self.t_record.append(self.t)
            if animation:
                clock.tick(self.frames_per_sec)
        if save_animation:
            print('Saving animation as {}.gif'.format(self.simulation_name))
            imageio.mimsave("{}.gif".format(self.simulation_name), frames, duration=1/self.frames_per_sec)
        if animation:
            pygame.quit()
        speeds=self.compute_speeds()
        v_rms=np.sqrt(np.sum(speeds**2)/self.n_particles)
        T=1*v_rms**2/(2*self.k_b)
        E_o=1/2*self.m[0]*2*self.v_o**2+1/2*1*0.0001**2*(self.n_particles-1)
        E_f=np.sum(1/2*self.m*self.v**2)#self.k_b*self.n_particles*T
        return self.x_positions_record,self.y_positions_record,self.t_record,speeds,T,E_f,E_o
    
    def check_particle_collision(self):
        distances = pdist(self.positions)
        distance_matrix = np.round(squareform(distances),3)#Matrix is rounded to 3 decimal places to avoid particles overlapping
        colliding_paritcle_indices=[]
        registered_particle_indices=[]
        for n,distances_for_nth_particle in enumerate(distance_matrix):
            if n not in registered_particle_indices:
                if np.where(distances_for_nth_particle<=self.D)[0].shape[0]==2:
                    colliding_paritcle_indices.append(np.where(distances_for_nth_particle<=self.D)[0])
                    registered_particle_indices.append(np.where(distances_for_nth_particle<=self.D)[0][0])
                    registered_particle_indices.append(np.where(distances_for_nth_particle<=self.D)[0][1])
        return np.array(colliding_paritcle_indices)

    def compute_speeds(self):
        speeds=np.linalg.norm(self.v, axis=1)
        return speeds

    def update_velocity_for_colliding_particles(self,colliding_paritcle_indices):
        '''
        colliding_paritcle_indices: np array, rows are pairs, columns contain indices of colliding particles
        '''
        for i_pair in colliding_paritcle_indices:
            v_A,v_B=self.v[i_pair[0]],self.v[i_pair[1]]
            m_A,m_B=self.m[i_pair[0]],self.m[i_pair[1]]
            v=self.v[i_pair]
            r_A=self.positions[i_pair][0]
            r_B=self.positions[i_pair][1]
            n_hat=(r_B-r_A)/np.linalg.norm(r_B-r_A) #New basis n (normalized)
            t_hat=np.array([-n_hat[1],n_hat[0]])/np.linalg.norm(np.array([-n_hat[1],n_hat[0]])) #New basis t (normalized)
            v_A_new_frame,v_B_new_frame=np.array([np.dot(v_A,n_hat),np.dot(v_A,t_hat)]),np.array([np.dot(v_B,n_hat),np.dot(v_B,t_hat)])
            v_A_new_frame_prime,v_B_new_frame_prime=np.copy(v_A_new_frame), np.copy(v_B_new_frame)
            v_A_new_frame_prime[0],v_B_new_frame_prime[0]=(v_A_new_frame[0]*(m_A-m_B)+2*m_B*v_B_new_frame[0])/(m_A+m_B),(v_B_new_frame[0]*(m_B-m_A)+2*m_A*v_A_new_frame[0])/(m_A+m_B)
            v_A_prime,v_B_prime=v_A_new_frame_prime[0]*n_hat+v_A_new_frame_prime[1]*t_hat,v_B_new_frame_prime[0]*n_hat+v_B_new_frame_prime[1]*t_hat
            self.v[i_pair[0]],self.v[i_pair[1]]=v_A_prime,v_B_prime

    def update_location(self):
        self.positions=self.positions+self.v*(1/self.frames_per_sec)
        self.x_positions_record.append(self.positions[:,0])
        self.y_positions_record.append(self.positions[:,1])
        
    def check_collision_box(self):
        for n_particle in range(self.n_particles):
            current_x=self.positions[n_particle][0]
            current_y=self.positions[n_particle][1]
            #Check x position:
            if current_x >= self.L/2 - self.sigma_x[n_particle]:
                self.v[n_particle][0]=-1*self.v[n_particle][0]
            elif current_x <= - self.L/2 + self.sigma_x[n_particle]:
                self.v[n_particle][0]=-1*self.v[n_particle][0]
            #Check y position:
            if current_y >= self.L/2 - self.sigma_y[n_particle]:
                self.v[n_particle][1]=-1*self.v[n_particle][1]
            elif current_y <= - self.L/2 + self.sigma_y[n_particle]:
                self.v[n_particle][1]=-1*self.v[n_particle][1]
  

