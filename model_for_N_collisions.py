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
import pickle
warnings.filterwarnings("ignore", category=RuntimeWarning)

class model_2d_N_particles:
    def __init__(self,L=1,v_o=0.2,m=1,t_total=10,frames_per_sec=1000,n_particles=2,simulation_name='Test'):
        """
        
        Initializes the model_2d_N_particles object

        Parameters:
        L (int or float): length of one of the sides of the square particle container in 'meters'
        v_o (int or float): initial speed of the energetic particle (all others start from rest)
        m (int or float): particle mass in 'kg'
        t_total (int or float): duration of the simulation in seconds
        frames_per_sec (int): number of frames per second
        n_particles (int): total number of particles including the energetic particle
        simulation_name (str): name for the simulation run, which will be used as the animation file name if saved later on

        """
        self.n_particles=n_particles
        self.L=L
        self.v_o=v_o
        
        self.m=np.full((n_particles,1),m)
        self.t_total=t_total
        self.frames_per_sec=frames_per_sec
        self.simulation_name=simulation_name
        self.x_positions_record=[]
        self.y_positions_record=[]
        
        self.positions=np.random.uniform(low=-self.L/2, high=self.L/2, size=(self.n_particles,2))
        
        self.v=np.full((self.n_particles,2),0.001)#All particles are static
        self.a=np.zeros((self.n_particles,2))#All particles are instantiated with no acceleration
        self.sigma_x=self.v[:,0]*(1/self.frames_per_sec)
        self.sigma_y=self.v[:,1]*(1/self.frames_per_sec)
        self.v[0]=np.array([self.v_o,self.v_o])/np.sqrt(2) #One highly energetic particle
        self.t=0
        self.t_record=[]
        
        #Pressure computation
        self.t_P_average=2
        self.t_since_P=0 #Time since pressure average was last computed
        self.accumulated_momentum=0
        self.P=[]#Pressures get stored here
        
        #Particle dimensions
        self.R=2/600 #Radius in "SI"
        self.D=2*self.R #Diameter in "SI"
        self.PARTICLE_RADIUS=self.R*600/L #Radius in pixels (for PyGame)
        self.PARTICLE_DIAMETER=2*self.PARTICLE_RADIUS #Diameter in pixels (for PyGame)

        self.WIDTH, self.HEIGHT = 600,600#They MUST be the same, or else change the translation code (the scaling in particular)
        self.PARTICLE_COLOR = (255, 255, 255)

        self.PARTICLE_COLLISION_COLOR = (255, 0, 0)
        self.BACKGROUND_COLOR = (0, 0, 0)
        #constants:
        self.k_b=1.380649e-23
        
    def cartesian_to_pixel(self,cartesean_positions):
        """
        
        PyGame locates the particles with x and y coordiantes in pixels, while all computations are performed in meters.
        This function converts particle positions in meters to pixels for PyGame compatibility.
        
        Initializes the model_2d_N_particles object

        Parameters:
        cartesean_positions (numpy array): array with the x, y cartesean coordinates of all particles, expected shape (2,N)
        
        Returns:
        scaled_positons (numpy array): same shape as input array, but coordinates are converted to pixels

        """
        translated_positions=cartesean_positions+np.full(cartesean_positions.shape,self.L/2)
        scaled_positons=translated_positions*self.WIDTH/self.L
        return scaled_positons

    def initialize_model(self,LJ_potential=False,animation=True,save_animation=False,save_history=False,save_last_only=True,history_path='simulation_history.pkl',epsilon=0.1,sigma=0.01):
        '''
        
        Executes the simulation with all given paramters for the alloted t_total time and may record data as specified.
        Once executed, a progress bar displays the progress of the simulation.
        
        Parameters:
        LJ_potential (bool): if True, the modified Lennard-Jonnes potential is turned on and default particle collisions are turned off
        animation (bool): if True, a live simulation animation is displayed on a PyGame window, if False simulation is executed without animation
        save_animation (bool): determines if simulation is saved, if True, simulation is saved as 'simulation_name'.gif under this file's directory,
        save_history (bool): if true, the positions, velocities, and pressure are saved as a pickle file
        save_last_only (bool): if true, only the last frame's positions, velocities, and pressure are saved, given that save_history=True
        history_path (string): pickle file name (.pkl) under which the simulation history will be saved if save_history=True
        epsilon (float): if LJ_potential=True, epsilon determines the depth of the potential well
        sigma (float): if LJ_potential=True, sigma determines the distance beyond which the potential goes from attractive to repulsive
        
        Returns:
        self.x_positions_record (np.array): 1D array with x coordinate in meters of all particles at each frame
        self.y_positions_record (np.array): 1D array with y coordinate in meters of all particles at each frame
        self.t_record (np.array): 1D array with the time-stamp of each frame
        speeds: 1D array with the speeds of all the particles at the last frame
        
        '''
        frames=[]
        history_velocities=[]
        history_positions=[]
        if animation:
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))#pygame.FULLSCREEN
            pygame.display.set_caption("Ideal Gas Simulation")
            clock=pygame.time.Clock()

        for i in tqdm(range(int(self.frames_per_sec*self.t_total)),unit='frames',desc='Running 2D simulation'):
            if save_history:
                history_velocities.append(np.copy(self.v))
                history_positions.append(np.copy(self.positions))
            self.update_location()
            self.check_collision_box()
            
            if not LJ_potential:
                #Inter-particle collisions, no forces:
                colliding_paritcle_indices=self.check_particle_collision()
                self.update_velocity_for_colliding_particles(colliding_paritcle_indices)
            else:
                #Inter-particle forces, no collisions:
                self.update_acceleration_and_velocities(epsilon=epsilon,sigma=sigma)
                
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
            
        if save_history:
            if save_last_only:
                # all_history={'Positions':history_positions[-1],'Velocities':history_velocities[-1],'Pressures':np.average(np.array(self.P))}
                all_history={'Positions':history_positions[-1],'Velocities':history_velocities[-1]}
                
            else:
                # all_history={'Positions':history_positions,'Velocities':history_velocities,'Pressures':np.array(self.P)}
                all_history={'Positions':history_positions,'Velocities':history_velocities}
            with open(history_path, 'wb') as file:
                pickle.dump(all_history, file)

        speeds=self.compute_speeds()
        
        return self.x_positions_record,self.y_positions_record,np.array(self.t_record),speeds
    
    def check_particle_collision(self):
        '''
        
        Determines the index of the particles that are within colliding distance.
        
        Returns:
        colliding_paritcle_indices (np.array): (2,N) array with the index of colliding pairs of particles.
        
        '''
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
        '''
        
        Convers velocity vectors for all particles into a 1D vector with speeds.
        
        Returns:
        speeds (np.array): 1D vector with all particles' speed in meters per second
        
        '''
        speeds=np.linalg.norm(self.v, axis=1)
        return speeds

    def update_velocity_for_colliding_particles(self,colliding_paritcle_indices):
        '''
        
        Computes the post-collision velocities of colliding particles. It directly modifies self.v, the global array containg velocities.

        Parameters:
        colliding_paritcle_indices (numpy array): 2D array contaiung pairs of colliding particles as determined by self.check_particle_collision()
        
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
        '''
        
        Updates the location of each particle in every frame according to the particles' velocities.
        It modifies the global self.positions array that contains the position vector for each particle.
        
        '''
        self.positions=self.positions+self.v*(1/self.frames_per_sec)
        self.x_positions_record.append(self.positions[:,0])
        self.y_positions_record.append(self.positions[:,1])
        
    def check_collision_box(self):
        '''
        
        Determines and updates the velocity of particles within colliding distance to the walls of the container.
        It modifies the global self.v array that contains the velocity vector for each particle.
        
        '''
        if self.t_since_P >= self.t_P_average and self.t >= self.t_total/2:
            self.t_since_P=0
            self.P.append(self.accumulated_momentum/(self.t_P_average*self.D*self.L))
            self.accumulated_momentum=0
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
                #compute pressure:
                self.accumulated_momentum+=2*self.m[n_particle]*abs(self.v[n_particle][1])
        self.t_since_P+=1/self.frames_per_sec
        
    def update_acceleration_and_velocities(self,epsilon,sigma):
        '''
        
        Calculates the force and acceleration experienced by each particle resulting from the modified Lennard-Jonnes potential.
        It modifies the global self.v array that contains the velocity vector for each particle.
        
        '''
        all_particles_i=np.arange(self.n_particles)
        x1,y1=self.positions[:,0],self.positions[:,1]
        n=6.01
        # n=12
        for i_particle,particle_position in enumerate(self.positions):
            x2,y2=particle_position[0],particle_position[1]
            idxs=all_particles_i!=i_particle
            r=np.sqrt((x2-x1[idxs])**2+(y2-y1[idxs])**2)
            # LJ potential:
            # F_x=np.abs((-4*epsilon/r)*(6*sigma**6/r**7 - n*sigma**n/r**(n+1)))*(x2-x1[idxs])
            # Modified potential:
            F_x=np.abs(-epsilon*((sigma-r+1)/np.exp(-sigma+r-1)))*(x2-x1[idxs])
            F_x=np.sum(-F_x)
            # LJ potential:
            # F_y=np.abs((-4*epsilon/r)*(6*sigma**6/r**7 - n*sigma**n/r**(n+1)))*(y2-y1[idxs])
            # Modified potential:
            F_y=np.abs(-epsilon*((sigma-r+1)/np.exp(-sigma+r-1)))*(y2-y1[idxs])
            F_y=np.sum(-F_y)
            self.a[i_particle]=np.array([F_x/self.m[i_particle],F_y/self.m[i_particle]]).reshape(2)
        self.v=self.v+self.a*(1/self.frames_per_sec)