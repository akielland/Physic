# code for exercise 1A.6 and 1A.7
# egen kode 
# Den er derfor vektorisert og uten loper. Dvs at Big-O er en konstant funksjon.

import numpy as np
import matplotlib.pyplot as plt
import ast_tools                      # importing ast2000tool packages

class RocketEngine():
    def __init__(self):
        # self.k_b = 1.38064852e-23
        self.k_B = ast_tools.const.k_B
        self.G = ast_tools.const.G
        self.mass_hydrogen_molecule = ast_tools.const.m_H2  # mass of H2 in kg
        
        self.home_planet_idx = 0 # The home planet has index 0
        self.mass_home_planet = ast_tools.system.masses[self.home_planet_idx] * ast_tools.const.m_sun  #1.989e30
        self.radius_home_planet = ast_tools.system.radii[self.home_planet_idx] * 1000
        print("home planet mass: ", self.mass_home_planet/5.972e24) # homeplanet in earth masses
        print("home planet radius: ", self.radius_home_planet/1000) # homeplanet radius in km
        
        self.T = 10_000
        self.mass_satellite = 1000 
        self.N = 100_000
        self.box_length = 1e-6

    def v_esc(self):
        G = self.G
        radius = self.radius_home_planet
        mass = self.mass_home_planet
        return np.sqrt(2*G*mass/radius)

    def initial_values(self):
        # set the initial position and velocity of the particles
        # can also be used to refill the box at respective timesteps, but then preferentially without seed
        np.random.seed(0)
        self.SD = np.sqrt(self.k_B*self.T/self.mass_hydrogen_molecule)
        # print("SD = {:.3f} m/s".format(self.SD))
        self.position = np.random.uniform(low=0, high=self.box_length, size=(self.N, 3))
        self.velocity = np.random.normal(loc=0, scale=self.SD, size=(self.N, 3))
        
    def plot_initial_values(self):
        plt.subplot(121)
        count, bins, ignored = plt.hist(self.velocity, 30, density=True)
        plt.plot(bins, 1/(np.sqrt(2 * np.pi)*self.SD) * np.exp(-(bins)**2 / (2*self.SD**2) ), linewidth=2, color='b')
        plt.subplot(122)
        plt.hist(self.position, 10)
        plt.show() 

    def kinetic_energy_mean(self):
        analytic_kinetic_energy = 3/2*self.k_B*self.T
        squared_velocity = np.sum(self.velocity*self.velocity, axis=1)
        avg_KE = np.sum(squared_velocity*self.mass_hydrogen_molecule/2) / self.N 
        print("Analytic kinetic energy: ", analytic_kinetic_energy)
        print("Numerical average kinetic energy: ", avg_KE)

    def speed_mean(self):
        speed_num = np.linalg.norm(self.velocity, axis=1)
        speed_mean_num = np.sum(speed_num)/self.N
        print("Average speed numerical: ", speed_mean_num)
        # calculating mean absolute velocity/speed analytically
        self.speed_mean_anal = np.sqrt((8*self.k_B*self.T)/(np.pi*self.mass_hydrogen_molecule))  
        print("Average speed analytically: ", self.speed_mean_anal)

    def pressure(self, dt=1e-12, steps=1000):
        r = self.position
        v_out = 0
        for timestep in range(steps):
            r = r + self.velocity*dt
            v_out = v_out + np.sum(np.where(r[:,2] < 0, self.velocity[:,2], 0))
            self.velocity[:,2] = np.where(r[:,2] > 1e-6, -self.velocity[:,2], self.velocity[:,2])
            self.velocity[:,2] = np.where(r[:,2] < 0, -self.velocity[:,2], self.velocity[:,2])
        
       
        self.particle_wall_count = np.count_nonzero(v_out)  # after 1e-9 s

        self.momentum = -v_out * self.mass_hydrogen_molecule * 2
        force = self.momentum/(dt*steps)
        pressure_num = force/(self.box_length**2)
        print("number of particels colliding with the wall: ", self.particle_wall_count)
        print("Numerical  pressure: ", pressure_num)
        # r = self.position # refill cumbustion box
        pressure_anal = self.N * self.k_B * self.T/(1e-6**3)
        print("Analytical pressure: ", pressure_anal)

    def engine(self, dt=1e-12, steps=100):
        L = self.box_length
        r = self.position
        v_out = 0
        v_out_count = 0
        v_out_sum = 0
        for timestep in range(steps):
            r = r + self.velocity*dt

            conditions = (r[:,2] < 0) & ((abs(L/2 - r[:,1]) < L/4) & (abs(L/2 - r[:,0]) < L/4))
            v_out = np.where(conditions,  self.velocity[:,2], 0)           
            v_out_count += np.count_nonzero(v_out)
            v_out_sum += np.sum(v_out)
            self.velocity[:,:] = np.where(r[:,:] > L-L/10, -self.velocity[:,:], self.velocity[:,:])
            self.velocity[:,:] = np.where(r[:,:] < L/10, -self.velocity[:,:], self.velocity[:,:])

            # self.velocity[:,0] = np.where(r[:,0] <= 0, -self.velocity[:,0], self.velocity[:,0])
            # self.velocity[:,1] = np.where(r[:,1] <= 0, -self.velocity[:,1], self.velocity[:,1])
            # self.velocity[:,2] = np.where(r[:,2] <= 0, -self.velocity[:,2], self.velocity[:,2])
            # conditions = (r[:,2] <= 0)# & ((abs(L/2 - r[:,1]) < L/4) & (abs(L/2 - r[:,0]) < L/4))
            # r[:,2] = np.where(conditions, r[:,2]+L, r[:,2])

        plt.subplot(121)
        count, bins, ignored = plt.hist(self.velocity, 30, density=True)
        plt.plot(bins, 1/(np.sqrt(2 * np.pi)*self.SD) * np.exp(-(bins)**2 / (2*self.SD**2) ), linewidth=2, color='b')
        plt.subplot(122)
        plt.hist(r, 100)
        plt.xticks(rotation=30)
        plt.show() 
        
        # self.particle_wall_count = np.count_nonzero(v_out)  # after 1e-9 s

        # self.momentum = -v_out * self.mass_hydrogen_molecule * 2
        # force = self.momentum/(dt*steps)
        # pressure_num = force/(self.box_length**2)
        # print("number of particels colliding with the wall: ", self.particle_wall_count)
        # print("Numerical  pressure: ", pressure_num)
        # # r = self.position # refill cumbustion box
        # pressure_anal = self.N * self.k_B * self.T/(1e-6**3)
        # print("Analytical pressure: ", pressure_anal)     

    def speed_gain(self, dt=1e-12, steps=1000):
        # blir dele på 4 for enkelt?
        p_escape = self.momentum/4
        F_rocket = p_escape/(dt*steps)
        # Euler her... ikke når konstant akselerasjon og konstant masse
        self.a_rocket = F_rocket/self.mass_satellite
        self.speed_gain_rocket_one_box = self.a_rocket*(dt*steps)
        print("speed gain: ", self.speed_gain_rocket_one_box)

    def number_boxes(self, time=20*60):
        # a = self.v_esc()/time
        # self.box_n = a/self.a_rocket
        self.box_n = self.v_esc()/(self.speed_gain_rocket_one_box * time * 1e9)
        print("number of boxes my numbers: {0:1.2e}".format(self.box_n))
        return(self.box_n)

    def fuel_estimate(self):
        particle_escape_count_20min = self.box_n * 20*60/10e-9 * self.particle_wall_count_/4 
        fuel = particle_escape_count_20min * self.mass_hydrogen_molecule
        # print(particle_escape_count)
        print("Fuel [kg]: {0:1.3e}" .format(fuel))

    def rocket_launch(self, box_n=3.96e+12, N=100_000, dt=0.1):
        M = self.mass_home_planet
        radius = self.radius_home_planet
        k = self.k_B
        G = self.G
        time = 20*60
        n = int(time/dt)
        time_steps = np.linspace(0, time, n)

        v = np.zeros(n)
        r = np.zeros(n)
        t=0
        particle_density = box_n*N/self.box_length**3
        m = np.zeros(n)
        initial_fuel = box_n * time * 1e9*self.particle_wall_count_/4 * self.mass_hydrogen_molecule
        used_fuel_dt = box_n  * dt * 1e9*self.particle_wall_count_/4 * self.mass_hydrogen_molecule
        m[0] = self.mass_satellite + initial_fuel
        print("Mass of rocket before launch: ", m[0])
       
        A_whole = (self.box_length/2)**2

        a_thrust = (1/m[t]) * (particle_density*k*A_whole)
        a_gravitation = - G*M/(radius+r[t])**2
        a = a_thrust# + a_gravitation
        for t in range(n-1):
            v[t+1] = v[t] + a*dt
            r[t+1] = r[t] + v[t+1]*dt
            m[t+1] = m[t] - used_fuel_dt
            # Check if escape velocity is achieved
            # if 0.5*v[t+1]**2 - G*M/(radius+r[t]) > 0: 
            #     print("timestep: ", time_steps[t+1])
            #     k = t+1
            #     break
        k = -1
        print("fuel left: {0:1.3g}".format(m[k]))
        plt.subplot(121)
        plt.title("")
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity [m/s]")
        plt.plot(time_steps[0:k],v[0:k])
        plt.subplot(122)
        plt.title("")    
        plt.xlabel("Time [s]")
        plt.ylabel("Height [m]")
        plt.plot(time_steps[0:k],r[0:k])

        # plt.plot(time_steps, r)
        plt.show()

if __name__ == "__main__":
    solar_system_0 = RocketEngine()
    # print("Escape velocity (m/s): ", solar_system_0.v_esc())
    solar_system_0.initial_values()
    # print("SD = {:.3f} m/s".format(solar_system_0.SD))
    # solar_system_0.plot_initial_values()
    # solar_system_0.kinetic_energy_mean()
    # solar_system_0.speed_mean()
    solar_system_0.pressure()
    # solar_system_0.engine()
    # solar_system_0.speed_gain()
    # solar_system_0.number_boxes()
    # solar_system_0.fuel_estimate()
    # solar_system_0.rocket_launch()


