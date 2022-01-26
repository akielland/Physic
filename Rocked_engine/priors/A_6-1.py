# code for exercise 1A.6 and 1A.7
# egen kode

import numpy as np
import matplotlib.pyplot as plt
import ast_tools                      # importing ast2000tool packages

class RocketEngine():
    def __init__(self, T=10_000, mass=1000, box_length=1e-6):
        self.k_B = ast_tools.const.k_B
        self.G = ast_tools.const.G
        self.mass_hydrogen_molecule = ast_tools.const.m_H2  # mass of H2 in kg
        self.home_planet_idx = 0 # The home planet has index 0
        self.mass_home_planet = ast_tools.system.masses[self.home_planet_idx] * ast_tools.const.m_sun  # [kg]
        self.radius_home_planet = ast_tools.system.radii[self.home_planet_idx] * 1000  # [m]
        # print("home planet mass: ", self.mass_home_planet/5.972e24) # homeplanet in earth masses
        # print("home planet radius: ", self.radius_home_planet/1000) # homeplanet radius in km
        
        self.T = T
        self.mass_satellite = mass 
        self.N = 100_000
        self.box_length = box_length

    def v_esc(self):
        # Calculate escape velocity
        G = self.G
        radius = self.radius_home_planet
        mass = self.mass_home_planet
        return np.sqrt(2*G*mass/radius)

    def initial_values(self):
        # set the initial position and velocity of the particles
        np.random.seed(0)
        self.SD = np.sqrt(self.k_B*self.T/self.mass_hydrogen_molecule)
        # print("SD = {:.3f} m/s".format(self.SD))
        self.position = np.random.uniform(low=0, high=self.box_length, size=(self.N, 3))
        self.velocity = np.random.normal(loc=0, scale=self.SD, size=(self.N, 3))

    def plot_engine_gas(self):
        # plot position and velocity of the particles in the box to check the code at initiation and after running the engine
        plt.subplot(121)
        count, bins, ignored = plt.hist(self.velocity, 30, density=True)
        plt.plot(bins, 1/(np.sqrt(2 * np.pi)*self.SD) * np.exp(-(bins)**2 / (2*self.SD**2) ), linewidth=2, color='b')
        plt.xlabel("bins")
        plt.ylabel("number of particles")
        plt.subplot(122)
        plt.hist(self.position, 30)
        plt.xlabel("bins")
        plt.xticks(rotation=30)
        plt.tight_layout()
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
        # numerical estimate of the pressure in the box
        L = self.box_length
        r = self.position
        v = self.velocity
        wall_adjust = L/10 # adjust for the fact that that the average turning point should be at the wall not after the wall
        v_out = 0
        for timestep in range(steps):
            r = r + self.velocity*dt
            v_out = v_out + np.sum(np.where(r[:,2] < wall_adjust, v[:,2], 0))
            v[:,2] = np.where(r[:,2] > L-wall_adjust, -v[:,2], v[:,2])
            v[:,2] = np.where(r[:,2] < wall_adjust, -v[:,2], v[:,2])

        force = - 2 * v_out * self.mass_hydrogen_molecule/(dt*steps)
        pressure_num = force/(self.box_length**2)
        print("Numerical  pressure: {:.0f} Pa".format(pressure_num))
        pressure_anal = self.N * self.k_B * self.T/(self.box_length**3)
        print("Analytical pressure: {:.0f} Pa".format(pressure_anal))

    def engine(self, dt=1e-12, steps=1000):
        # Calculate momentum and particles leaving the box over time
        L = self.box_length
        r = self.position
        
        v_out = 0
        v_out_count = 0
        v_out_sum = 0
        for timestep in range(steps):
            r = r + self.velocity*dt
            conditions = (r[:,2] < L/10) & ((abs(L/2 - r[:,1]) < L/4) & (abs(L/2 - r[:,0]) < L/4))
            v_out = np.where(conditions,  self.velocity[:,2], 0)     
            v_out_count += np.count_nonzero(v_out)
            v_out_sum += np.sum(v_out)
            self.velocity = np.where(r > L-L/10, -self.velocity, self.velocity)
            self.velocity = np.where(r < L/10,   -self.velocity, self.velocity)

        self.momentum = - v_out_sum * self.mass_hydrogen_molecule
        print("Momentum leaving one box after {:.3g} s: {:.3e}".format(dt*steps, self.momentum))
        self.position = r
        # self.velocity = v
        self.particle_out_count = v_out_count
        print("Particles leaving one box after {:.3g} s: {:.3g}".format(dt*steps, self.particle_out_count))

    def speed_gain(self):
        speed_gain_rocket_one_box = self.momentum/self.mass_satellite
        print("speed gain: ", speed_gain_rocket_one_box)
        return speed_gain_rocket_one_box

    def number_boxes(self, timestep=1e-9, time=20*60):
        self.box_n = self.v_esc()/(self.speed_gain()/timestep * time)
        print("number of boxes: {0:1.2e}".format(self.box_n))
        return(self.box_n)

    def fuel_estimate(self, timestep=1e-9, time=20*60):
        particle_escape_count_20min = self.number_boxes() * self.particle_out_count * time/timestep
        fuel_mass = particle_escape_count_20min * self.mass_hydrogen_molecule
        print(fuel_mass/1200)
        print("Fuel [kg]: {:.1f}" .format(fuel_mass))

    def rocket_launch(self, box_n=7.95e+12, fuel=80501):
        dt=0.1
        m_planet = self.mass_home_planet
        radius = self.radius_home_planet
        k_B = self.k_B
        G = self.G
        time = 20*60
        n = int(time/dt)
        # initial_fuel = fuel
        # print("fuel: ", initial_fuel)
        time_steps = np.linspace(0, time, n)
        v = np.zeros(n)
        r = np.zeros(n)
        m = np.zeros(n)
        
        A_hole = (self.box_length/2)**2
    
        pressure = self.N/(self.box_length**3) * k_B * self.T
        initial_fuel = time * box_n * 1e9*self.particle_out_count * self.mass_hydrogen_molecule
        used_fuel_dt = box_n * 1e9*self.particle_out_count * self.mass_hydrogen_molecule
        m[0] = self.mass_satellite + initial_fuel
        print("Mass of rocket before launch: ", m[0])
        k = -1
        last_time_point = time_steps[-1]
        for t in range(n-1):
            a_thrust = (1/m[t]) * box_n * (1/2) * pressure * A_hole
            a_gravitation = - G * m_planet/(radius+r[t])**2
            a = a_thrust + a_gravitation
            v[t+1] = v[t] + a*dt
            r[t+1] = r[t] + v[t+1]*dt
            m[t+1] = m[t] - used_fuel_dt * dt
            
            if v[t+1] < 0:
                # adjust velcity to zero if initial negative velocity
                v[t+1] = 0
    
            if ((1/2)*v[t+1]**2 - G*m_planet/(radius+r[t])) > 0: 
                # stop the loop when escape velocity is achieved
                last_time_point = time_steps[t+11]
                k = t+1
                break
    
        fuel_left = m[k-1] - self.mass_satellite
        print("Fuel left: {:.3g} kg".format(fuel_left))
        print("Velocity reached: ", v[k])
        print("Time to engine stop: {:.0f}".format(last_time_point))
        
        plt.subplot(131)
        plt.title("")
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity [m/s]")
        plt.plot(time_steps[0:k], v[0:k])
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    solar_system_0 = RocketEngine(T=10_000, mass=1000, box_length=0.000_0001)
    # print("Escape velocity (m/s): ", solar_system_0.v_esc())
    solar_system_0.initial_values()
    # print("SD = {:.3f} m/s".format(solar_system_0.SD))
    # solar_system_0.kinetic_energy_mean()
    # solar_system_0.speed_mean()
    # solar_system_0.pressure()
    solar_system_0.engine()
    # solar_system_0.speed_gain()
    # solar_system_0.number_boxes()
    # solar_system_0.fuel_estimate()
    # solar_system_0.rocket_launch(box_n=1e+14)
    solar_system_0.plot_engine_gas()
