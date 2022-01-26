import numpy as np
import matplotlib.pyplot as plt
import my_solar_system as my        # importing parameters from my solar system (se file: my_solar_system.py)

class RocketEngine():
    def __init__(self):
        # self.k_b = 1.38064852e-23
        self.k_B = my.system.k_B
        # self.G = 6.67408e-11
        self.G = my.const.G
        print(self.G)
        # self.mass_hydrogen_molecule = 2*1.00784/6.02214076e23
        self.mass_hydrogen_molecule = my.const.m_H2  # mass of H2 in kg
        # self.SD = np.sqrt(self.k_b*self.T/self.mass_hydrogen_molecule)
        
        self.home_planet_idx = 0 # The home planet has index 0
        self.mass_home_planet = my.system.masses[self.home_planet_idx] * my.const.m_sun  #1.989e30
        print("home planet mass: ", self.mass_home_planet)
        self.radius_home_planet = my.system.radii[self.home_planet_idx] * 1000
        print("home planet radius: ", self.radius_home_planet)
        
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
        print("SD = {:.3f} m/s".format(self.SD))
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
        analytic_kinetic_energy = 3/2*self.k_b*self.T
        squared_velocity = np.sum(self.velocity*self.velocity, axis=1)
        avg_KE = np.sum(squared_velocity*self.mass_hydrogen_molecule/2) / self.N 
        print("Analytic kinetic energy: ", analytic_kinetic_energy)
        print("Numerical average kinetic energy: ", avg_KE)

    def speed_mean(self):
        speed_num = np.linalg.norm(self.velocity, axis=1)
        speed_mean_num = np.sum(speed_num)/self.N
        print("Average speed numerical: ", speed_mean_num)
        # calculating mean absolute velocity/speed analytically
        self.speed_mean_anal = np.sqrt((8*self.k_b*self.T)/(np.pi*self.mass_hydrogen_molecule))  
        print("Average speed analytically: ", self.speed_mean_anal)

    def pressure(self, dt=1e-12, steps=1000):
        # r = self.position
        # for timestep in range(steps):
        #     r = r + self.velocity*dt
        #     v_out = np.where(r[:,2] > 1e-6, (self.velocity[:,2]), 0)

        # self.particle_wall_count = np.count_nonzero(v_out)  # after 1e-9 s
        
        new_position= self.position + self.velocity*dt*steps
        v_out_ = np.where(new_position[:,2] > 1e-6, (self.velocity[:,2]), 0)    #v_out_ : velocity of particles passed the border of the box
        self.particle_wall_count_ = np.count_nonzero(v_out_)  # after 1e-9 s
        self.momentum = np.sum(v_out_) * self.mass_hydrogen_molecule * 2
        force = self.momentum/(dt*steps)
        Pressure = force/(self.box_length**2)
        print("Numerical pressure: ", Pressure)
        print("number of particels colliding with the wall: ", self.particle_wall_count_)

        # r = self.position # refill cumbustion box

        # self.p = np.sum(v_out) * self.mass_hydrogen_molecule * 2
        # F = self.p/(dt*steps)
        # pressure_num = F/(self.box_length**2)
        pressure_anal = self.N * self.k_b * self.T/(1e-6**3)
        # print("number of particels colliding with the wall: ", self.particle_wall_count)
        # print("Numeric pressure: ", pressure_num)
        print("Analytical pressure: ", pressure_anal)

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
        k = self.k_b
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
    test = RocketEngine()
    print("Escape velocity (m/s): ", test.v_esc())
    test.initial_values()
    # test.plot_initial_values()
    test.kinetic_energy_mean()
    test.speed_mean()
    test.pressure()
    test.speed_gain()
    test.number_boxes()
    test.fuel_estimate()
    test.rocket_launch()


