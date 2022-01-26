# kode for exercise 1B.7
# egen kode
import numpy as np
import matplotlib.pyplot as plt
import my_solar_system as my        # importing parameters from my solar system (se file: my_solar_system.py)

class PlanetOrbits():
    def __init__(self):
        self.G = 6.67408e-11    # Gravitational constant
        self.mass_star = my.system.star_mass
        self.planet_numbers = my.system.number_of_planets
        self.masses = my.system.masses
        self.earth_mass = 5.9722e24
        self.planets_init_positions = my.system.initial_positions
        self.planets_init_velocities = my.system.initial_velocities
        print("Plannets initial position in AU: {}".format(self.planets_init_positions))
        print(self.mass_star)

    def convert(self):
        # convert to SI units
        AU = my.const.AU
        year_in_s = my.const.yr
        solar_mass = my.const.m_sun
        self.planets_init_positions_SI = self.planets_init_positions * AU
        self.planets_init_velocities_SI = self.planets_init_velocities * AU / year_in_s
        self.masses_planets_SI = self.masses * solar_mass 
        self.mass_star_SI = my.system.star_mass * solar_mass
        print(self.masses_planets_SI/self.earth_mass)

    def make_file(self, parameter_vector, filename):
        # Make file of paramters
        out = parameter_vector
        f= open(filename,"w+")
        for i in range(len(out[0])):
            x, y = str(out[0][i]), str(out[1][i])
            f.write("{}\t{}\n".format(x, y))
        f.close()

    def RHS(self, r_v):
        # Right hand side of the differential equation
        G, star_m = my.const.G, self.mass_star_SI
        v_x =  - G*star_m * r_v[:,0] / np.sqrt(r_v[:,0]**2 + r_v[:,1]**2)**3
        v_y =  - G*star_m * r_v[:,1] / np.sqrt(r_v[:,0]**2 + r_v[:,1]**2)**3
        r_x = r_v[:,2]
        r_y = r_v[:,3]
       
        drdt = np.array([r_x, r_y, v_x, v_y])
        return np.transpose(drdt)

    def forward_euler(self, r_v, dt):
        drdt = self.RHS(r_v)
        return drdt

    def euler_cromer(self, r_v, dt):
        k1 = self.RHS(r_v)
        k2 = self.RHS(r_v+k1*dt)
        drdt = k1
        drdt[:,0:2] = k2[:,0:2]
        return drdt

    def RK4(self, r_v, dt):
        k1 = self.RHS(r_v)
        k2 = self.RHS(r_v+k1*dt/2)
        k3 = self.RHS(r_v+k2*dt/2)
        k4 = self.RHS(r_v+k3*dt)
        drdt = (k1 + 2*k2 + 2*k3 + k4)/6
        return drdt

    def generate_orbits(self, time=my.const.yr*60, N=1000, method="euler_cromer"):
        method = getattr(self, method)
        dt = time/N
        print("Integration step length {:g} hours".format(dt/3600))
        self.time_steps = np.linspace(0, time, N)
        
        # Array to hld all numerical data : 8x4xtime steps
        r_v_t = np.zeros((8, 4, N))
        r_v_t[:,0:2,0] = np.transpose(self.planets_init_positions_SI)
        r_v_t[:,2:4,0] = np.transpose(self.planets_init_velocities_SI)
      
        # integration
        for t in range(N-1):
            drdt = method(r_v_t[:,:,t], dt) 
            r_v_t[:,:,t+1] = r_v_t[:,:,t] + drdt*dt 
        
        r_v_t = r_v_t/my.const.AU
  
        for i in range(8):
            plt.plot(r_v_t[i,0,:], r_v_t[i,1,:], label="Planet {}".format(i))
        plt.legend()
        plt.xlabel("x-position [AU]")
        plt.ylabel("y-position [AU]")
        plt.savefig('Orbits.png')
        plt.axis("equal")
        plt.show()
        self.r_v_t = r_v_t
       
    def energy(self):
        # calculate mechanical energy at each time step
        r_v_t = self.r_v_t
        G, star_m = my.const.G, self.mass_star_SI
        potential =  G*star_m * self.masses  / np.transpose(np.sqrt(r_v_t[:,0,:]**2 + r_v_t[:,1,:]**2))
        kinetic = 0.5  * self.masses * np.transpose(r_v_t[:,2,:]**2 + r_v_t[:,3,:]**2)
        energy = potential + kinetic
        sd = np.zeros(8)
        for i in range(8):
            sd[i] = np.std(energy[:,i])

        plt.subplot(131)
        for i in range(8):
            plt.plot(self.time_steps/my.const.yr, energy[:, i], label="Planet {:g}".format(i))
        plt.yscale("log")
        plt.legend()
        plt.xlabel("time [year]")
        plt.ylabel("energy [J]")
        plt.show()
  
    def video(self):
        times, planet_positions = self.time_steps, np.transpose(self.r_v_t[:,0:2,:],(1,0,2))
        my.system.generate_orbit_video(times, planet_positions, filename='orbit_video.xml')

if __name__ == "__main__":
    oblig1 = PlanetOrbits()
    oblig1.convert()

    # oblig1.make_file(oblig1.planets_init_velocities_SI, "solarsystem_init_vel1.txt")
    # oblig1.make_file(oblig1.masses_planets_SI, "solarsystem_masses.txt")

    # oblig1.generate_orbits(N=1000, method="forward_euler")
    # oblig1.generate_orbits(N=1000, method="euler_cromer")
    # oblig1.generate_orbits(N=1000, method="RK4")

    # oblig1.energy()

    # oblig1.video()


