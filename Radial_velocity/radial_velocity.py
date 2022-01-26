# kode for exercise C5
# egen kode
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import my_solar_system as my        # importing parameters from my solar system (se file: my_solar_system.py)

class PlanetOrbits():
    def __init__(self):
        self.G = 6.67408e-11 # Gravitational constant
        self.mass_star = my.system.star_mass
        self.planet_numbers = my.system.number_of_planets
        self.masses = my.system.masses[0:3]
        self.earth_mass = 5.9722e24
        self.planets_init_positions = my.system.initial_positions[:,0:3]
        self.planets_init_velocities = my.system.initial_velocities[:,0:3]
        self.masses_all = np.zeros(4)
        self.masses_all[1:4] = self.masses
        self.masses_all[0] = self.mass_star
        print("Plannets initial position in AU: {}".format(self.planets_init_positions))
        print("Masses in the system [in sum masses]: {}".format(self.masses_all))

    def convert(self):
        # Convert to SI units
        AU = my.const.AU
        year_in_s = my.const.yr
        solar_mass = my.const.m_sun
        self.planets_init_positions_SI = self.planets_init_positions * AU
        self.planets_init_velocities_SI = self.planets_init_velocities * AU / year_in_s
        self.masses_planets_SI = self.masses * solar_mass 
        self.mass_star_SI = my.system.star_mass * solar_mass
        self.masses_all_SI = self.masses_all * solar_mass
        print(self.masses_all_SI/ self.earth_mass)

    def RHS(self, r_v):
        # Right hand side od the differential equation
        G = my.const.G
        star_m = self.masses_all_SI[0]
        planet_m = self.masses_all_SI[1:4]
       
        v_x = np.zeros(4)
        v_y = np.zeros(4)
        r_x = np.zeros(4)
        r_y = np.zeros(4)

        v_x[1:4] =  - G*star_m * (r_v[1:4,0] - r_v[0,0]) / np.sqrt((r_v[1:4,0] - r_v[0,0])**2 + (r_v[1:4,1] - r_v[0,1])**2)**3
        v_y[1:4] =  - G*star_m * (r_v[1:4,1] - r_v[0,1]) / np.sqrt((r_v[1:4,0] - r_v[0,0])**2 + (r_v[1:4,1] - r_v[0,1])**2)**3
        
        v_x[0] = np.sum(G*planet_m * (r_v[1:4,0] - r_v[0,0]) / np.sqrt((r_v[1:4,0] - r_v[0,0])**2 + (r_v[1:4,1] - r_v[0,1])**2)**3)
        v_y[0] = np.sum(G*planet_m * (r_v[1:4,1] - r_v[0,1]) / np.sqrt((r_v[1:4,0] - r_v[0,0])**2 + (r_v[1:4,1] - r_v[0,1])**2)**3)

        r_x = r_v[:,2]
        r_y = r_v[:,3]
       
        drdt = np.array([r_x, r_y, v_x, v_y])
        return np.transpose(drdt)

    def euler_cromer(self, r_v, dt):
        k1 = self.RHS(r_v)
        k2 = self.RHS(r_v+k1*dt)
        drdt = k1
        drdt[:,0:2] = k2[:,0:2]
        return drdt

    def generate_orbit(self, time=60, timestep=10, method="euler_cromer"):
        method = getattr(self, method)
        dt = timestep*my.const.day
        self.time = time*my.const.yr
        N = int(self.time/dt)
        self.N = N
        print("Numner of steps: {:g}".format(N))
        self.time_steps = np.linspace(0, time, N)
        
        # create array: 4x4xtime steps
        r_v_t = np.zeros((4, 4, N))
        r_v_t[1:4,0:2,0] = np.transpose(self.planets_init_positions_SI)
        r_v_t[1:4,2:4,0] = np.transpose(self.planets_init_velocities_SI)
        
        for t in range(N-1):
            drdt = method(r_v_t[:,:,t], dt) 
            r_v_t[:,:,t+1] = r_v_t[:,:,t] + drdt*dt 
        
        self.r_v_t = r_v_t # make object with all data of the numeric simulations of the orbits
    
    def plot_orbits(self):
        r_v_t = self.r_v_t
        for i in range(0,4):
            plt.plot(r_v_t[i,0,:], r_v_t[i,1,:], "-", label="Planet {}".format(i))
        # plt.legend()
        plt.xlabel("x-position [AU]")
        plt.ylabel("y-position [AU]")
        plt.savefig('Orbits.png')
        plt.axis("equal")
        plt.show()

    def radial_velocity(self):
        x_velocity = self.r_v_t[0,2,:]
        peculiar_velocity = np.mean(x_velocity)
        x_velocity_max = np.max(x_velocity)
        print("Max radial velcity: {:2f} m/s".format(x_velocity_max))
        print("Peculiar velcity: {:2f} m/s".format(peculiar_velocity))
        self.x_velocity_max = x_velocity_max

        N = self.N
        time = self.time/my.const.yr
        time_steps = np.linspace(0, time, N)
        plt.subplot(211)
     
        plt.plot(time_steps, self.r_v_t[0,2,:])
        plt.xlabel("time [year]")
        plt.ylabel("radial velocity [m/s]")
        plt.show()

    def noise_radial_vel(self):
        N = self.N
        x_velocity = self.r_v_t[0,2,:]
        star_vel_noise1 = np.random.normal(loc=x_velocity, scale=self.x_velocity_max/5, size=N)

        time_steps = np.linspace(0, self.time, N)
        plt.subplot(212)
        plt.plot(time_steps/my.const.yr, star_vel_noise1)
        plt.plot(time_steps/my.const.yr, self.r_v_t[0,2,:])
        plt.xlabel("time [year]")
        plt.ylabel("radial velocity [m/s]")
        plt.show()
        

    def eclipse(self):
        # Calculate the reduction flux by eclipses
        R = my.system.star_radius*1000 # Convert to m
        print("Radus of star: {:.5g} m".format(R))
        r = my.system.radii[0:3]*1000     # radii of planets is concerted to m
        print("Radii of planets [m]: {}".format(r))

        r_v_t = self.r_v_t # array with all data from numerical simulation of orbit
        
        N = np.size(r_v_t, 2)

        d = np.abs(r_v_t[0,1,:] - r_v_t[1:4,1,:]) # array of all distances between the star and the planets
        N = self.N
        flux = np.ones((3,N))
        R_minus_r = np.transpose(np.full((N,3), np.abs(R-r)))
        R_plus_r = np.transpose(np.full((N,3), np.abs(R+r)))
        
        condition_full_eclipse = (d <= R_minus_r)
        condition_half_eclipse = (d < R_plus_r ) & (d > R_minus_r)
        
        def full_eclipse_area():
            # calculates reduction in flux during full eclipse
            return np.transpose(np.full((N,3), 1 - (r**2)/(R**2)))
        
        def half_eclipse_area():
            # calculates reduction in flux during interemediate eclipse
            d_t = np.transpose(d)
            return np.transpose(1 - 0.5*(r*r + r*(R-d_t))/R**2)

        flux = np.where(condition_full_eclipse, full_eclipse_area(), flux)
        flux = np.where(condition_half_eclipse, half_eclipse_area(), flux)
        
        time = self.time/my.const.yr
        self.time_steps = np.linspace(0, time, N)
        
        plt.subplot(211)
        plt.plot(self.time_steps, flux[0], label="planet 0")
        plt.plot(self.time_steps, flux[1], label="planet 1")
        plt.plot(self.time_steps, flux[2], label="planet 2")
        plt.legend()
        plt.ylabel("Relative flux")
        plt.subplot(212)
        plt.plot(self.time_steps, flux[0], label="planet 0")
        plt.plot(self.time_steps, flux[1], label="planet 1")
        plt.plot(self.time_steps, flux[2], label="planet 2")
        plt.ylabel("Relative flux")
        plt.xlabel("Time [years]")
        plt.ylim(0.9998, 1.000005)
        plt.tight_layout()
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.5f}')) # 2 decimal places
        plt.show()

        # Addition of noise
        noise = np.random.normal(scale=0.001,size=N)
        plt.plot(self.time_steps,flux[2] + noise)
        plt.xlabel("Time (year)")
        plt.ylabel("Relative flux")
        plt.show()


if __name__ == "__main__":
    obligC5 = PlanetOrbits()
    obligC5.convert()
    obligC5.generate_orbit(time=120, timestep=0.1, method="euler_cromer")
    # obligC5.plot_orbits()
    # obligC5.radial_velocity()
    # obligC5.noise_radial_vel()
    # obligC5.eclipse()



