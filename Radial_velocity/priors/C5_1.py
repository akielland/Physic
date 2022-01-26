# kode for exercise C4
# egen kode
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import my_solar_system as my        # importing parameters from my solar system (se file: my_solar_system.py)

class PlanetOrbits():
    def __init__(self):
         # Gravitational constant
        self.G = 6.67408e-11        
        self.mass_star = my.system.star_mass
        self.planet_numbers = my.system.number_of_planets
        self.masses = my.system.masses[0:3]
        self.earth_mass = 5.9722e24
        self.planets_init_positions = my.system.initial_positions[:,0:3]
        self.planets_init_velocities = my.system.initial_velocities[:,0:3]
        # print("Plannets initial position in AU: {}".format(self.planets_init_positions))
        AU = my.const.AU
        
        self.masses_all = np.zeros(4)
        self.masses_all[1:4] = self.masses
        self.masses_all[0] = self.mass_star

        # print(self.planets_init_positions * my.const.AU)
        print(self.masses_all)

    def convert(self):
        AU = my.const.AU
        year_in_s = my.const.yr
        solar_mass = my.const.m_sun
        self.planets_init_positions_SI = self.planets_init_positions * AU
        self.planets_init_velocities_SI = self.planets_init_velocities * AU / year_in_s
        self.masses_planets_SI = self.masses * solar_mass 
        self.mass_star_SI = my.system.star_mass * solar_mass
        self.masses_all_SI = self.masses_all * solar_mass
        # print(self.masses_planets_SI/ self.earth_mass)

    def RHS_planets(self, r_v):
        G, star_m = my.const.G, self.mass_star_SI
        v_x =  - G*star_m * r_v[:,0]/ np.sqrt(r_v[:,0]**2 + r_v[:,1]**2)**3
        v_y =  - G*star_m * r_v[:,1] / np.sqrt(r_v[:,0]**2 + r_v[:,1]**2)**3
        r_x = r_v[:,2]
        r_y = r_v[:,3]
       
        drdt = np.array([r_x, r_y, v_x, v_y])
        return np.transpose(drdt)

    def RHS(self, r_v):
        G = my.const.G
        star_m = self.masses_all_SI[0]
        planet_m = self.masses_all_SI[1:4]
       
        v_x = np.zeros(4)
        v_y = np.zeros(4)
        r_x = np.zeros(4)
        r_y = np.zeros(4)
        # print("before")
        # print(r_v[1:4,0])
        # print("after")

        v_x[1:4] =  - G*star_m * (r_v[1:4,0] - r_v[0,0]) / np.sqrt((r_v[1:4,0] - r_v[0,0])**2 + (r_v[1:4,1] - r_v[0,1])**2)**3
        v_y[1:4] =  - G*star_m * (r_v[1:4,1] - r_v[0,1]) / np.sqrt((r_v[1:4,0] - r_v[0,0])**2 + (r_v[1:4,1] - r_v[0,1])**2)**3
        
        # v_x[1:4] =  - G*star_m * r_v[1:4,0] / np.sqrt(r_v[1:4,0]**2 + r_v[1:4,1]**2)**3
        # v_y[1:4] =  - G*star_m * r_v[1:4,1] / np.sqrt(r_v[1:4,0]**2 + r_v[1:4,1]**2)**3
        v_x[0] = np.sum(G*planet_m * (r_v[1:4,0] - r_v[0,0]) / np.sqrt((r_v[1:4,0] - r_v[0,0])**2 + (r_v[1:4,1] - r_v[0,1])**2)**3)
        v_y[0] = np.sum(G*planet_m * (r_v[1:4,1] - r_v[0,1]) / np.sqrt((r_v[1:4,0] - r_v[0,0])**2 + (r_v[1:4,1] - r_v[0,1])**2)**3)
        # print(v_x[0])
        # print(v_y[0])
        r_x = r_v[:,2]
        r_y = r_v[:,3]
       
        drdt = np.array([r_x, r_y, v_x, v_y])
        return np.transpose(drdt)

    def euler_cromer(self, r_v, dt):
        k1 = self.RHS(r_v)
        k2 = self.RHS(r_v+k1*dt)
        drdt = k1
        drdt[:,0:2] = k2[:,0:2]
        # print("DRDT: ",r_v)
        return drdt

    def forward_euler(self, r_v, dt):
        drdt = self.RHS(r_v)
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
        
        self.r_v_t = r_v_t#/my.const.AU
    
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
        # star_vel = np.transpose(self.r_v_t[0,2:4,:])
        # radial_vel = np.dot(star_vel, np.array([1,1]/np.sqrt(2)))
        x_velocity = self.r_v_t[0,2,:]
        peculiar_vel = np.mean(x_velocity)
        x_velocity_max = np.max(x_velocity) - peculiar_vel
        print("Max radial velcity: {:2f} m/s".format(x_velocity_max))
        self.x_velocity_max = x_velocity_max

        N = self.N
        time = self.time/my.const.yr
        time_steps = np.linspace(0, time, N)
        plt.subplot(211)
     
        plt.plot(time_steps, self.r_v_t[0,2,:])
        plt.xlabel("time [year]")
        plt.ylabel("radial velocity [m/s]")
        plt.show()

    def noise(self):
        N = self.N
        x_velocity = self.r_v_t[0,2,:]

        star_vel_noise1 = np.random.normal(loc=x_velocity, scale=self.x_velocity_max/5, size=N)
        # star_vel_noise2 = np.random.normal(loc=star_vel, scale=0.2, size=N)

        time_steps = np.linspace(0, self.time, N)
        plt.subplot(212)
        plt.plot(time_steps, star_vel_noise1)
        # plt.plot(time_steps, star_vel_noise2)
        plt.plot(time_steps, self.r_v_t[0,2,:])
        plt.show()
        

    def intersection_area(self):
        """Return the area of intersection of two circles.
        The circles have radii R and r, and their centres are separated by d.
        """
        R = my.system.star_radius*1000 #gjoer om til m
        r = my.system.radii[0:3]*1000     #radiusen til planetene
        r_v_t = self.r_v_t
        
        d = np.abs(r_v_t[0,1,:] - r_v_t[1:4,1,:])
        N = self.N
        flux = np.ones((3,N))
        R_minus_r = np.transpose(np.full((N,3), np.abs(R-r)))
        R_plus_r = np.transpose(np.full((N,3), np.abs(R+r)))
        
        condition_full_eclipse = (d <= R_minus_r)
        condition_half_eclipse = (d < R_plus_r ) & (d > R_minus_r)
        
        def full_eclipse_area():
            return np.transpose(np.full((N,3), 1 - (r**2)/(R**2)))
        
        def half_eclipse_area():
            d_t = np.transpose(d)
            return np.transpose(1 - 0.5*(r*r + r*(R-d_t))/R**2)
        
        # print("distance")
        # print(full_eclipse)
        # print("cond")
        # print(condition1)

        flux = np.where(condition_full_eclipse, full_eclipse_area(), flux)
        flux = np.where(condition_half_eclipse, half_eclipse_area(), flux)
        
        plt.subplot(211)
        plt.plot(self.time_steps, flux[0], label="planet 0")
        plt.plot(self.time_steps, flux[1], label="planet 1")
        plt.plot(self.time_steps, flux[2], label="planet 2")
        plt.legend()
        plt.subplot(212)
        plt.plot(self.time_steps, flux[0], label="planet 0")
        plt.plot(self.time_steps, flux[1], label="planet 1")
        plt.plot(self.time_steps, flux[2], label="planet 2")
        plt.ylim(0.9999, 1.000005)
        
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.5f}')) # 2 decimal places
        plt.show()

        noise = np.random.normal(scale=0.2,size=N)
        plt.plot(self.time_steps,flux[2]+noise)
        plt.xlabel("time (year)")
        plt.ylabel("relative flux")
        plt.title("Relative flux with noise")
        # plt.show()

        # print(flux)
        # print(np.sum(flux))
        # if d <= abs(R-r):
        #     # One circle is entirely enclosed in the other.
        #     return np.pi * min(R, r)**2
        # if d >= r + R:
        #     # The circles don't overlap at all.
        #     return 0

        # r2, R2, d2 = r**2, R**2, d**2
        # alpha = np.arccos((d2 + r2 - R2) / (2*d*r))
        # beta = np.arccos((d2 + R2 - r2) / (2*d*R))
        # return (r2 * alpha + R2 * beta - 0.5 * (r2 * np.sin(2*alpha) + R2 * np.sin(2*beta)))

if __name__ == "__main__":
    obligC5 = PlanetOrbits()
    obligC5.convert()
    obligC5.generate_orbit(time=60, timestep=0.1, method="euler_cromer")
    # obligC5.plot_orbits()
    # obligC5.radial_velocity()
    # obligC5.noise()
    obligC5.intersection_area()


        # for i in range(8):
        #     plt.plot(r_v_t[i,0,:], r_v_t[i,1,:], label="Planet {}".format(i))
        # plt.legend()
        # plt.xlabel("x-position [AU]")
        # plt.ylabel("y-position [AU]")
        # plt.savefig('Orbits.png')
        # plt.axis("equal")
        # plt.show()
        # self.r_v_t = r_v_t
