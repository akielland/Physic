# kode for exercise C4
# egen kode
import numpy as np
import matplotlib.pyplot as plt
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
        print(self.planets_init_positions)
        print(self.masses)

    def convert(self):
        AU = my.const.AU
        year_in_s = my.const.yr
        solar_mass = my.const.m_sun
        self.planets_init_positions_SI = self.planets_init_positions * AU
        self.planets_init_velocities_SI = self.planets_init_velocities * AU / year_in_s
        self.masses_planets_SI = self.masses * solar_mass 
        self.mass_star_SI = my.system.star_mass * solar_mass
        # print(self.masses_planets_SI/ self.earth_mass)

    def RHS_planets(self, r_v):
        G, star_m = my.const.G, self.mass_star_SI
        v_x =  - G*star_m * (r_v[:,0] )/ np.sqrt(r_v[:,0]**2 + r_v[:,1]**2)**3
        v_y =  - G*star_m * r_v[:,1] / np.sqrt(r_v[:,0]**2 + r_v[:,1]**2)**3
        r_x = r_v[:,2]
        r_y = r_v[:,3]
       
        drdt = np.array([r_x, r_y, v_x, v_y])
        return np.transpose(drdt)

    def RHS_star(self, r_v):
        G, planet_m = my.const.G, my.system.masses[0:3]
        v_x =  - np.sum(G*planet_m * r_v[:,0] / np.sqrt(r_v[:,0]**2 + r_v[:,1]**2)**3)
        v_y =  - np.sum(G*planet_m * r_v[:,1] / np.sqrt(r_v[:,0]**2 + r_v[:,1]**2)**3)
        r_x = r_v[:,2]
        r_y = r_v[:,3]
       
        drdt = np.array([r_x, r_y, v_x, v_y])
        return np.transpose(drdt)


    def euler_cromer(self, r_v, dt):
        k1 = self.RHS_planets(r_v)
        k2 = self.RHS_planets(r_v+k1*dt)
        drdt = k1
        drdt[:,0:2] = k2[:,0:2]
        return drdt

    def euler_cromer_S(self, r_v, dt):
        k1 = self.RHS_star(r_v)
        k2 = self.RHS_star(r_v+k1*dt)
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

    def generate_orbit(self, time=my.const.yr*60, N=1000, method="euler_cromer"):
        method = getattr(self, method)
        dt = time/N
        print("Integration step length {:g} hours".format(dt/3600))
        self.time_steps = np.linspace(0, time, N)
        
        # create array: 3x4xtime steps
        r_v_t = np.zeros((3, 4, N))
        r_v_t[:,0:2,0] = np.transpose(self.planets_init_positions_SI)
        r_v_t[:,2:4,0] = np.transpose(self.planets_init_velocities_SI)
        for t in range(N-1):
            drdt = method(r_v_t[:,:,t], dt) 
            r_v_t[:,:,t+1] = r_v_t[:,:,t] + drdt*dt 

        R_v_t = np.zeros((1, 4, N))
        for t in range(N-1):
            drdt = self.euler_cromer_S(R_v_t[:,:,t], dt) 
            R_v_t[:,:,t+1] = R_v_t[:,:,t] + drdt*dt 
        
        self.r_v_t = r_v_t#/my.const.AU
        self.R_v_t = r_v_t#/my.const.AU

    # def intersection_area(d, R, r):
    #     """Return the area of intersection of two circles.
    #     The circles have radii R and r, and their centres are separated by d.
    #     """

    #     r_p = self.r_v_t[]
    #     d = np.linalg.norm([i][1][j] - xs[i][1])  #  < Rp[j] + Rs:

    #     if d <= abs(R-r):
    #         # One circle is entirely enclosed in the other.
    #         return np.pi * min(R, r)**2
    #     if d >= r + R:
    #         # The circles don't overlap at all.
    #         return 0

    #     r2, R2, d2 = r**2, R**2, d**2
    #     alpha = np.arccos((d2 + r2 - R2) / (2*d*r))
    #     beta = np.arccos((d2 + R2 - r2) / (2*d*R))
    #     return ( r2 * alpha + R2 * beta -
    #             0.5 * (r2 * np.sin(2*alpha) + R2 * np.sin(2*beta))
    #         )

if __name__ == "__main__":
    oblig1 = PlanetOrbits()
    oblig1.convert()


    # oblig1.generate_orbit(N=1000, method="forward_euler")
    oblig1.generate_orbit(N=1000, method="euler_cromer")
    # oblig1.generate_orbit(N=1000, method="RK4")






        # for i in range(8):
        #     plt.plot(r_v_t[i,0,:], r_v_t[i,1,:], label="Planet {}".format(i))
        # plt.legend()
        # plt.xlabel("x-position [AU]")
        # plt.ylabel("y-position [AU]")
        # plt.savefig('Orbits.png')
        # plt.axis("equal")
        # plt.show()
        # self.r_v_t = r_v_t
