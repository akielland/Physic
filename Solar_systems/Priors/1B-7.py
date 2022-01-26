# kode for exercise 1B.7
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
        self.masses = my.system.masses
        self.planets_init_positions = my.system.initial_positions
        self.planets_init_velocities = my.system.initial_velocities

    def convert_to_SI(self):
        AU = my.const.AU
        year_in_s = my.const.yr
        solar_mass = my.const.m_sun
        self.planets_init_positions_SI = self.planets_init_positions * AU
        self.planets_init_velocities_SI = self.planets_init_velocities * AU / year_in_s
        self.masses_planets_SI = my.system.masses * solar_mass
        self.mass_star_SI = my.system.star_mass * solar_mass

    def make_file(self, parameter_vector, filename):
        out = parameter_vector
        f= open(filename,"w+")
        for i in range(len(out[0])):
            x, y = str(out[0][i]), str(out[1][i])
            f.write("{}\t{}\n".format(x, y))
        f.close()

    def generate_orbit(self, time=3600*24*365*60, N=1000):
        # N = int(time/dt)
        dt = time/N
        print(dt/3600)
        # time_steps = np.linspace(0, time, N)
        # create array: 8x2xtime steps
        pos = np.zeros((8,2, N))
        vel = np.zeros((8,2, N))
        pos[:,:,0] = np.transpose(self.planets_init_positions_SI)
        vel[:,:,0] = np.transpose(self.planets_init_velocities_SI)
        print(np.shape(np.transpose(self.planets_init_positions)))
        # print(pos)
        # print(vel)
        
        # print(position[1,:,0])
    #     print(self.planets_init_positions)
    
    # def forward_euler(self):
        G, star_m = self.G, self.mass_star_SI
        print(np.shape(vel))
        for t in range(N-1):
            vel[:,0,t+1] = vel[:,0,t] - dt * (G*star_m * pos[:,0,t] / np.sqrt(pos[:,0,t]**2 + pos[:,1,t]**2)**3)
            vel[:,1,t+1] = vel[:,1,t] - dt * (G*star_m * pos[:,1,t] / np.sqrt(pos[:,0,t]**2 + pos[:,1,t]**2)**3)
            pos[:,0,t+1] = pos[:,0,t] + dt * vel[:,0,t+1]
            pos[:,1,t+1] = pos[:,1,t] + dt * vel[:,1,t+1]
        
        for i in range(8):
            plt.plot(pos[i,0,:], pos[i,1,:], label="Planet {}".format(i))
        plt.legend()
        plt.show()
        
     

        G, star_m = self.G, self.mass_star_SI
        # G = 6.67*10**(-11)*(1.989*10**30)*(3600*24*365)**2/(149597870700**3)
        # star_m = my.system.star_mass
       
        # for t in range(N-1):
        #     vel[0,t+1] = vel[0,t] - dt * G*star_m * pos[0,t] / (np.sqrt( pos[0,t]**2 + pos[1,t]**2 )**3)
        #     vel[1,t+1] = vel[1,t] - dt * (G*star_m * pos[1,t] / np.sqrt(pos[0,t]**2 + pos[1,t]**2)**3)
        #     pos[0,t+1] = pos[0,t] + dt * vel[0,t+1]
        #     pos[1,t+1] = pos[1,t] + dt * vel[1,t+1]
        
        # plt.plot(pos[0,:], pos[1,:])
        # plt.show()
        # # print(vel)


    def RHS(self, r_v):
        G, star_m = self.G, self.mass_star_SI
        v_x =  - (G*star_m * r_v[:,0] / np.sqrt(r_v[:,0]**2 + r_v[:,1]**2)**3)
        v_y =  - (G*star_m * r_v[:,1] / np.sqrt(r_v[:,0]**2 + r_v[:,1]**2)**3)
        r_x = r_v[:,2]
        r_y = r_v[:,3]
       
        drdt = np.array([r_x, r_y, v_x, v_y])
        # print(np.shape(drdt))
        return np.transpose(drdt)

    def forward_euler(self, r_v, dt):
        drdt = self.RHS(r_v)
        return drdt

    def euler_cromer(self, r_v, dt):
        k1 = self.RHS(r_v)
        k2 = self.RHS(r_v+k1*dt)
        drdt = k1
        # print(np.shape(drdt))
        drdt[:,0:2] = k2[:,0:2]
        return drdt

    def RK4(self, r_v, dt):
        k1 = self.RHS(r_v)
        k2 = self.RHS(r_v+k1*dt/2)
        k3 = self.RHS(r_v+k2*dt/2)
        k4 = self.RHS(r_v+k3*dt)
        drdt = (k1 + 2*k2 + 2*k3 + k4)/6
        return drdt

    def integration(self, time=3600*24*365*60, N=10, method="euler_cromer"):
        method = getattr(self, method)
        dt = time/N
        r_v = np.zeros((8, 4, N))
        r_v[:,0:2,0] = np.transpose(self.planets_init_positions_SI)
        r_v[:,2:4,0] = np.transpose(self.planets_init_velocities_SI)
      
        for t in range(N-1):
            drdt = method(r_v[:,:,t], dt) 
            r_v[:,:,t+1] = r_v[:,:,t] + drdt*dt 
             

        for i in range(8):
            plt.plot(r_v[i,0,:], r_v[i,1,:], label="Planet {}".format(i))
        plt.legend()
        plt.show()
        
        return r_v
        


    # def video():
    #     times, planet_positions = pass # Your own orbit simulation code
    #     my.system.generate_orbit_video(times, planet_positions, filename='orbit_video.xml')

if __name__ == "__main__":
    oblig1 = PlanetOrbits()
    oblig1.convert_to_SI()
    # oblig1.make_file()
    oblig1.make_file(oblig1.planets_init_velocities_SI, "solarsystem_init_vel1.txt")
    # print(oblig1.masses_planets_SI)
    # oblig1.make_file(oblig1.masses_planets_SI, "solarsystem_masses.txt")
    # oblig1.generate_orbit()
    oblig1.integration()



    # def generate_orbit(self, time=10, dt=1):
    #     N = int(time/dt)
    #     # time_steps = np.linspace(0, time, N)
    #     # create array: 8x2xtime steps
    #     pos = np.zeros((self.planet_numbers, 2, N))
    #     vel = np.zeros((self.planet_numbers, 2, N))
    #     pos[:,:,0] = np.transpose(self.planets_init_positions)
    #     vel[:,:,0] = np.transpose(self.planets_init_velocities)
    #     # print(np.shape(np.transpose(self.planets_init_positions)))
    #     # print(position[0,0,0])
    #     # print(position[1,:,0])
    # #     print(self.planets_init_positions)
    
    # # def forward_euler(self):
    #     G, star_m = self.G, self.mass_star_SI
    #     for t in range(N-1):
    #         vel[:,0,t+1] = vel[:,0,t] - dt * (G*star_m * pos[:,0,t] / np.sqrt(pos[:,0,t]**2 + pos[:,1,t]**2)**3)
    #         vel[:,1,t+1] = vel[:,1,t] - dt * (G*star_m * pos[:,1,t] / np.sqrt(pos[:,0,t]**2 + pos[:,1,t]**2)**3)
    #         pos[:,0,t+1] = pos[:,0,t] + dt * vel[:,0,t+1]
    #         pos[:,1,t+1] = pos[:,1,t] + dt * vel[:,1,t+1]
        
    #     plt.plot(pos[0,0,:], pos[0,1,:], "bo")
    #     plt.show()




        # print("star mass: ", self.mass_star)
        # print("number of planets: ", self.planet_numbers)
        # print("Masses in sun mass: ", self.masses)
        # print("positions: ", self.planets_init_positions)
        # print("velocities: ", self.planets_init_velocities)

