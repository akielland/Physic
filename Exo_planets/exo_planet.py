# egen kode
import numpy as np
import matplotlib.pyplot as plt
import my_solar_system as my        # importing parameters from my solar system (se file: my_solar_system.py)
import statsmodels.api as sm
print(my.seed)

class C4():
    def __init__(self, filename):
        self.name = filename[:-9]
        print(self.name)
        self.c = my.const.c
        self.H_alpha = 656.28e-9
        self.read_data(filename)

    def read_data(self, filename):
        time, wavelength, relative_flux = [], [], []
        with open(filename, "r") as f:
            for line in f:
                t_, w_, r_ = [float(x) for x in line.split()]
                time.append(t_), wavelength.append(w_), relative_flux.append(r_)
        
        self.time, self.wavelength, self.relative_flux = np.array(time), np.array(wavelength), np.array(relative_flux)
        
    def plot_raw_data(self):  
        plt.subplot(211)
        plt.plot(self.time, self.wavelength,label="wavelength"), 
        plt.subplot(212)
        plt.plot(self.time, self.relative_flux,label="relative flux"),  
        plt.xlabel("time [s]"), plt.ylabel("relative flux []"), plt.xlim(-0.1,2)
        plt.legend(), plt.show()

    def planet_mass(self, star_mass, star_velocity, period):
        mass = (star_mass*my.const.m_sun)**(2/3) * star_velocity * period**(1/3) / (5.972e24 * (2*np.pi*my.const.G)**(1/3))
        return mass

    def star_velocity(self):
        # Use Doppler formula to find the velocity of the star relative to earth
        wavelength = self.wavelength * 1e-9
        star_velocity = self.c*(wavelength - self.H_alpha)/self.H_alpha
        return star_velocity

    def peculiar_velocity(self):
        # Calculate peculiar velocity of the star 
        peculiar_velocity = np.mean(self.star_velocity())
        print("Peculiar velocity of {} is: {:.1f} km/s".format(self.name, peculiar_velocity/1000))
        return peculiar_velocity

    def velocity_CM(self):
        # Calculate the redial velocity of the star within its solar system
        return self.star_velocity() - self.peculiar_velocity()

    def plot_velocity_CM_and_flux(self, plotname):
        # plot of star velocity and relative flux for publication
        plt.figure(figsize=(3.5,4))
        plt.subplot(211)
        plt.ylabel("Radial velocity [m/s]")
        
        plt.plot(self.time, self.velocity_CM())
        plt.ylim()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.subplot(212)
        plt.plot(self.time, self.relative_flux)
        
        plt.xlabel("Time [days]")
        plt.ylabel("Relative flux")
        plt.savefig(plotname)
        plt.tight_layout()
        plt.show()

    def initial_values(self):
        # Find initial values to be used in the curve fit method
        vel = self.velocity_CM()
        # Find points where the data cross zero in positive direction
        zero_cross = np.where(np.logical_and(np.diff(vel) > 5, np.diff(np.sign(vel)) != 0))[0]
        n = len(zero_cross)
        # loop over the zero crossing point and select where there is a large raise. Pic the indices there
        zero_raise =[]
        for i in range(10, n-1):
            a = np.mean(zero_cross[i-3:i-1]) + 200
            if zero_cross[i] > a:
                zero_raise.append(i-10)
        raise_ = np.asarray(zero_raise)
        i = np.where(np.diff(raise_) > 15)[0]
        indices = zero_cross[raise_[i]]
       
        # Find P range
        start_i, end_i = indices[0], indices[1]
        P = 2*(self.time[end_i] - self.time[start_i])
        P_ = (np.where(np.logical_and(P-1<self.time, P+1>self.time)))
        P_ = int(P_[0]/2)
        # Ensure positive part of curve
        if np.mean(vel[start_i:end_i]) < 0:
            start_i = start_i + P_
            end_i = end_i + P_
        
        # set parameters to test
        max_position = self.time[start_i + np.argmax(np.abs(vel[start_i:end_i]))]
        self.velocity_max = np.max(np.abs(vel[start_i:end_i])) - 5
        self.velocity_min = self.velocity_max - 10
        self.t0_min, self.t0_max = max_position - 0.2*P, max_position + 0.2*P
        self.P_min, self.P_max = 0.8*P, 1.2*P
        self.start_i, self.end_i = start_i, end_i

    def curve_fit(self, N=20):
        # use least square method to fit a sine function to the star velocity data
        t0 = np.linspace(self.t0_min, self.t0_max, N)
        P = np.linspace(self.P_min, self.P_max, N)
        velocity = np.linspace(self.velocity_min, self.velocity_max, N)

        # Create a 3D matrix with test parameters
        t0_m, P_m, v_m = np.meshgrid(t0, P, velocity)
        v_CM = self.velocity_CM()
        time = self.time[self.start_i: self.end_i]
        i = self.start_i
        s_sum = np.zeros((N,N,N))
        # loop over the timepoints with the test matrix
        for t in time:
            v_model_r = v_m*np.cos(2*np.pi/P_m*(t0_m - t))
            distance = (v_CM[i] - v_model_r)**2
            s_sum = s_sum + distance
            i = i + 1

        # pic index of lowest value
        ind = np.unravel_index(np.argmin(s_sum, axis=None), s_sum.shape)

        t=self.time
        model = v_m[ind]*np.cos(2*np.pi/P_m[ind]*(t0_m[ind] - t))
        self.model = model

        plt.figure(figsize=(5,4))
        plt.plot(t, self.velocity_CM()),
        plt.plot(t, model)
        
        plt.xlabel("Time [days]")
        plt.ylabel("Radial velocity [m/s]")
        plt.savefig("model.png")
        plt.show()

    def qq_std(self):    
        #create Q-Q plot of the noise in the star velocity data
        data = self.star_velocity() - self.model
        print(np.std(data))
        sm.qqplot(data, line='s')
        plt.show()


if __name__ == "__main__":
    star0 = C4("star0_1.63.txt")
    star1 = C4("star1_1.15.txt")
    star2 = C4("star2_0.84.txt")
    star3 = C4("star3_3.52.txt")
    star4 = C4("star4_0.97.txt")

    # print(star0.planet_mass(0.97, 33, 5000))
    # star0.plot_raw_data()
    # star0.star_velocity()
    star4.initial_values()
    star4.curve_fit(N=50)
    # star4.plot_velocity_CM_and_flux("star4_vel-flux.png")
    # star4.qq_std()
