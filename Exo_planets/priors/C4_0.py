# kode for exercise 1B.7
# egen kode
import numpy as np
import matplotlib.pyplot as plt
import my_solar_system as my        # importing parameters from my solar system (se file: my_solar_system.py)
print(my.seed)
from scipy.signal import savgol_filter
import statsmodels.api as sm


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
        # plt.xlabel("time [s]"), plt.ylabel("temp[celcius degree]"), plt.xlim(-0.1,2)
        plt.legend(), plt.show()

    def star_velocity(self):
        wavelength = self.wavelength * 1e-9
        star_velocity = self.c*(wavelength - self.H_alpha)/self.H_alpha
        return star_velocity

    def peculiar_velocity(self):
        peculiar_velocity = np.mean(self.star_velocity())
        print("Peculiar velocity of {} is: {:.1f} km/s".format(self.name, peculiar_velocity/1000))
        return peculiar_velocity

    def velocity_CM(self):
        return self.star_velocity() - self.peculiar_velocity()


    def plot_velocity_CM(self):
        plt.plot(self.time, self.velocity_CM())
        plt.show()

    def periodic_function_v_star_r(self, t, t0, P, v_star_r):
        v_model_r = v_star_r*np.cos(2*np.pi/P*(t-t0))
        return v_model_r

    def LSM(self, t0, P, v_r):
        v_r_model = periodic_function_v_star_r(t, t0, P, v_star_r)
        return np.sum((v_r - v_r_model)**2)

    def matrix(self):
        pass

    def smooth(self):
        w = savgol_filter(self.star_velocity(), 201, 2)
        plt.plot(self.time,w)
        plt.show()

    def initial_values(self):
        x = self.velocity_CM()
        zero_cross = np.where(np.logical_and(np.diff(x) > 5, np.diff(np.sign(x)) != 0))[0]
        n = len(zero_cross)
        zero_raise =[]
        for i in range(10, n-1):
            a = np.mean(zero_cross[i-3:i-1]) + 200
            if zero_cross[i] > a:
                zero_raise.append(i-10)
        r = np.asarray(zero_raise)
        test = np.where(np.diff(r) > 15)[0]
        # print(zero_raise)
        # print(test)

        indices = zero_cross[r[test]]
        # print(zero_cross)
        # print(zero_raise)
        # print(indices)
       
        start_i, end_i = indices[0], indices[1]
        
        # start_t, end_t = self.time[indices[-2]], self.time[indices[-1]]
        P = 2*(self.time[end_i] - self.time[start_i])
        P_ = (np.where(np.logical_and(P-1<self.time, P+1>self.time)))
        P_ = int(P_[0]/2)
        if np.mean(x[start_i:end_i]) < 0:
            start_i = start_i + P_
            end_i = end_i + P_
        
        max_position = self.time[start_i + np.argmax(np.abs(x[start_i:end_i]))]


        self.velocity_max = np.max(np.abs(x[start_i:end_i])) - 5

        self.velocity_min = self.velocity_max - 10
        self.t0_min, self.t0_max = max_position - 0.2*P, max_position + 0.2*P
        self.P_min, self.P_max = 0.8*P, 1.2*P
        self.start_i, self.end_i = start_i, end_i

        # print("velocity max ", self.velocity_max)
        print("max_position: ", max_position)
        print("t interval", self.t0_min, self.t0_max)
        print("P interval", self.P_min, self.P_max)
        print(self.velocity_min, self.velocity_max)

    def curve_fit(self, N=20):
        t0 = np.linspace(self.t0_min, self.t0_max, N)
        P = np.linspace(self.P_min, self.P_max, N)
        velocity = np.linspace(self.velocity_min, self.velocity_max, N)
        
        # t0 = np.linspace(4500, 5500, N)
        # P = np.linspace(4000, 6000, N)
        # velocity = np.linspace(5, 20, N)

        t0_m, P_m, v_m = np.meshgrid(t0, P, velocity)
        v_CM = self.velocity_CM()
        time = self.time[self.start_i: self.end_i]
        i = self.start_i
        s_sum = np.zeros((N,N,N))
        for t in time:
            v_model_r = v_m*np.cos(2*np.pi/P_m*(t0_m - t))
            distance = (v_CM[i] - v_model_r)**2
            s_sum = s_sum + distance
            i = i + 1

        ind = np.unravel_index(np.argmin(s_sum, axis=None), s_sum.shape)

        # print(v_m[ind])
        # print(P_m[ind])
        # print(t0_m[ind])

        t=self.time
        model = v_m[ind]*np.cos(2*np.pi/P_m[ind]*(t0_m[ind] - t))
        self.model = model

        plt.figure(figsize=(3.5,4))
        plt.plot(t, self.velocity_CM()),
        plt.plot(t, model)
        
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.savefig("test.png")
        plt.show()

    def qq(self):    #create Q-Q plot with 45-degree line added to plot
        data = self.star_velocity() - self.model
        sm.qqplot(data, line='s')
        plt.show()

if __name__ == "__main__":
    
    star0 = C4("star0_1.63.txt")
    star1 = C4("star1_1.15.txt")
    star2 = C4("star2_0.84.txt")
    star3 = C4("star3_3.52.txt")
    star4 = C4("star4_0.97.txt")
   
    # star0.read_data("star0_1.63.txt")
    # star0.plot_raw_data()
    # star0.star_velocity()
    star2.initial_values()
    star2.curve_fit(N=50)
    # star4.plot_velocity_CM()
    
    # star0.qq()

    # def crossing(self):
    #     # zero_crossings = np.where(np.diff(np.sign(self.star_velocity())))[0]
    #     # zero_crossings = np.where(np.diff(np.sign(self.star_velocity())))[0]
    #     x = self.velocity_CM()
    #     indices = np.where(np.logical_and(np.abs(np.diff(x)) > 2, np.diff(np.sign(x)) != 0))[0]
    #     test = np.where(np.diff(indices) > 50)[0]
    #     test1 = np.where(np.diff(test) > 8)[0]
    #     # test=np.sign(self.velocity_CM())

    #     print(indices[test[test1]])
    #     print(test)
    #     # print(len(test))
    #     # print(test1)
    #     # print(len(test1))
    #     # # plt.plot(indices)
    #     # plt.plot(test1)
    #     # plt.plot(test)
    #     # plt.show()