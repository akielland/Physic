# kode for exercise 1B.7
# egen kode
import numpy as np
import matplotlib.pyplot as plt
import my_solar_system as my     

N = 4000
dt = 400_000 #timestep
T = dt * N #total time
time_points = np.linspace(0, T, N) 

r_v_t = np.zeros((3, 4, N))
AU = my.const.AU

r_v_t[:, 0:2, 0] = AU * np.array([[-1.5,0],[0,0],[3,0]]) #initial positions
r_v_t[:, 2:4, 0] = 1000 * np.array([[0,-1],[0,30],[0,-7.5]]) #initial velocities

def RHS(r_v):
    # Right hand side of the differential equation
    G = my.const.G
    planet_m, star1_m, star2_m  = [6.39e23, 2e30, 8e30]

    # Forces between the objects
    F_p_s1 = G * planet_m * star1_m / np.linalg.norm(r_v[0,0:2]-r_v[1,0:2])**3
    F_p_s2 = G * planet_m * star2_m / np.linalg.norm(r_v[0,0:2]-r_v[2,0:2])**3
    F_s1_s2 = G * star1_m * star2_m / np.linalg.norm(r_v[1,0:2]-r_v[2,0:2])**3

    # Using the forces
    da_p = (F_p_s1*(r_v[1,0:2]-r_v[0,0:2]) + F_p_s2*(r_v[2,0:2]-r_v[0,0:2]))/planet_m 
    da_s1 = (F_p_s1*(r_v[0,0:2]-r_v[1,0:2]) + F_s1_s2*(r_v[2,0:2]-r_v[1,0:2]))/star1_m
    da_s2 = (F_p_s2*(r_v[0,0:2]-r_v[2,0:2]) + F_s1_s2*(r_v[1,0:2]-r_v[2,0:2]))/star2_m

    dv_p = r_v[0,2:4]
    dv_s1 = r_v[1,2:4]
    dv_s2 = r_v[2,2:4] 
    drdt = np.empty((3,4))

    drdt[0,0:2] = dv_p
    drdt[1,0:2] = dv_s1
    drdt[2,0:2] = dv_s2
    drdt[0,2:4] = da_p
    drdt[1,2:4] = da_s1
    drdt[2,2:4] = da_s2

    return drdt

  
def RK4(r_v, dt):
    # Runge-Kutta 4
    k1 = RHS(r_v)
    k2 = RHS(r_v+k1*dt/2)
    k3 = RHS(r_v+k2*dt/2)
    k4 = RHS(r_v+k3*dt)
    drdt = (k1 + 2*k2 + 2*k3 + k4)/6
    return drdt

# Running the integration
for t in range(N-1):
    drdt = RK4(r_v_t[:,:,t], dt) 
    r_v_t[:,:,t+1] = r_v_t[:,:,t] + drdt*dt

r_v_t = r_v_t/AU
plt.plot(r_v_t[0,0,:],r_v_t[0,1,:],label="planet")
plt.plot(r_v_t[1,0,:],r_v_t[1,1,:],label="small star")
plt.plot(r_v_t[2,0,:],r_v_t[2,1,:],label="large star")
plt.axis("equal")
plt.xlabel("x-position [AU]")
plt.ylabel("y-position [AU]")
plt.legend()
plt.show()
