import numpy as np
import sys as sys
import matplotlib.pyplot as plt

n = 100000                    #Antall hydrogenmolekyler
kb = 1.3806*10**(-23)    #boltzmann konstant
T = 10000                      #Temperatur
m = 3.32*10**(-27)         #Massen til et hydrogenmolekyl           
SD = np.sqrt((kb*T)/m)    #Standardavviket
L = 10**(-6)                    #Box sidelength

vel = np.random.normal(loc=0,scale=SD,size=(n,3)) #Drawing velocities from the normal distribution
pos = np.random.rand(n,3)*L   #Drawing positions from the uniform distribution between 0 and 10**-6



#Calculating the mean kinetic energy of the molecules
Ek = 0
for i in range(0,n):
    Ek += (vel[i][0]**2 + vel[i][1]**2 + vel[i][2]**2)*0.5*m
Ek = Ek/n
AnalyEk = (3/2)*kb*T #Calculating the mean kinetic energy using the analytic expression
print(AnalyEk,Ek)



#Calculating mean absolute velocity of the created molecules
meanv = 0
for i in range(0,n):
    meanv += np.linalg.norm(vel[i])
meanv = meanv/n
analymeanv = np.sqrt((8*kb*T)/(np.pi*m))  #calculating mean absolute velocity analyticaly
print(analymeanv,meanv)


#Calculating the mean pressure over a time period of 10**-9 s
time = 10**(-9)
dt = 10**(-12)
k = int(time/dt)
r = np.zeros((k,3,n))
u = np.zeros((k,3,n))
u[0] = np.transpose(vel)
Sum = np.zeros((k,3,n))
r[0] = np.transpose(pos)
momentum = 0
for i in range(0,k-1):
    Sum = np.sum(np.where(r[i][1] < L, 0, 2*abs(u[i][1])))
    momentum += Sum*m
    u[i+1] = np.where(np.logical_or(r[i] < 0, r[i] > L),-u[i],u[i])
    r[i+1] = r[i] + dt*u[i+1]

avrF = momentum/time
avrp = avrF/(L*L)
analyp = (n*kb*T)/(L**3)    #Calculating the pressure given by the analytic expression
print(analyp,avrp)

#Simulating with a square hole in the bottom of the box with sides of length L/2
r = np.zeros((k,3,n))
u = np.zeros((k,3,n))
u[0] = np.transpose(vel)
Sum = np.zeros((k,3,n))
r[0] = np.transpose(pos)
momentum = 0
escaped = 0
for i in range(0,k-1):
    #The following line checks for molecules that are exiting, and sums up their speedcomponent in z-direction.(And multiplies it by 2)
    Sum = np.sum(np.where(np.logical_and(np.logical_and(np.logical_and(r[i][0] > L/4, r[i][0] < 3*L/4),np.logical_and(r[i][1] > L/4, r[i][1] < 3*L/4)),r[i][2] < 0), abs(u[i][2]), 0))
    momentum += Sum*m
    #The next line checks for escaping molecules, and counts them up
    escaped += np.sum(np.where(np.logical_and(np.logical_and(np.logical_and(r[i][0] > L/4, r[i][0] < 3*L/4),np.logical_and(r[i][1] > L/4, r[i][1] < 3*L/4)),r[i][2] < 0), 1, 0))
    #The next line moves the escaped molecules to the top of the box (the opposite side of the hole) in order to keep the pressure constant
    r[i][2] = np.where(np.logical_and(np.logical_and(np.logical_and(r[i][0] > L/4, r[i][0] < 3*L/4),np.logical_and(r[i][1] > L/4, r[i][1] < 3*L/4)),r[i][2] < 0), L-10**(-10), r[i][2])
    u[i][0] = np.where(np.logical_and(np.logical_and(np.logical_and(r[i][0] > L/4, r[i][0] < 3*L/4),np.logical_and(r[i][1] > L/4, r[i][1] < 3*L/4)),r[i][2] < 0), np.random.normal(loc=0,scale=SD), u[i][0])
    u[i][1] = np.where(np.logical_and(np.logical_and(np.logical_and(r[i][0] > L/4, r[i][0] < 3*L/4),np.logical_and(r[i][1] > L/4, r[i][1] < 3*L/4)),r[i][2] < 0), np.random.normal(loc=0,scale=SD), u[i][1])
    u[i][2] = np.where(np.logical_and(np.logical_and(np.logical_and(r[i][0] > L/4, r[i][0] < 3*L/4),np.logical_and(r[i][1] > L/4, r[i][1] < 3*L/4)),r[i][2] < 0), np.random.normal(loc=0,scale=SD), u[i][2])
    u[i+1] = np.where(np.logical_or(r[i] < 0, r[i] > L),-u[i],u[i])
    r[i+1] = r[i] + dt*u[i+1]

M = 1000
speedgain = momentum/M
print(escaped)
print(speedgain)
speedgain20min = speedgain*10**9*60*20
print(speedgain20min)
vesc = 10750
nboxes = vesc/speedgain20min
print (nboxes)


"""
Kjøre eksempel:
$python3 1A.6.py
2.0708999999999998e-19 2.0683750706916262e-19
10290.463814120372 10279.491157649065
13806.0 13675.385484247703
71216
1.9296964821049357e-21
2.315635778525923e-09
4642353559955.4375
"""


