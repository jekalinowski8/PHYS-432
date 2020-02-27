#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finite-differencing
Lax-Frederich Solver w/ diffusion

@author: Jonathan Kalinowski 
Feb. 27th 2020
"""

import numpy as np
from matplotlib import pyplot as plt

dx = 0.01 #step size
x = np.arange(0,1+dx,step=dx) #x values
n = x.size
dt = 0.01 #timestep - satisfies Courant condition for Lax-Frederich
T = 5 #total time
t=0 #track the current time
f1 = x.copy()  #f1 initial conditions
f2 = f1.copy() #f2 initial conditions
#u = -.1
D1 = .1; D2 = 1; 
u = -.1*np.ones(x.size) #velocity
alpha = u*dt/dx
beta1 = D1*dt/dx**2; beta2 = D2*dt/dx**2
plt.ion() #configure plotting and plot initial conditions
fig, (ax1,ax2) = plt.subplots(1,2)
plt1, = ax1.plot(x,f1)
plt2, = ax2.plot(x,f2)
ax1.set_xlim([0,1])
ax1.set_ylim([0,2])
ax2.set_xlim([0,1])
ax2.set_ylim([0,2])
ax1.set_title("D="+str(D1))
ax2.set_title("D="+str(D2))

fig.canvas.draw()



while (t<T): #update until t=dt
    A1 = np.eye(n)*(1.0+2.0*beta1)+np.eye(n,k=1)*-beta1+np.eye(n,k=-1)*-beta1
    A1[0][0]=1;A1[0][1]=0; A1[-1][-1]=1+beta1
    f1 = np.linalg.solve(A1,f1)
    f1[1:-1] = 0.5*(f1[2:]+f1[:-2])-0.5*alpha[1:-1]*(f2[2:]-f2[:-2])#Lax-Frederich w/ fixed boundary conditions
    
    A2 = np.eye(n)*(1.0+2.0*beta2)+np.eye(n,k=1)*-beta2+np.eye(n,k=-1)*-beta2
    A2[0][0]=1;A2[0][1]=0; A2[-1][-1]=1+beta2
    f2 = np.linalg.solve(A2,f2)
    f2[1:-1] = 0.5*(f2[2:]+f2[:-2])-0.5*alpha[1:-1]*(f2[2:]-f2[:-2])#Lax-Frederich w/ fixed boundary conditions
    
    plt1.set_ydata(f1) #plot updated data
    plt2.set_ydata(f2)
    fig.canvas.draw()
    plt.pause(0.001)
    t+=dt #update current time


