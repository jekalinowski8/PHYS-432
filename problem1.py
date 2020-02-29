#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finite-differencing
FCTS and Lax-Frederich Solvers

@author: Jonathan Kalinowski 
Feb. 25th 2020
"""

import numpy as np
from matplotlib import pyplot as plt

#configure initial conditions:
dx = 0.01 #step size
x = np.arange(0,1+dx,step=dx) #x values
dt = 0.01 #timestep - satisfies Courant condition for Lax-Frederich
T = 3 #total time
t=0 #track the current time
f1 = x.copy()  #f1 (FCTS) initial conditions
f2 = f1.copy() #f2 (Lax-Frederich) initial conditions
u = -.1 #velocity
alpha = u*dt/dx #alpha coefficient


plt.ion() #set up plots
fig, (ax1,ax2) = plt.subplots(1,2)
fig.set_figheight(10); fig.set_figwidth(10)
plt1, = ax1.plot(x,f1)
plt2, = ax2.plot(x,f2)
ax1.set_xlim([0,1])
ax1.set_ylim([0,2])
ax2.set_xlim([0,1])
ax2.set_ylim([0,2])
ax1.set_title("FCTS"); ax2.set_title("Lax-Frederich")
ax1.set_xlabel('x'); ax1.set_ylabel('f(x)')
ax2.set_xlabel('x'); ax2.set_ylabel('f(x)')
fig.canvas.draw()



while (t<T): #update until t=T

    f1[1:-1] -= 0.5*alpha*(f1[2:]-f1[:-2]) #FTCS w/ fixed boundary conditions
    f2[1:-1] = 0.5*(f2[2:]+f2[:-2])-0.5*alpha*(f2[2:]-f2[:-2]) #Lax-Frederich w/ fixed boundary conditions
    
    
    plt1.set_ydata(f1) #plot updated data
    plt2.set_ydata(f2)
    fig.canvas.draw()
    plt.pause(0.01)
    t+=dt #update current time


