# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 10:18:51 2022

@author: MaxRo
"""
import numpy as np
import scipy
from scipy.integrate import odeint,solve_ivp
import matplotlib.pyplot as plt
import time

start = time.time()
a0 = 1
b0 = 0
c0 = 1
d0 = 0

k1 = 200
k2 = 1000
k3 = 0.05

def odefunc(t,y0):
    dA = -k1*y0[0]-k2*y0[0]*y0[2]+k3*y0[3]
    dB = k1*y0[0]
    dC = -2*k2*y0[0]*y0[2] + 2*k3*y0[3]
    dD = k2*y0[0]*y0[2]-k3*y0[3]
    return [dA,dB,dC,dD]

y0 = [a0,b0,c0,d0]

tspan = np.arange(0,100,0.0001)
#tspan = [0,100]
#sol = odeint(odefunc,y0,tspan)
sol = solve_ivp(odefunc,t_span=(0,100),y0=y0,method="RK45",t_eval=tspan)
#sol = np.transpose(sol)

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.plot(tspan,sol['y'][0],label="A")
ax.plot(tspan,sol['y'][1],label="B")
ax.plot(tspan,sol['y'][2],label="C")
ax.plot(tspan,sol['y'][3],label="D")
fig.legend()
e_time = time.time() - start
print(e_time)

