# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:27:20 2026

@author: Acer
"""

import matplotlib.pyplot as pt
import numpy as np
#create time axis
n=50000
T=50
t_grid=np.linspace(0,T,n+1)
h=T/n

#define variables
g=9.8
l1=3
l2=2
m1=5
m2=7
μ=m2/(m1+m2)

#create ODE function
def f(r):                                                         #r=(θ,ω₁,φ,ω₂)
    θ,w1,φ,w2=r[0,:],r[1,:],r[2,:],r[3,:]                         #define the state vector
    Δ=θ-φ                                                         #difference between angle made by pendulum 1 with the vertical and angle made by pendulum 2 with the vertical
    ẇ1=((-μ*(l2/l1)*(w2**2)*np.sin(Δ))-
        ((g/l1)*np.sin(θ))-
        (μ*(w1**2)*np.cos(Δ)*np.sin(Δ))+
        (μ*(g/l1)*np.cos(Δ)*np.sin(φ)))/(1-μ*np.cos(Δ)**2)
    ẇ2=((μ*(w2**2)*np.cos(Δ)*np.sin(Δ))+
        ((g/l2)*np.cos(Δ)*np.sin(θ))+
        ((l1/l2)*(w1**2)*np.sin(Δ))-
        ((g/l2)*np.sin(φ)))/(1-μ*np.cos(Δ)**2)
    return np.array([w1,ẇ1,w2,ẇ2])                                #dr/dt=f(r(t))=(ω₁,ẇ₁(r(t)),ω₂,ẇ₂(r(t)))

#create two empty trajectory arrays
R=np.zeros([len(t_grid),4,2])                                     #Indexing of the form R[time step , variable , trajectory number].
R[0,0,:]=[2*np.pi/3,0,np.pi/4,0]                                  #r₁(0)=(2π/3,0,π/4,0)
ε=1e-6
R[1,0,:]=R[0,0,:]+np.array([ε,0,ε,0])                             #r₂(0)=r₁(0)+(ε,0,ε,0)

#create the RK4 loop
for i in range(len(t_grid)-1):
    k1=f(R[i,:,:])
    k2=f(R[i,:,:]+0.5*h*k1)
    k3=f(R[i,:,:]+0.5*h*k2)
    k4=f(R[i,:,:]+h*k3)
    R[i+1,:,:]=R[i,:,:]+(h/6)*(k1+2*k2+2*k3+k4)

#create figure and axes
fig1=pt.figure(dpi=150)
ax1=pt.axes()
fig2=pt.figure(dpi=150)
ax2=pt.axes()

#plot (t,θ) and (t,φ) for both trajectories
ax1.plot(t_grid,R[0,:,0],linewidth=1,label='Trajectory 1')
ax1.plot(t_grid,R[1,:,0],linewidth=1,label='Trajectory 2')
ax1.legend()
ax1.set_xlabel('time (sec) →')
ax1.set_ylabel('θ (rad) →')
ax1.set_title('θ vs t graph for two trajectories with initial angles 2π/3 rad and 2π/3 + ε rad respectively')
ax1.grid(True,linestyle='dashed',alpha=0.5)

ax2.plot(t_grid,R[0,:,2],linewidth=1,label='Trajectory 1')
ax2.plot(t_grid,R[1,:,2],linewidth=1,label='Trajectory 2')
ax2.legend()
ax2.set_xlabel('time (sec) →')
ax2.set_ylabel('φ (rad) →')
ax2.set_title('φ vs t graph for two trajectories with initial angles π/4 rad and π/4 + ε rad respectively')
ax2.grid(True,linestyle='dashed',alpha=0.5)
pt.show()
