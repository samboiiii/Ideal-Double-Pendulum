# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 22:11:48 2026

@author: Acer
"""

import matplotlib.pyplot as pt
from matplotlib.animation import FuncAnimation
import numpy as np

#create time axis
n=30000
T=30
t=np.linspace(0,T,n+1)
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
    θ,w1,φ,w2=r[0],r[1],r[2],r[3]                                 #define the state vector
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

#create empty state vector array
R=np.zeros([len(t),4])                                            #Indexing of the form R[time step, variable]
R[0,:]=[2*np.pi/3,0,np.pi/4,0]                                    #initial state r(0)=(2π/3,0,π/4,0)

#create the RK4 loop
for i in range(len(t)-1):
    k1=f(R[i,:])
    k2=f(R[i,:]+0.5*h*k1)
    k3=f(R[i,:]+0.5*h*k2)
    k4=f(R[i,:]+h*k3)
    R[i+1,:]=R[i,:]+(h/6)*(k1+2*k2+2*k3+k4)

#create figure and axes
fig=pt.figure(dpi=150)
ax=pt.axes(xlim=(-1.5*(l1+l2),1.5*(l1+l2)),ylim=(-1.5*(l1+l2),1.5*(l1+l2)))
ax.set_xlabel(r'$x\;(\mathrm{m})\rightarrow$')
ax.set_ylabel(r'$y\;(\mathrm{m})\rightarrow$')
ax.set_title('Numerical simulation of the double pendulum',fontsize=10)
ax.set_aspect('equal')
ax.grid(True)

#define the pendulum positions
Θ=R[:,0]                  #all θ values, this is a len(t)x1 vector
Φ=R[:,2]                  #all φ values, this is a len(t)x1 vector
x1=l1*np.sin(Θ)           #l₁sin(θ)
y1=-l1*np.cos(Θ)          #-l₁cos(θ)
x2=x1+l2*np.sin(Φ)        #l₁sin(θ)+l₂sin(φ)
y2=y1-l2*np.cos(Φ)        #-l₁cos(θ)-l₂cos(φ)

#create string, bob and trail placeholder plots
string,=ax.plot([],[])
bob1,=ax.plot([],[],marker='o',label=r'$(\theta(0),\omega_1(0))=\left(\frac{2\pi}{3},0\right)$')
bob2,=ax.plot([],[],marker='o',label=r'$(\phi(0),\omega_2(0))=\left(\frac{\pi}{4},0\right)$')
trail1,=ax.plot([],[],linestyle='dashed',alpha=0.5)
trail2,=ax.plot([],[],linestyle='dashed',alpha=0.5)

#define animation parameters
fps=60
total_frames=T*fps
frame_skip=int(n/total_frames)

#create the frame update function
def update(frame):
    current_x1=x1[frame*frame_skip]                                             #multiplying by frame_skip to skip some points because we want to fit 50000 data points within 3000 frames in 50 seconds to get a smooth 60fps playback
    current_x2=x2[frame*frame_skip]
    current_y1=y1[frame*frame_skip]
    current_y2=y2[frame*frame_skip]
    string.set_data([0,current_x1,current_x2],[0,current_y1,current_y2])
    bob1.set_data([current_x1],[current_y1])
    bob2.set_data([current_x2],[current_y2])
    trail1.set_data(x1[:frame*frame_skip],y1[:frame*frame_skip])
    trail2.set_data(x2[:frame*frame_skip],y2[:frame*frame_skip])
    return string,bob1,bob2,trail1,trail2

#animate
anim=FuncAnimation(fig,update,frames=total_frames,interval=16,blit=True)
pt.legend(fontsize=7)
print('saving animation')
anim.save("double_pendulum.html",writer='html',fps=60)
