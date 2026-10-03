# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:25:57 2026

@author: Acer
"""

import matplotlib.pyplot as pt
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

# Windows Setup
fig1=pt.figure(dpi=180)
ax1=pt.axes()
fig2=pt.figure(dpi=180)
ax2=pt.axes()

#Map angles greater than 2π onto [-π, π]
θ_map=(R[:,0]+np.pi)%(2*np.pi)-np.pi
φ_map=(R[:,2]+np.pi)%(2*np.pi)-np.pi
#Explaination for how the mapping works:
#For any two real numbers a,b: a mod b = a-(b*⌊a/b⌋) (b≠0) where ⌊x⌋ is the greatest integer (floor) function
#Now ⌊x⌋<=x<=⌊x⌋+1 ∀x∈R
#Thus ⌊a/b⌋<=a/b<=⌊a/b⌋+1
#For b>0 , b≠0 we have: b*⌊a/b⌋<=b*(a/b)<=b*(⌊a/b⌋+1) ⟹ b*⌊a/b⌋<=a<=b*⌊a/b⌋+b 
#So, 0<=a-(b*⌊a/b⌋)<=b
#Thus, 0<=(θ+π)-(2π*⌊(θ+π)/2π⌋)<=2π ⟹ 0<=(θ+π) mod 2π<=2π ⟹ -π<=[(θ+π) mod 2π]-π<=π

#plot state space orbits (θ_map,ω₁) and (φ_map,ω₂)
ax1.plot(θ_map,R[:,1],marker='.',markersize=0.3,linestyle='None')
ax1.set_xlabel('θ (rad) →')
ax1.set_ylabel('ω₁ (rad/s) →')
ax1.set_title('State space orbit for first pendulum')
ax2.plot(φ_map,R[:,3],color='orange',marker='.',markersize=0.3,linestyle='None')
ax2.set_xlabel('φ (rad) →')
ax2.set_ylabel('ω₂ (rad/s) →')
ax2.set_title('State space orbit for second pendulum')

#Display the angles in radian on the x axis
ticks=[-np.pi, -2*np.pi/3, -np.pi/3, 0, np.pi/3, 2*np.pi/3, np.pi]
labels=['-π','-2π/3','-π/3','0','π/3','2π/3','π']
for ax in (ax1,ax2):
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.grid(True, linestyle='dashed', alpha=0.3)
pt.show()