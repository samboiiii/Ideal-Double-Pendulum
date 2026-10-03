# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 12:17:56 2026

@author: Acer
"""

import matplotlib.pyplot as pt
import numpy as np

#create time axis
n=500000
T=500
t_grid=np.linspace(0,T,n+1)
η=T/n

#define variables
g=9.8
l1=3
l2=2
m1=5
m2=7
μ=m2/(m1+m2)

#create ODE function
def G(r):                                                       #r=(θ,w₁,φ,w₂)
    θ,w1,φ,w2=r[0],r[1],r[2],r[3]                               #define the state vector
    Δ=θ-φ                                                       #difference between angle made by pendulum 1 with the vertical and angle made by pendulum 2 with the vertical
    ẇ1=((-μ*(l2/l1)*(w2**2)*np.sin(Δ))-
        ((g/l1)*np.sin(θ))-
        (μ*(w1**2)*np.cos(Δ)*np.sin(Δ))+
        (μ*(g/l1)*np.cos(Δ)*np.sin(φ)))/(1-μ*np.cos(Δ)**2)
    ẇ2=((μ*(w2**2)*np.cos(Δ)*np.sin(Δ))+
        ((g/l2)*np.cos(Δ)*np.sin(θ))+
        ((l1/l2)*(w1**2)*np.sin(Δ))-
        ((g/l2)*np.sin(φ)))/(1-μ*np.cos(Δ)**2)
    F=np.array([w1,ẇ1,w2,ẇ2])                                   #This vector field governs the pendulum EOMs, dr/dt=F(r(t)) , F(r(t))=(w₁,ẇ₁(r(t)),w₂,ẇ₂(r(t)))
    
    ẇ1_θ=((-μ*(l2/l1)*(w2**2)*np.cos(Δ))-
          ((g/l1)*np.cos(θ))-
          (μ*(w1**2)*np.cos(2*Δ))-
          (μ*(g/l1)*np.sin(φ)*np.sin(Δ))-
          (ẇ1*μ*np.sin(2*Δ)))/(1-μ*np.cos(Δ)**2)                #This equation is ∂ẇ₁/∂θ
    ẇ1_w1=(-μ*w1*np.sin(2*Δ))/(1-μ*np.cos(Δ)**2)                #This equation is ∂ẇ₁/∂w₁
    ẇ1_φ=((μ*(l2/l1)*(w2**2)*np.cos(Δ))+
          (μ*(w1**2)*np.cos(2*Δ))+
          (μ*(g/l1)*np.cos(Δ-φ))+
          (ẇ1*μ*np.sin(2*Δ)))/(1-μ*np.cos(Δ)**2)                #This equation is ∂ẇ₁/∂φ
    ẇ1_w2=(-2*μ*(l2/l1)*w2*np.sin(Δ))/(1-μ*np.cos(Δ)**2)        #This equation is ∂ẇ₁/∂w₂
    ẇ2_θ=((μ*(w2**2)*np.cos(2*Δ))+
          ((g/l2)*np.cos(Δ+θ))+
          ((l1/l2)*(w1**2)*np.cos(Δ))-
          (ẇ2*μ*np.sin(2*Δ)))/(1-μ*np.cos(Δ)**2)                #This equation is ∂ẇ₂/∂θ
    ẇ2_w1=(2*(l1/l2)*w1*np.sin(Δ))/(1-μ*np.cos(Δ)**2)           #This equation is ∂ẇ₂/∂w₁
    ẇ2_φ=((-μ*(w2**2)*np.cos(2*Δ))+
          ((g/l2)*np.sin(Δ)*np.sin(θ))-
          ((l1/l2)*(w1**2)*np.cos(Δ))-
          ((g/l2)*np.cos(φ))+
          (ẇ2*μ*np.sin(2*Δ)))/(1-μ*np.cos(Δ)**2)                #This equation is ∂ẇ₂/∂φ
    ẇ2_w2=(μ*w2*np.sin(2*Δ))/(1-μ*np.cos(Δ)**2)                 #This equation is ∂ẇ₂/∂w₂
    DF=np.array([[0,1,0,0],
                [ẇ1_θ,ẇ1_w1,ẇ1_φ,ẇ1_w2],
                [0,0,0,1],
                [ẇ2_θ,ẇ2_w1,ẇ2_φ,ẇ2_w2]])                       #This is the 4x4 jacobian matrix of F, [DF(r(t))]
    return F,DF

# create empty state vector and perturbation vector arrays
R = np.zeros([len(t_grid), 4]) 
R[0,:] = [2*np.pi/3,0,np.pi/4,0] 
h = np.zeros([len(t_grid), 4]) 
h_initial = np.array([1e-6, 0, 1e-6, 0])
h[0,:] = h_initial / np.linalg.norm(h_initial)                  #Initialize the tangent vector to have a magnitude of exactly 1
lyap_sum = 0.0                                                  #Initialize a running sum to keep track of the logarithmic growth
λ = np.zeros(len(t_grid))                                       #to plot how the exponent converges over time

# create the RK4 loop
for i in range(len(t_grid)-1):
    #Standard RK4 for the pendulum ODEs
    k1 = G(R[i,:])[0]
    k2 = G(R[i,:] + 0.5*η*k1)[0]
    k3 = G(R[i,:] + 0.5*η*k2)[0]
    k4 = G(R[i,:] + η*k3)[0]
    
    #Standard RK4 for the perturbed vector ODE
    H1 = G(R[i,:])[1] @ h[i,:]
    H2 = G(R[i,:] + 0.5*η*k1)[1] @ (h[i,:] + 0.5*η*H1)
    H3 = G(R[i,:] + 0.5*η*k2)[1] @ (h[i,:] + 0.5*η*H2)
    H4 = G(R[i,:] + η*k3)[1] @ (h[i,:] + η*H3)
    
    #Update the state vector
    R[i+1,:] = R[i,:] + (η/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    #Calculate the UNNORMALIZED tangent vector for the next step
    h_new = h[i,:] + (η/6)*(H1 + 2*H2 + 2*H3 + H4)
    
    #Measure the magnitude
    α = np.linalg.norm(h_new)
    
    #Store the natural log of the magnitude
    lyap_sum += np.log(α)
    
    #Renormalize the tangent vector back to length 1
    h[i+1,:] = h_new / α
    
    #Store the current time-averaged Lyapunov exponent (for plotting convergence)
    λ[i+1] = lyap_sum / (t_grid[i+1])

#Calculate final maximal Lyapunov exponent
λ_max = lyap_sum / T
print(f"Maximal Lyapunov Exponent: {λ_max:.4f}")

#Plot the convergence of the exponent
fig = pt.figure()
ax = pt.axes()
ax.plot(t_grid[1:], λ[1:])                                          #Skip index 0 to avoid division by zero
ax.plot(t_grid[-1],λ[-1],marker='o',linestyle='none')
ax.axhline(λ[-1],linestyle='dashed',alpha=0.7)
ax.set_xlabel('t (sec)')
ax.set_ylabel('Running average of λ')
ax.set_title('Convergence of the Maximal Lyapunov Exponent')
pt.show()
