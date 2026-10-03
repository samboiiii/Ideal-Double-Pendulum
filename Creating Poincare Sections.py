# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 18:14:11 2026

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

#create empty state vector array
sim_num=1000                                                      #number of simulated trajectories
R=np.zeros([len(t_grid),4,sim_num])                               #Indexing of the form R[time step, state variable, trajectory number]

#Define energy level and initial conditions that satisfy the constant energy constraint
E=490
K=(-E-(m2*g*l2))/((m1+m2)*g*l1)
rng=np.random.default_rng(seed=42)
if K<=-1:
    R[0,0,:]=rng.uniform(-np.pi,np.pi,sim_num)                    #Generates sim_num random numbers from a uniform distribution of [-π,π]. Since the seed is pre-set, identical numbers are generated every time the code is run
elif -1<K<1:
    R[0,0,:]=rng.uniform(-np.arccos(K),np.arccos(K),sim_num)      #Generates sim_num random numbers from a uniform distribution of [-cos⁻¹(K),cos⁻¹(K)]. Since the seed is pre-set, identical numbers are generated every time the code is run
else:
    print("K>1, energy level is physically impossible")
Θ=R[0,0,:]                                                        #Initial θ values for all trajectories
V=(-(m1+m2)*g*l1*np.cos(Θ))-(m2*g*l2)                             #Initial potential energy of the system with φ=0 for all trajectories
L=np.sqrt(2*(E-V)/((l1**2)*(m1+m2)))
R[0,1,:]=rng.uniform(-L,L,sim_num)
Ω=R[0,1,:]                                                        #Initial ω₁ values for all trajectories
A=m2*(l2**2)/2
B=m2*l1*l2*Ω*np.cos(Θ)
C=(((m1+m2)*(l1**2)*(Ω**2))/2)+V
D=np.sqrt((B**2)-(4*A*(C-E)))
R[0,3,:]=(-B+D)/(2*A)                                             #Initial ω₂ values for all trajectories

#create the RK4 loop
for i in range(len(t_grid)-1):
    k1=f(R[i,:,:])
    k2=f(R[i,:,:]+0.5*h*k1)
    k3=f(R[i,:,:]+0.5*h*k2)
    k4=f(R[i,:,:]+h*k3)
    R[i+1,:,:]=R[i,:,:]+(h/6)*(k1+2*k2+2*k3+k4)

#map every angle φ onto [-π,π]
φ_map=(R[:,2,:]+np.pi)%(2*np.pi)-np.pi #This is a 50000x3 matrix

#Extract times when φ changes sign i.e. φ(tᵢ)<0 and φ(tᵢ₊₁)>0 and also when ω₂>0
sign_changes=(φ_map[:-1,:]<0) & (φ_map[1:,:]>0)                   #This creates a 50000x3 matrix containing True or False boolean values

#Because we have mapped every angle onto [-π,π], angles like 3.15 are mapped to -3.13.
#Thus if φ(tᵢ)=-3.13 and φ(tᵢ₊₁)=3.14 then φ DOES change sign, however the second pendulum never crosses the lowermost point φ=0, which is the constraint we have set to determine the initial θ,ω₁ and ω₂ values.
#So we need to filter all the sign changes caused due to the angle mapping and not because the second pendulum crosses the lowermost point.

not_a_mapping_jump=np.abs(φ_map[1:,:]-φ_map[:-1,:])<np.pi                               #The difference between angles after every mapping jump is greater than π. So if the sign change is really because the second pendulum crosses the lowermost point then φ(tᵢ₊₁)-φ(tᵢ) will be < π
valid_ω2=R[:-1,3,:]>0                                                                   #This creates a 50000x3 matrix containing True or False values
crossing_mask=sign_changes & not_a_mapping_jump & valid_ω2                              #All three conditions φ(tᵢ)<0,φ(tᵢ₊₁)>0 and ω₂>0 must be met
row_idx,col_idx=np.where(crossing_mask)                                                 #Extracts all row and column indices where crossing_mask is True

#Now, model φ(t) as a linear function between the time interval [tᵢ,tᵢ₊₁]
#m=[φ(tᵢ₊₁)-φ(tᵢ)]/(tᵢ₊₁-tᵢ) , slope of the straight line between (tᵢ,φ(tᵢ)) and (tᵢ₊₁,φ(tᵢ₊₁))
#[φ(t)-φ(tᵢ)]/(t-tᵢ)=[φ(tᵢ₊₁)-φ(tᵢ)]/(tᵢ₊₁-tᵢ) , slope of a linear function is constant
#So, φ(t) = φ(tᵢ) + [φ(tᵢ₊₁)-φ(tᵢ)](t-tᵢ)/(tᵢ₊₁-tᵢ)
#Now find the exact time t=t_crossing when φ(t_crossing)=0
# 0 = φ(tᵢ) + [φ(tᵢ₊₁)-φ(tᵢ)](t_crossing-tᵢ)/(tᵢ₊₁-tᵢ)
# -φ(tᵢ)/[φ(tᵢ₊₁)-φ(tᵢ)] = (t_crossing-tᵢ)/(tᵢ₊₁-tᵢ)
#Define r=-φ(tᵢ)/[φ(tᵢ₊₁)-φ(tᵢ)]. Then r=(t_crossing-tᵢ)/(tᵢ₊₁-tᵢ)⟹tᵢ+r(tᵢ₊₁-tᵢ)=t_crossing
#Similarly modelling θ(t),ω₁(t),ω₂(t) as linear functions between the time interval [tᵢ,tᵢ₊₁] gives:
#θ(t_crossing)=θ(tᵢ)+r[θ(tᵢ₊₁)-θ(tᵢ)] , ω₁(t_crossing)=ω₁(tᵢ)+r[ω₁(tᵢ₊₁)-ω₁(tᵢ)], ω₂(t_crossing)=ω₂(tᵢ)+r[ω₂(tᵢ₊₁)-ω₂(tᵢ)]

φ_i=φ_map[row_idx,col_idx]
φ_next=φ_map[row_idx+1,col_idx]
r=-φ_i/(φ_next-φ_i)
θ_crossing=R[row_idx,0,col_idx]+r*(R[row_idx+1,0,col_idx] - R[row_idx,0,col_idx])        #A 1D vector containing θ values for all trajectories at the exact moment φ=0 and ω₂>0
ω1_crossing=R[row_idx,1,col_idx]+r*(R[row_idx+1,1,col_idx]-R[row_idx,1,col_idx])         #A 1D vector containing ω₁ values for all trajectories at the exact moment φ=0 and ω₂>0
ω2_crossing=R[row_idx,3,col_idx]+r*(R[row_idx+1,3,col_idx]-R[row_idx,3,col_idx])
θ_crossing_map=(θ_crossing+np.pi)%(2*np.pi)-np.pi
p_θ=((m1+m2)*(l1**2)*(ω1_crossing))+(m2*l1*l2*(ω2_crossing)*np.cos(θ_crossing))          #Generalized momentum ∂L/∂ω₁

# Create the Poincaré section plot
fig1=pt.figure(figsize=(10,8))
ax1=pt.axes()
fig2=pt.figure(figsize=(10,8))
ax2=pt.axes()

# col_idx tells matplotlib which trajectory owns each point, coloring them automatically
ax1.scatter(θ_crossing_map, ω1_crossing, c=col_idx, cmap='tab20', s=2, alpha=0.7)
ax1.set_title(f"Poincaré Section for E = {E} ({sim_num} Trajectories)")
ax1.set_xlabel("θ (rad)")
ax1.set_ylabel("ω₁ (rad/s)")
ax1.grid(True, linestyle='--', alpha=0.5)
ax2.scatter(θ_crossing_map, p_θ, c=col_idx, cmap='tab20', s=2, alpha=0.7)
ax2.set_title(f"Poincaré Section for E = {E} ({sim_num} Trajectories)")
ax2.set_xlabel("θ (rad)")
ax2.set_ylabel("p_θ (kg m²/s)")
ax2.grid(True, linestyle='--', alpha=0.5)
pt.show()
 