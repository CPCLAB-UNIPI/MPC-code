# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:11:54 2016

@author: marcovaccari
"""
from __future__ import division
from builtins import range
from past.utils import old_div
from casadi import *
from casadi.tools import *
from matplotlib import pylab as plt
import math
import scipy.linalg as scla
import numpy as np
from Utilities import*

### 1) Simulation Fundamentals

# 1.1) Simulation discretization parameters
Nsim = 1000 # Simulation length

N = 50    # Horizon

h = 5.0 # Time step

# 3.1.2) Symbolic variables
xp = SX.sym("xp", 6)    # process state vector        # 6 --> 4 tanks + 2 valves output
x = SX.sym("x", 6)      # model state vector          # 6 --> 4 tanks + 2 valves output
u = SX.sym("u", 2)      # control vector              # 2 --> 2 valves input
y = SX.sym("y", 2)      # measured output vector      # 2 --> 2 lower tanks
d = SX.sym("d", 2)      # disturbance                 # 2 --> 2 lower tanks
                       

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### 2) Process and Model construction 
# 2.1) Process Parameters

# Time CONTINUOUS dynamics of 4 tanks   
def Fdyn_p(x,u):
    # TANKS parameters
    
    # g acceleration of gravity [cm/s^2]
    g = 981.0
    
    # ai cross-section of the outlet hole [cm^2]
    a1 = 0.071
    a2 = 0.057
    a3 = 0.071
    a4 = 0.057
    
    # Ai cross-section of Tank [cm^2]
    A1 = 28.0
    A2 = 32.0
    A3 = 28.0
    A4 = 32.0
    
    # gmi flow splitting factor in (0,1)
    gm1 = 0.7
    gm2 = 0.6
    
    # max tank level [cm]
    h1_max = 20.0
    h2_max = 20.0
    
    # max flowrate [cm^3/s]
    q1_max = (a1+a4)*(2.0*g*h1_max)**0.5
    q2_max = (a2+a3)*(2.0*g*h2_max)**0.5
    K1 = old_div(q1_max,100.0)
    K2 = old_div(q2_max,100.0)

    fx = SX(4,1)
    
    # To avoid numerical instability
    for i in range(x.shape[0]):
        x[i] = if_else(x[i]<0, 0., x[i])
        x[i] = if_else(x[i]>20, 20., x[i])

    # TC system of equations:
    ## tank #1 x[2]:fx_p[2] (left lower)
    fx[0] = -(old_div(a1,A1))*(2.0*g*x[0])**0.5 + (old_div(a3,A1))*(2.0*g*x[2])**0.5 + (old_div(gm1,A1))*K1*u[0]
    
    ## tank #2 x[3]:fx_p[3] (right lower)
    fx[1] = -(old_div(a2,A2))*(2.0*g*x[1])**0.5 + (old_div(a4,A2))*(2.0*g*x[3])**0.5 + (old_div(gm2,A2))*K2*u[1]
    
    ## tank #3 x[4]:fx_p[4] (left upper)
    fx[2] = -(old_div(a3,A3))*(2.0*g*x[2])**0.5 + (old_div((1.0 - gm2),A3))*K2*u[1]
    
    ## tank #4 x[5]:fx_p[5] (right upper)
    fx[3] = -(old_div(a4,A4))*(2.0*g*x[3])**0.5 + (old_div((1.0 - gm1),A4))*K1*u[0]
    
    return fx

# State map
def User_fxp_Dis(x,t,u,pxp,pxmp):
    """
    SUMMARY:
    It constructs the function User_fxp_Dis for the non-linear case
    
    SYNTAX:
    assignment = User_fxp_Cont(xp,t,u)
  
    ARGUMENTS:
    + xp,u          - Process State and input variable    
    + t             - Variable that indicate the current iteration
    
    OUTPUTS:
    + fx_p          - Non-linear plant function     
    """ 
    ## Corresponding time DISCRETE dynamics of 4 tanks (integration by hand)
             
    # initialize variables
    fx_p = SX(x.size1(),1)
      
    # Explicit Runge-Kutta 4 (TC dynamics integrateed by hand)
    Mx = 5                   # Number of elements in each time step
    dt = old_div(h,Mx)
    x0 = x[2:6]
    fx_p[0:2] = u   
    for i in range(Mx):         
        k1 = Fdyn_p(x0, u)
        k2 = Fdyn_p(x0 + dt/2.0*k1, u)
        k3 = Fdyn_p(x0 + dt/2.0*k2, u)
        k4 = Fdyn_p(x0 + dt*k3, u)
        x0 = x0 + (old_div(dt,6.0))*(k1 + 2.0*k2 + 2.0*k3 + k4)
    fx_p[2:6] = x0
    
    return fx_p

# Output Map
def User_fyp(x,u,t,pyp,pymp):
    """
    SUMMARY:
    It constructs the function User_fyp for the non-linear case
    
    SYNTAX:
    assignment = User_fyp(x,t)
  
    ARGUMENTS:
    + x             - State variable
    + t             - Variable that indicate the current iteration
    
    OUTPUTS:
    + fy_p          - Non-linear plant function     
    """
    # (output equation) 
    fy_p = vertcat\
    (\
    x[2],\
    x[3] \
    )
    
    return fy_p

# Additive State Disturbances 
def def_pxp(t):
    """
    SUMMARY:
    It constructs the additive disturbances for the linear case
    
    SYNTAX:
    assignment = defdp(k)
  
    ARGUMENTS:
    + t             - Variable that indicate the current time
    
    OUTPUTS:
    + dxp      - State disturbance value      
    """ 

    if t <= 2250:
        dxp = np.array([0., 0., 0.5, 0., 0., 0.]) # State disturbance
    elif t <= 4000:
        dxp = np.array([0., 0., 0., 0.5, 0., 0.]) # State disturbance
    else: 
        dxp = np.array([0., 0., 0., 0., 0., 0.]) # State disturbance
        
    return [dxp]



# 2.2) Model Parameters

## Time CONTINUOUS dynamics of 4 tanks
def Fdyn_m(x,u):
    # TANKS parameters
    
    # g acceleration of gravity [cm/s^2]
    g = 981.0
    
    # ai cross-section of the outlet hole [cm^2]
    a1 = 0.071
    a2 = 0.057
    a3 = 0.071
    a4 = 0.057
    
    # Ai cross-section of Tank [cm^2]
    A1 = 28.0
    A2 = 32.0
    A3 = 28.0
    A4 = 32.0
    
    # gmi flow splitting factor in (0,1)
    gm1 = 0.7
    gm2 = 0.6
    
    # max tank level [cm]
    h1_max = 20.0
    h2_max = 20.0
    
    # max flowrate [cm^3/s]
    q1_max = (a1+a4)*(2.0*g*h1_max)**0.5
    q2_max = (a2+a3)*(2.0*g*h2_max)**0.5
    K1 = old_div(q1_max,100.0)
    K2 = old_div(q2_max,100.0)

    fx = SX(4,1)
    
    # To avoid numerical instability
    for i in range(x.shape[0]):
        x[i] = if_else(x[i]<0, 0., x[i])
        x[i] = if_else(x[i]>20, 20., x[i])

    # TC system of equations:
    ## tank #1 x[2]:fx_p[2] (left lower)
    fx[0] = -(old_div(a1,A1))*(2.0*g*x[0])**0.5 + (old_div(a3,A1))*(2.0*g*x[2])**0.5 + (old_div(gm1,A1))*K1*u[0]
    
    ## tank #2 x[3]:fx_p[3] (right lower)
    fx[1] = -(old_div(a2,A2))*(2.0*g*x[1])**0.5 + (old_div(a4,A2))*(2.0*g*x[3])**0.5 + (old_div(gm2,A2))*K2*u[1]
    
    ## tank #3 x[4]:fx_p[4] (left upper)
    fx[2] = -(old_div(a3,A3))*(2.0*g*x[2])**0.5 + (old_div((1.0 - gm2),A3))*K2*u[1]
    
    ## tank #4 x[5]:fx_p[5] (right upper)
    fx[3] = -(old_div(a4,A4))*(2.0*g*x[3])**0.5 + (old_div((1.0 - gm1),A4))*K1*u[0]
    
    return fx
    
    
# State Map    
def User_fxm_Dis(x,u,d,t,px):
    """
    SUMMARY:
    It constructs the function User_fxm_Dis for the non-linear case
    
    SYNTAX:
    assignment = User_fxm_Dis(x,u,d,t)
  
    ARGUMENTS:
    + x,u,d             - State, input and disturbance variable
    + t                 - Variable that indicate the real time
    
    OUTPUTS:
    + fx_model          - Non-linear MODEL plant function     
    """ 
    ## Corresponding time DISCRETE dynamics of 4 tanks (integration by hand)
    
    # initialize variables
    fx_model = SX(x.size1(),1)

    # Explicit Runge-Kutta 4 (TC dynamics integrateed by hand)       
    Mx = 5                   # Number of elements in each time step
    dt = old_div(h,Mx)
    x0 = x[2:6]
    fx_model[0:2] = u   
    for i in range(Mx):         
        k1 = Fdyn_m(x0, u)
        k2 = Fdyn_m(x0 + dt/2.0*k1, u)
        k3 = Fdyn_m(x0 + dt/2.0*k2, u)
        k4 = Fdyn_m(x0 + dt*k3, u)
        x0 = x0 + (old_div(dt,6.0))*(k1 + 2.0*k2 + 2.0*k3 + k4)
    fx_model[2:6] = x0
                               
    return fx_model

# Output Map
def User_fym(x,u,d,t,px):
    
    """
    SUMMARY:
    It constructs the function User_fym for the non-linear case
    
    SYNTAX:
    assignment = User_fym(x,t)
  
    ARGUMENTS:
    + x             - State variable
    + t             - Variable that indicate the current iteration
    
    OUTPUTS:
    + fy_model          - Non-linear MODEL plant function
     
    """ 
   
    # (output equation)  
    # --> PV (h1, h2)    
    fy_model = vertcat\
                (\
                x[2],\
                x[3]\
                )
        
    return fy_model

# 2.3) Disturbance model for Offset-free control
offree = "lin" 
Bd = np.zeros((x.size1(),d.size1()))
Cd = np.eye(d.size1())
            
# 2.4) Initial condition
x0_p = np.array([39.5794, 38.1492, 11.9996, 12.1883, 1.51364, 1.42194])      # [plant]  
x0_m = np.array([39.5794, 38.1492, 11.9996, 12.1883, 1.51364, 1.42194])      # [model]   
u0 = np.array([39.5794, 38.1492])


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### 3) State Estimation
    
## Luemberger observer tuning params 
lue = True # Set True if you want the Luemberger observer
nx = x.size1()
ny = y.size1()
nd = d.size1()
Kx = np.zeros((nx,ny))
Kd = np.eye(nd)
K = np.row_stack([Kx,Kd])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### 4) Steady-state and dynamic optimizers

# 4.1) Setpoints
def defSP(t):
    """
    SUMMARY:
    It constructs the setpoints vectors for the steady-state optimisation 
    
    SYNTAX:
    assignment = defSP(t)
  
    ARGUMENTS:
    + t             - Variable that indicates the current time [min]
    
    OUTPUTS:
    + ysp, usp      - Input and output setpoint values      
    """ 
    if t<= 50:
        ysp = np.array([11.9996, 12.1883])                            # Output setpoint
        usp = np.array([39.5185, 38.1743])                            # Input setpoints
        xsp = np.array([50.0, 50.0, 10.0, 10.0, 2.0, 2.0])            # State setpoints
    elif t>50 and t<=1000:
        ysp = np.array([11.9996, 6.0])                                # Output setpoint
        usp = np.array([39.5185, 38.1743])                            # Control setpoints
        xsp = np.array([60.0, 50.0, 12.0, 8.0, 2.0, 2.0])             # State setpoints                
    elif t>1000 and t<=2000:
        ysp = np.array([6.0, 6.0])                                    # Output setpoint
        usp = np.array([39.5185, 38.1743])                            # Control setpoints
        xsp = np.array([60.0, 40.0, 12.0, 8.0, 2.0, 2.0])             # State setpoints
    elif t>2000 and t<=3000:
        ysp = np.array([12.0, 12.0])                                  # Output setpoint
        usp = np.array([39.5185, 38.1743])                            # Control setpoints
        xsp = np.array([40.0, 40.0, 8.0, 8.0, 2.0, 2.0])              # State setpoints
    elif t>3000 and t<=4000:
        ysp = np.array([8.0, 12.0])                                   # Output setpoint
        usp = np.array([39.5185, 38.1743])                            # Control setpoints
        xsp = np.array([40.0, 60.0, 8.0, 12.0, 2.0, 2.0])             # State setpoints
    elif t>4000 and t<=5000:
        ysp = np.array([10.0, 10.0])                                  # Output setpoint
        usp = np.array([39.5185, 38.1743])                            # Control setpoints
        xsp = np.array([50.0, 50.0, 10.0, 10.0, 2.0, 2.0])            # State setpoints
    else:
        ysp = np.array([8.0, 12.0])                                   # Output setpoint
        usp = np.array([39.5185, 38.1743])                            # Control setpoints
        xsp = np.array([40.0, 40.0, 8.0, 12.0, 2.0, 2.0])             #  State setpoints    

    return [ysp, usp, xsp]
    
# 4.2) Bounds constraints
## Input bounds
umin = np.array([0.0, 0.0])
umax = np.array([100.0, 100.0])

## State bounds
xmin = np.zeros((x.size1(),1))
xmax = np.array([100.0, 100.0, 20.0, 20.0, 20.0, 20.0])

## Output bounds
ymin = np.array([0.0, 0.0])
ymax = np.array([20.0, 20.0])

## Input rate-of-change bounds
Dumin = np.array([-50.0, -50.0])
Dumax = np.array([50.0, 50.0])

# 4.3) Steady-state optimization : objective function
Qss = np.array([[1.0, 0.0], [0.0, 1.0]])        # Output matrix
Sss = np.array([[0.0, 0.0], [0.0, 0.0]])        # Delta Control matrix


# 4.4) Dynamic optimization : objective function 
Q = np.array([[1e3, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1e3, 0.0, 0.0, 0.0, 0.0], \
              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], \
              [0.0, 0.0, 0.0, 0.0, 1e-6, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1e-6]])            
S = np.array([[10., 0.0], [0.0, 10.0]])                                          # DeltaU matrix

# Terminal weight
def User_vfin(x,xs):
    """
    SUMMARY:
    It constructs the terminal weight for the dynamic optimization problem 
    
    SYNTAX:
    assignment = User_vfin(x)
  
    ARGUMENTS:
    + x             - State variables
    
    OUTPUTS:
    + vfin          - Terminal weight     
    """ 
    Vn = 100.0
    vfin = mtimes(x.T,mtimes(Vn,x))  
    return vfin
