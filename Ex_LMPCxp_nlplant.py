# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:11:54 2016

@author: marcovaccari
"""
from __future__ import division
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
Nsim = 200 # Simulation length

N = 50    # Horizon

h = 0.2 # Time step

# 3.1.2) Symbolic variables
xp = SX.sym("xp", 3) # process state vector       
x = SX.sym("x", 4)  # model state vector          
u = SX.sym("u", 2)  # control vector              
y = SX.sym("y", 2)  # measured output vector      
d = SX.sym("d", 2)  # disturbance                       

nx = x.size1()
ny = y.size1()
nd = d.size1()
nu = u.size1()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### 2) Process and Model construction 

# 2.1) Process Parameters

# State map
def User_fxp_Cont(x,t,u,pxp,pxmp):
    """
    SUMMARY:
    It constructs the function fx_p for the non-linear case
        
    SYNTAX:
    assignment = User_fxp_Cont(x,t,u)
        
    ARGUMENTS:
    + x         - State variable     
    + t         - Current time
    + u         - Input variable  
        
    OUTPUTS:
    + fx_p      - Non-linear plant function
    """ 
    F0 = 0.1  # m^3/min
    T0 = 350  # K
    c0 = 1.0  # kmol/m^3
    r = 0.219 # m
    k0 = 7.2e10 # min^-1
    EoR = 8750 # K
    U0 = 915.6*60/1000  # kJ/min*m^2*K
    rho = 1000.0 # kg/m^3
    Cp2 = 0.239 # kJ/kg
    DH = -5.0e4 # kJ/kmol
    pi = math.pi
    kT0 = k0*exp(old_div(-EoR,T0))
    
    fx_p = vertcat\
    (\
    F0*(c0 - x[0])/(pi* r**2 *x[2]) - kT0*exp(-EoR*(old_div(1.0,x[1])-old_div(1.0,T0)))*x[0], \
    F0*(T0 - x[1])/(pi* (r**2) *x[2]) -DH/(rho*Cp2)*kT0*exp(-EoR*(old_div(1.0,x[1])-old_div(1.0,T0)))*x[0] + 2*U0/(r*rho*Cp2)*(u[0] - x[1]), \
    old_div((F0 - u[1]),(pi*r**2)) \
    )
    
    return fx_p

Mx = 10 # Number of elements in each time step 

# Output map
Cp = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


# 2.2) Model Parameters
Alin = np.array([[0.51448, -0.00917517, -0.117995],[53.6817, 2.15004, -3.77725], [0.0, 0.0, 1]])
Blin = np.array([[-0.0017669, 0.0864569], [0.639423, 1.60696],  [0.0, -1.32737]])
Clin = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

## Extra state (to test different plant/model size)
Phi = 0.01
B_Phi = np.array([[1.0 - Phi, 0.0]])
C_Phi = (old_div(Phi,10.0))*np.array([[1.0],[0.0]])

A = scla.block_diag(Alin, Phi)
B = np.row_stack([Blin,B_Phi])
C = np.column_stack([Clin, C_Phi])

# Linearisation parameters
xlin = np.array([0.5, 350, 0.659, 0.0])
ulin = np.array([300, 0.1])
ylin = np.array([0.5, 0.659]) 

# 2.3) Disturbance model for Offset-free control
offree = "lin" 
Bd = B
Cd = np.zeros((ny,nd))

# 2.4) Initial condition
x0_p = np.array([0.5, 350, 0.659]) # plant
x0_m = np.array([0.5, 350, 0.659, 0.0]) # plant
u0 = np.array([300, 0.1])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### 3) State Estimation
 
# Kalman filter tuning params 
kal = True # Set True if you want the Kalman filter
Qx_kf = 1.0e-2*np.eye(nx)
Qd_kf = np.eye(nd)
Q_kf = scla.block_diag(Qx_kf, Qd_kf)
R_kf = 1.0e-2*np.eye(ny)
P0 = Q_kf  

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### 4) Steady-state and dynamic optimizers

# 4.1) Setpoints
def defSP(t):
    """
    SUMMARY:
    It constructs the setpoints vectors for the steady-state optimization 
    
    SYNTAX:
    assignment = defSP(t)
  
    ARGUMENTS:
    + t             - Variable that indicates the current time
    
    OUTPUTS:
    + ysp, usp, xsp - Input, output and state setpoint values      
    """ 
    xsp = np.array([0.0, 0.0, 0.0, 0.0]) # State setpoints    
    usp = np.array([300., 0.1]) # Control setpoints
    
    if t < 20:
        ysp = np.array([0.5, 0.659]) # Output setpoint
    else:
        ysp = np.array([0.51, 0.659]) # Output setpoint
     
    return [ysp, usp, xsp]
    
# 4.2) Bounds constraints
## Input bounds
umin = np.array([295, 0.00])
umax = np.array([305, 0.25])

## State bounds
xmin = np.array([0.0, 300, 0.45, -1.0])
xmax = np.array([1.0, 375, 0.75, 1.0])

## Output bounds
ymin = np.array([0.0, 0.0])
ymax = np.array([1.0, 1.0])

# 4.3) Steady-state optimization : objective function
Qss = np.array([[1.0,  0.0],  [0.0, 1.0]]) #Output matrix
Rss = np.array([[0.0, 0.0], [0.0, 0.0]]) # Control matrix

# 4.4) Dynamic optimization : objective function 
Q = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, .1]])
S = 0.10*np.eye(nu) # DeltaU matrix