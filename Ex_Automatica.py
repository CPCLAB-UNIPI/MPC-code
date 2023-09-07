# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:11:54 2016

@author: marcovaccari
"""
from casadi import *
from casadi.tools import *
from matplotlib import pylab as plt
import math
import scipy.linalg as scla
import numpy as np
from Utilities import*

### 1) Simulation Fundamentals

# 1.1) Simulation discretization parameters
Nsim = 60*24 # Simulation length

N = 50       # Horizon

h = 1./60    # Time step (hr)

# 3.1.2) Symbolic variables
xp = SX.sym("xp", 20) # process state vector        
x = SX.sym("x", 20)  # model state vector          
u = SX.sym("u", 6)  # control vector              
y = SX.sym("y", 5)  # measured output vector      
d = SX.sym("d", 0)  # disturbance                 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### 2) Process and Model construction 
from scipy.io import loadmat
MM = loadmat("DTmatrices.mat")

# 2.1) Process Parameters
Ap = MM["Ad"]
Bp = MM["Bd"]
Cp = MM["Cd"]

# Additive State Disturbances 
#def defdxp(t):
#    """
#    SUMMARY:
#    It constructs the additive disturbances for the linear case
#    
#    SYNTAX:
#    assignment = defdp(k)
#  
#    ARGUMENTS:
#    + t             - Variable that indicate the current time
#    
#    OUTPUTS:
#    + dxp      - State disturbance value      
#    """ 
#
#    if t <= 20:
#        dxp = np.array([0.1, 0.0, 0.0]) # State disturbance
#    else:
#        dxp = np.array([0.0, 0.0, 0.0]) # State disturbance
#
#    return [dxp]

#def defdyp(t):
#    """
#    SUMMARY:
#    It constructs the additive disturbances for the linear case
#    
#    SYNTAX:
#    assignment = defdp(k)
#  
#    ARGUMENTS:
#    + t             - Variable that indicate the current time
#    
#    OUTPUTS:
#    + dyp      - Output disturbance value      
#    """ 
#    
#    dyp = np.array([0.1, 0.1, 0.0]) # Output disturbance
#    
#    return [dyp]


# 2.2) Model Parameters
A = Ap
B = Bp
C = Cp

# 2.3) Disturbance model for Offset-free control
offree = "no" 
#Bd = np.eye(d.size1())
#Cd = np.zeros((y.size1(),d.size1()))

# 2.4) Initial condition
x0_p = 0*np.ones((xp.size1(),1))
x0_m = 0*np.ones((x.size1(),1))
u0 = 0*np.ones((u.size1(),1))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### 3) State Estimation
 
# Kalman filter tuning params 
kalss = True # Set True if you want the Kalman filter
########## Variables dimensions ###############
nx = x.size1()  # state vector                #
nu = u.size1()  # control vector              #
ny = y.size1()  # measured output vector      #
nd = d.size1()  # disturbance                 #
###############################################
#Qx_kf = 1.0e-7*np.eye(nx)
#Qd_kf = np.eye(nd)
Q_kf = 1.0e-5*np.eye(nx)
R_kf = 1.0e-5*np.eye(ny)
x_ss = np.zeros(nx)
u_ss = np.zeros(nu)
#P0 = 1.0e-8*np.eye(nx+nd) 

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
    + t             - Variable that indicates the current time
    
    OUTPUTS:
    + ysp, usp, xsp - Input, output and state setpoint values      
    """ 
    xsp = np.zeros(nx) # State setpoints  
    if t<=0.5:
        ysp = np.zeros(ny)# Output setpoint
        usp = np.zeros(nu) # Input setpoints
    else:
        ysp = np.array([0., 0., 1., 0., 0.]) # Output setpoint
        usp = np.zeros(nu)  # Control setpoints
        
    return [ysp, usp, xsp]
    
# 4.2) Bounds constraints
## Input bounds
umin = -100.0*np.ones((u.size1(),1))
umax = 100.0*np.ones((u.size1(),1))

## State bounds
#xmin = np.array([-10.0, -8.0, -10.0])
#xmax = 10.0*np.ones((x.size1(),1))

## Output bounds
#ymin = np.array([-10.0, -8.0, -10.0])
#ymax = 10.0*np.ones((y.size1(),1))

# 4.3) Steady-state optimization : objective function
Qss = np.eye(ny) #Output matrix
Rss = 0.001*np.diag([1/0.2**2, 1/0.35**2, 1/0.1**2, 1./3**2, 1/65**2, 1/25**2]) # Control matrix

# 4.4) Dynamic optimization : objective function 
Q = np.dot(C.T, C)
Q = Q + 1e-3*np.linalg.norm(Q,ord=2)*np.eye(nx)
R = np.diag([1/0.2**2, 1/0.35**2, 1/0.1**2, 1./3**2, 1/65**2, 1/25**2])
