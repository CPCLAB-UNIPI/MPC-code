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
Nsim = 100 # Simulation length

N = 50    # Horizon

h = 1 # Time step

# 3.1.2) Symbolic variables
xp = SX.sym("xp", 4) # process state vector        
x = SX.sym("x", 4)  # model state vector          
u = SX.sym("u", 2)  # control vector              
y = SX.sym("y", 2)  # measured output vector      
d = SX.sym("d", 2)  # disturbance                 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### 2) Process and Model construction 

# 2.1) Process Parameters
Ap = np.diag([0.8871,0.8324,0.9092,0.8703])
Bp = np.array([[1,0], [1,0], [0.0, 1.], [0,2.]])
Cp = np.array([[1.4447, 0.0, -1.7169, 0.], [0.0, 1.1064, 0.0, -1.2579]])


# 2.2) Model Parameters
A = np.diag([0.8871,0.8324,0.9092,0.8703])
A2 = 2*np.diag([0.01,-0.01,-0.01,0.01])
A = A + A2
B = np.array([[1,0], [1,0], [0.0, 1.], [0,2.]])
C = np.array([[1.4447, 0.0, -1.7169, 0.], [0.0, 1.1064, 0.0, -1.2579]])

# 2.3) Disturbance model for Offset-free control
offree = "lin" 
Bd = np.zeros((x.size1(),d.size1()))
Cd = np.eye(d.size1())

# 2.4) Initial condition
x0_p = 0*np.ones((xp.size1(),1))
x0_m = 0*np.ones((x.size1(),1))
u0 = 0*np.ones((u.size1(),1))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### 3) State Estimation

# Dimensions 
nx = x.size1()  # state vector                #
nu = u.size1()  # control vector              #
ny = y.size1()  # measured output vector      #
nd = d.size1()  # disturbance                 #

# Luemberger filter tuning params 
lue = True
Kx = np.zeros((nx,nd))
Kd = np.eye(nd)
K = np.vstack((Kx,Kd))

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
    xsp = np.array([0.0, 0.0,0.0, 0.0]) # State setpoints  
    if t<= 10:
        ysp = np.array([0.0, 0.0]) # Output setpoint
        usp = np.array([0., 0.]) # Input setpoints
    else:
        ysp = np.array([1.0, -1.0]) # Output setpoint
        usp = np.array([0., 0.]) # Control setpoints
        
    return [ysp, usp, xsp]
    
# 4.2) Bounds constraints
## Input bounds
umin = -0.5*np.ones((u.size1(),1))
umax = 0.5*np.ones((u.size1(),1))

## State bounds
# xmin = np.array([-10.0, -8.0, -10.0])
# xmax = 10.0*np.ones((x.size1(),1))

## Output bounds
# ymin = np.array([-10.0, -8.0, -10.0])
# ymax = 10.0*np.ones((y.size1(),1))

# 4.3) Steady-state optimization : objective function
Qss = np.diag([1,1])
Rss = np.zeros((u.size1(),u.size1())) # Control matrix

# 4.4) Dynamic optimization : objective function 
Qy = np.diag([1,1])
Q = np.dot(C.T,np.dot(Qy,C))
S = np.diag([10,20])
