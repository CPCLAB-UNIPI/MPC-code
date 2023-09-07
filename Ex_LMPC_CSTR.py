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
xp = SX.sym("xp", 3) # process state vector
x = SX.sym("x", 3)  # model state vector
u = SX.sym("u", 2)  # control vector
y = SX.sym("y", 3)  # measured output vector
d = SX.sym("d", 3)  # disturbance

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### 2) Process and Model construction

# 2.1) Process Parameters
Ap = np.array([[0.2511, -3.368*1e-03, -7.056*1e-04], [11.06, .3296, -2.545], [0.0, 0.0, 1.0]])
Bp = np.array([[-5.426*1e-03, 1.53*1e-05], [1.297, .1218], [0.0, -6.592*1e-02]])
Cp = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],[0.0, 0.0, 1.0]])

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

    if t <= 20:
        dxp = np.array([0.1, 0.0, 0.0]) # State disturbance
    else:
        dxp = np.array([0.0, 0.0, 0.0]) # State disturbance

    return [dxp]

def def_pyp(t):
    """
    SUMMARY:
    It constructs the additive disturbances for the linear case

    SYNTAX:
    assignment = defdp(k)

    ARGUMENTS:
    + t             - Variable that indicate the current time

    OUTPUTS:
    + dyp      - Output disturbance value
    """

    dyp = np.array([0.1, 0.1, 0.0]) # Output disturbance

    return [dyp]


# 2.2) Model Parameters
A = np.array([[0.2511, -3.368*1e-03, -7.056*1e-04], [11.06, .3296, -2.545], [0.0, 0.0, 1.0]])
B = np.array([[-5.426*1e-03, 1.53*1e-05], [1.297, .1218], [0.0, -6.592*1e-02]])
C = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],[0.0, 0.0, 1.0]])

# 2.3) Disturbance model for Offset-free control
offree = "lin"
Bd = np.eye(d.size1())
Cd = np.zeros((y.size1(),d.size1()))

# 2.4) Initial condition
x0_p = 3*np.ones((xp.size1(),1))
x0_m = 3*np.ones((x.size1(),1))
u0 = 0*np.ones((u.size1(),1))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### 3) State Estimation

# Kalman filter tuning params
kal = True # Set True if you want the Kalman filter
########## Variables dimensions ###############
nx = x.size1()  # state vector                #
nu = u.size1()  # control vector              #
ny = y.size1()  # measured output vector      #
nd = d.size1()  # disturbance                 #
###############################################
Qx_kf = 1.0e-7*np.eye(nx)
Qd_kf = np.eye(nd)
Q_kf = scla.block_diag(Qx_kf, Qd_kf)
R_kf = 1.0e-7*np.eye(ny)
P0 = 1.0e-8*np.eye(nx+nd)

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
    xsp = np.array([0.0, 0.0, 0.0]) # State setpoints
    if t<= 15:
        ysp = np.array([0.2, 0.0, 0.0]) # Output setpoint
        usp = np.array([0., 0.]) # Input setpoints
    else:
        ysp = np.array([0.0, 0.0, 0.1]) # Output setpoint
        usp = np.array([0., 0.]) # Control setpoints

    return [ysp, usp, xsp]

# 4.2) Bounds constraints
## Input bounds
umin = -10.0*np.ones((u.size1(),1))
umax = 10.0*np.ones((u.size1(),1))

## State bounds
xmin = np.array([-10.0, -8.0, -10.0])
xmax = 10.0*np.ones((x.size1(),1))

## Output bounds
ymin = np.array([-10.0, -8.0, -10.0])
ymax = 10.0*np.ones(y.size1())

# 4.3) Steady-state optimization : objective function
Qss = np.array([[20.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) #Output matrix
Rss = np.zeros((u.size1(),u.size1())) # Control matrix

# 4.4) Dynamic optimization : objective function
Q = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
R = 0.1*np.eye(u.size1())
