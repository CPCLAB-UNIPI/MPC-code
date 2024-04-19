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
Nsim = 21 # Simulation length

N = 25    # Horizon

h = 2.0 # Time step

# 3.1.2) Symbolic variables
xp = SX.sym("xp", 2) # process state vector       
x = SX.sym("x", 2)  # model state vector          
u = SX.sym("u", 1)  # control vector              
y = SX.sym("y", 2)  # measured output vector      
d = SX.sym("d", 2)  # disturbance      

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### 2) Process and Model construction 
StateFeedback = True # Set to True if you have all the states measured 

cA0 = 1.0  # kmol/m^3 
V = 1.0 # m^3 ????
k1 = 1. # min^-1
k2 = 0.05 # min^-1
alfa = 1. # reactant price
beta = 4. # product price

# 2.1) Process Parameters

# State map
def User_fxp_Cont(xp,t,u,pxp,pxmp):
    """
    SUMMARY:
    It constructs the function User_fxp_Cont for the non-linear case
    
    SYNTAX:
    assignment = User_fxp_Cont(xp,t,u)
  
    ARGUMENTS:
    + xp,u          - Process State and input variable    
    + t             - Variable that indicate the current iteration
    
    OUTPUTS:
    + fx_p      - Non-linear plant function     
    """ 
    fx_p = vertcat(u[0]*(cA0 - xp[0])/V - k1*xp[0], \
                           -u[0]*xp[1]/V + k1*xp[0] - k2*xp[1])    
    return fx_p

Mx = 10 # Number of elements in each time step 


# # White Noise
# G_wn = 1e-2*np.array([[1.0, 0.0], [0.0, 1.0]]) # State white noise matrix
# Q_wn = 1e-1*np.array([[1.0, 0.0], [0.0, 1.0]]) # State white noise covariance matrix

# 2.2) Model Parameters
    
# State Map
def User_fxm_Cont(x,u,d,t,px):
    """
    SUMMARY:
    It constructs the function fx_model for the non-linear case
    
    SYNTAX:
    assignment = User_fxm_Cont(x,u,d,t)
  
    ARGUMENTS:
    + x,u,d         - State, input and disturbance variable
    + t             - Variable that indicate the real time
    
    OUTPUTS:
    + x_model       - Non-linear model function     
    """     
    x_model = vertcat(u[0]*(cA0 - x[0])/V - k1*x[0], \
                       -u[0]*x[1]/V + k1*x[0] - k2*x[1])
    return x_model


    
Mx = 10 # Number of elements in each time step 

# 2.3) Disturbance model for Offset-free control
offree = "lin" 
Bd = np.zeros((d.size1(),d.size1()))
Cd = np.eye(d.size1())

# 2.4) Initial condition
x0_p = np.array([0.9, 0.1]) # plant
x0_m = np.array([1.2, 0.5]) # model
u0 = np.array([0.])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### 3) State Estimation
mhe_mod = 'on' # Decide to use MHE ('on') as estimator or EKF ('off') 
 
if mhe_mod == 'off':
    #### Extended Kalman filter tuning params ###################################
    ekf = True # Set True if you want the Kalman filter
    nx = x.size1()
    ny = y.size1()
    nd = d.size1()
    Qx_kf = 1.0e-8*np.eye(nx)
    Qd_kf = 1.0*np.eye(nd)
    Q_kf = scla.block_diag(Qx_kf, Qd_kf)
    R_kf = 1.0e-8*np.eye(ny)
    P0 = 1.0e-8*np.eye(nx+nd)  
else:
    
    #### Moving Horizon Estimation params ###################################
    mhe = True # Set True if you want the MHE
    N_mhe = 10 # Horizon lenght
    mhe_up = 'smooth' # Updating method for prior weighting and x_bar
    nx = x.size1()
    ny = y.size1()
    nd = d.size1()
    w = SX.sym("w", nx+nd)  # state noise  
    P0 = np.eye(nx+nd)
    x_bar = np.row_stack((np.atleast_2d(x0_m).T,np.zeros((nd,1))))
    
    # Defining the state map
    def User_fx_mhe_Cont(x,u,d,t,px,w):
        """
        SUMMARY:
        It constructs the function fx_model for the non-linear case in MHE problem
        
        SYNTAX:
        assignment = User_fx_mhe_Cont(x,u,d,t,w)
      
        ARGUMENTS:
        + x,u,d         - State, input and disturbance variable
        + t             - Variable that indicate the real time
        + w             - Process noise variable 
        
        OUTPUTS:
        + x_model       - Non-linear model function     
        """     
        x_model = vertcat(u[0]*(cA0 - x[0])/V - k1*x[0], \
                       -u[0]*x[1]/V + k1*x[0] - k2*x[1])
        
        return x_model
    
    # Defining the MHE cost function
    def User_fobj_mhe(w,v,t):
        """
        SUMMARY:
        It constructs the objective function for MHE problem
        
        SYNTAX:
        assignment = User_fobj_mhe(w,v,t)
      
        ARGUMENTS:
        + w,v           - Process and Measurement noise variables
        + t             - Variable that indicate the real time
        
        OUTPUTS:
        + fobj_mhe      - Objective function      
        """ 
        Q = np.eye(nx+nd)
        R = np.eye(ny)
        fobj_mhe = 0.5*(xQx(w,inv(Q))+xQx(v,inv(R)))
        
        return fobj_mhe

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### 4) Steady-state and dynamic optimizers

# 4.1) Setpoints
# As this is an economic MPC example, setpoints are not useful,
# so there is no need to define them.
    
# 4.2) Bounds constraints
## Input bounds
umin = [0.00]
umax = [2.0]

## State bounds
xmin = np.array([0.00, 0.00])
xmax = np.array([1.00, 1.00])

# 4.3) Steady-state optimization : objective function
def User_fssobj(x,u,y,xsp,usp,ysp):
    """
    SUMMARY:
    It constructs the objective function for steady-state optimization 
    
    SYNTAX:
    assignment = User_fssobj(x,u,y,xsp,usp,ysp)
  
    ARGUMENTS:
    + x,u,y         - State, input and output variables
    + xsp,usp,ysp   - State, input and output setpoint variables
    
    OUTPUTS:
    + obj         - Objective function      
    """ 

    obj = u[0]*(alfa*cA0 - beta*y[1])
    
    return obj

# 4.4) Dynamic optimization : objective function 
def User_fobj_Cont(x,u,y,xs,us,ys):
    """
    SUMMARY:
    It constructs the objective function for dynamic optimization 
    
    SYNTAX:
    assignment = User_fobj_Cont(x,u,y,xs,us,ys)
  
    ARGUMENTS:
    + x,u,y         - State, input and output variables
    + xs,us,ys      - State, input and output stationary variables
    
    OUTPUTS:
    + obj         - Objective function      
    """ 
    obj = u[0]*(alfa*cA0 - beta*y[1]) 
    return obj

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
    diffx = x - xs 
    vfin = mtimes(diffx.T,mtimes(2000,diffx)) 
    return vfin

# Options
Sol_itmax = 200


# S_cust = 20

# R_cust = 10

# Q_cust = 5*np.eye((x.size1()))
# K_cust = 5*np.eye((x.size1())) #weight of adding variables for collocation methods


# b1 = 0.5; b2 = 0.5 


# def User_fobj_Dis(x,u,y,xs,us,ys):
#     """
#     SUMMARY:
#     It constructs the objective function for dynamic optimization

 

#     SYNTAX:
#     assignment = User_fobj_Dis(x,u,y,xs,us,ys)

 

#     ARGUMENTS:
#     + x,u,y         - State, input and output variables
#     + xs,us,ys      - State, input and output stationary variables

 

#     OUTPUTS:
#     + obj         - Objective function
#     """
    
#     obj = 0.5*(xQx(x,Q_cust) + xQx(u,R_cust) + xQx(us,S_cust))


#     return obj

    

# def User_fobj_Coll(x,u,y,xs,us,ys,s_Coll):
#     """
#     SUMMARY:
#     It constructs the objective function for dynamic optimization

 

#     SYNTAX:
#     assignment = User_fobj_Coll(x,u,y,xs,us,ys,s_Coll)

 

#     ARGUMENTS:
#     + x,u,y         - State, input and output variables
#     + xs,us,ys      - State, input and output stationary variables
#     + s_Coll        - Internal state variables

 

#     OUTPUTS:
#     + obj         - Objective function
#     """    
#     s_Coll1 = s_Coll[0:x.size1()]
#     s_Coll2 = s_Coll[x.size1():2*x.size1()]
    
#     # obj = 0.5*h*(b1*(xQx((s_Coll1-xs), K_cust) + xQx(u,R_cust) + xQx(us,S_cust)) + 
#     #               b2*(xQx((s_Coll2-xs), K_cust) + xQx(u,R_cust) + xQx(us,S_cust)))
#     # obj = 0.5*(xQx(x,Q_cust) + xQx(u,R_cust) + xQx(us,S_cust))
#     obj = u[0]*(alfa*cA0 - beta*y[1]) 

#     return obj    

# DUFormEcon = True
# QForm = True
# Collocation = True
ContForm = True

