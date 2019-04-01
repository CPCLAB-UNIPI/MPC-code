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
from Target_Calc import *
from Estimator import *
from Control_Calc import *
from Default_Values import *
import pdt_otto_mol as pdt

###############################################################################
###             Example file for Williams-Otto reactor                      ###
###############################################################################

pathfigure = './Images/123eosc/'

# NOC discretization parameters
Nsim = 70 # Simulation length
N = 25    # Horizon

# Time interval value
h = 2. # if you are in discrete time just put h = 1.0

########## Symbolic variables #####################
xp = SX.sym("xp", 6) # process state vector       #
x = SX.sym("x", 6)  # model state vector          #
u = SX.sym("u", 2)  # control vector              #
y = SX.sym("y", 2)  # measured output vector      #
d = SX.sym("d", 2)  # disturbance                 #
###################################################

nx = x.size1()
ny = y.size1()
nu = u.size1()
 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
## PLANT

#StateFeedback = True

ki = lambda Eai , ki0 , Tr : ki0*exp(Eai/(273.15+Tr))

# To set if the plant is non-linear

#Fp_nominal = True

def User_fxp_Cont(xp,t,u):
    """
    SUMMARY:
    It constructs the function fx_p for the non-linear case
    
    SYNTAX:
    assignment = User_fxp_Cont(xp,t,u)
  
    ARGUMENTS:
    + t         - Variable that indicate the current iteration
    
    OUTPUTS:
    + fx_p      - Non-linear plant function     
    """ 
     # Manipulated variables
     
    Qb0 = u[0]
    Tr = u[1]
        
    # Definitions
    
    Ca=xp[0]
    Cb=xp[1]
    Cc=xp[2]
    Ce=xp[3]
    Cg=xp[4]
    Cp=xp[5]
    
    Qr=pdt.Qa0+Qb0 
    
    # Reaction rates
    r1=Ca*Cb*ki(pdt.Ea1,pdt.k10mol,Tr)
    r2=Cb*Cc*ki(pdt.Ea2,pdt.k20mol,Tr)
    r3=Cc*Cp*ki(pdt.Ea3,pdt.k30mol,Tr)
      
    fx_p = vertcat(pdt.Qa0*pdt.Ca0/pdt.Vr-Qr*Ca/pdt.Vr - r1, \
                   Qb0*pdt.Cb0/pdt.Vr - Qr*Cb/pdt.Vr - r1 - r2, \
                   -Qr*Cc/pdt.Vr +  r1 - r2 - r3, \
                   -Qr*Ce/pdt.Vr +  r2, \
                   -Qr*Cg/pdt.Vr +  r3, \
                   -Qr*Cp/pdt.Vr +  r2 - r3) 
    return fx_p

def User_fyp(xp,t):
    """
    SUMMARY:
    It constructs the function fy_p for the non-linear case
    
    SYNTAX:
    assignment = User_fyp(xp,t,u)
  
    ARGUMENTS:
    + t             - Variable that indicate the current iteration
    
    OUTPUTS:
    + fy_p      - Non-linear plant function    
    """     
    fy_p = vertcat(xp[3], \
                   xp[5] )
    return fy_p
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
## Disturbance model
    
offree = "lin" # set "lin" or "nl" to have a disturbance model linear or non linear. "no" means no disturbance model will be implmented

#Bd = DM.zeros(x.size1(),d.size1())
#Cd = DM.eye(d.size1())

#
#Bd = DM([[1.,0.],\
#         [0.,1.],\
#         [1.,0.],\
#         [0.,1.],\
#         [1.,0.],\
#         [0.,1.]])

Bd = DM([[0.,1.],\
         [1.,0.],\
         [0.,1.],\
         [1.,0.],\
         [0.,1.],\
         [1.,0.]])
    
Cd = DM.zeros(d.size1(),d.size1())

                      
def User_fxm_Cont(x,u,d,t):
    """
    SUMMARY:
    It constructs the function fx_model for the non-linear case
    
    SYNTAX:
    assignment = fx_model(x,u,d,t)
  
    ARGUMENTS:
    + x,u,d         - State, input and disturbance variable
    + t             - Variable that indicate the real time
    
    OUTPUTS:
    + x_model       - Non-linear model function     
    """
    err1=1.4
    err2=1.2
    err3=0.
    
    Qb0 = u[0]
    Tr = u[1]
        
    # Definitions
    
    Ca=x[0]
    Cb=x[1]
    Cc=x[2]
    Ce=x[3]
    Cg=x[4]
    Cp=x[5]
    
    Qr=pdt.Qa0+Qb0 
        # Reaction rates
    r1=Ca*Cb*ki(pdt.Ea1,pdt.k10mol,Tr)*err1
    r2=Cb*Cc*ki(pdt.Ea2,pdt.k20mol,Tr)*err2
    r3=Cc*Cp*ki(pdt.Ea3,pdt.k30mol,Tr)*err3

    x_model = vertcat( pdt.Qa0*pdt.Ca0/pdt.Vr-Qr*Ca/pdt.Vr- r1, \
                   Qb0*pdt.Cb0/pdt.Vr-Qr*Cb/pdt.Vr- r1- r2, \
                   -Qr*Cc/pdt.Vr +  r1- r2- r3, \
                   -Qr*Ce/pdt.Vr +  r2, \
                   -Qr*Cg/pdt.Vr +   r3, \
                   -Qr*Cp/pdt.Vr + r2 -r3)
                    
    return x_model
   
def User_fym(x,d,t):
    """
    SUMMARY:
    It constructs the function fy_m for the non-linear case
    
    SYNTAX:
    assignment = User_fym(xp,t,u)
  
    ARGUMENTS:
    + t             - Variable that indicate the current iteration
    
    OUTPUTS:
    + fy_m      - Non-linear model function     
    """ 
    fy_m = vertcat(x[3], \
                   x[5])                    
    return fy_m


Mx = 10 # Number of elements in each time step 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Initial conditions

#Qb0=(pdt.QbL+pdt.QbU)/2
Qb0= pdt.QbU 

Qr0=Qb0+pdt.Qa0
Cain=(pdt.Ca0*pdt.Qa0)/(Qr0)
Cbin=(pdt.Cb0*Qb0)/(Qr0)



x0_p = vertcat(Cain , Cbin, 0., 0., 0., 0.) # plant 
x0_m = vertcat(Cain , Cbin, 0., 0., 0., 0.) # model
u0 = vertcat(360, 75)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
## BOUNDS
## Input bounds

umin = vertcat(pdt.QbL , pdt.TrL)
umax = vertcat(pdt.QbU, pdt.TrU)

## State bounds

xmin = vertcat(0. , 0., 0., 0., 0., 0.)
xmax = vertcat(Cbin , Cbin, Cbin, Cbin, Cbin, Cbin)
#xmax = vertcat(10. , DM.inf(1), DM.inf(1), DM.inf(1), DM.inf(1), DM.inf(1))

## Output bounds

#ymin = vertcat(0., 60.)
#ymax = vertcat(5., 90.)  
  
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
## OBJECTIVE FUNCTION

## Steady-state optimization 

    # To set if the objective function is neither quadratic nor linear
    
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
    
    Qb0 = u[0]
    Tr = u[1]
        
    # Definitions      
    Ce=y[0]
    Cp=y[1]
  
    Qr=pdt.Qa0+Qb0 
     
    obj=-(Qr*(Cp*pdt.Ppmol+Ce*pdt.Pemol)\
        -pdt.Qa0*pdt.Ca0*pdt.Pamol-Qb0*pdt.Cb0*pdt.Pbmol)/60  
 
    return obj

Adaptation = True

## Dynamic optimization 
#
#Q = DM.eye(nx)
#R = 1.0*DM.eye(nu)

 # To set if the objective function is neither quadratic nor linear
 
#ContForm = True

def User_fobj_Dis(x,u,y,xs,us,ys):
#def User_fobj_Cont(x,u,y,xs,us,ys):

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
#    Q = 1.0*DM.eye(ny)  

    S = 1.0e-2*DM.eye(nu)  
    
    Qb0 = u[0]
    Tr = u[1]
     
    # Definitions
       
    Ce=y[0]
    Cp=y[1]
    
    Qr=pdt.Qa0+Qb0 
    
    obj_eco=-(Qr*(Cp*pdt.Ppmol+Ce*pdt.Pemol)\
        -pdt.Qa0*pdt.Ca0*pdt.Pamol-Qb0*pdt.Cb0*pdt.Pbmol)/60
       
    obj_du = mtimes((us).T,mtimes(S,(us)))# + mtimes((y-ys).T,mtimes(Q,(y-ys))) #

    obj = obj_eco + obj_du
 
    return obj

DUFormEcon = True

alpha_mod = 0.1

#ContForm = True
#QForm = True
    
#TermCons = True
    # Specify the terminal cost if the problem is not QP and 0 value is not wanted
#vfin = mtimes(x.T,mtimes(2000,x)) 


#############################################################################
#### Extended Kalman filter tuning params ###################################
    
## Terminal weight

def User_vfin(x):
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
    vfin = mtimes(x.T,mtimes(2000,x)) 
    return vfin




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
## Estimator

#############################################################################
#### Luemberger observer tuning params ######################################
#lue = True # Set True if you want the Luemberger observer
#nx = x.size1()
#ny = y.size1()
#nd = d.size1()
#Kx = DM.zeros(nx,ny)
#Kd = DM.ones(nd,ny)
#K = vertcat(Kx,Kd)

#K = DM([[0.0,0.0],[0.0,0.0],[1.0,0.0],[0.0,1.0]]) 
#############################################################################
#### Steady-state Kalman filter tuning params #################################

    
    
#mhe_mod="off"
mhe_mod="on"

mhe_up = 'smooth'
#mhe_up = 'filter'

N_mhe=20

if mhe_mod == 'off':
    #### Extended Kalman filter tuning params ###################################
    ekf = True # Set True if you want the Kalman filter
    nx = x.size1()
    ny = y.size1()
    nd = d.size1()
#    Qx_kf = 1.0e-8*DM.eye(nx)
#    Qd_kf = 1.0*DM.eye(nd)
#    Q_kf = DM(scla.block_diag(Qx_kf, Qd_kf))
#    R_kf = 1.0e-8*DM.eye(ny)
#    P0 = 1.0e-8*DM.eye(nx+nd)  
#    nameoutputfile = 'ekf'
    Qx_kf = 1.0e-4*DM.eye(nx)
    Qd_kf = 1.0*DM.eye(nd)
    Q_kf = DM(scla.block_diag(Qx_kf, Qd_kf))
    R_kf = 1.0e-4*DM.eye(ny)
    P0 = 1.0e-4*DM.eye(nx+nd)  
    nameoutputfile = 'ekf'
else:
    
    #### Moving Horizon Estimation params ###################################
    mhe = True # Set True if you want the MHE
    nx = x.size1()
    ny = y.size1()
    nd = d.size1()
    P0 = 1.0*DM.eye(nx+nd)#,nx+nd)  # (it will be nx+nd,nx+nd when d will be include)
#    P0 = 1.0*DM.eye(nx+nd,nx+nd)  # (it will be nx+nd,nx+nd when d will be include)

    x_bar = vertcat(x0_m,DM.zeros(nd))
    
    w = SX.sym("w", nx+nd)  # state noise  
    
    if mhe_up == 'filter':
        nameoutputfile = 'mhe_filter'
    else:
        nameoutputfile = 'mhe_smooth'
#    
#    
    def User_fx_mhe_Cont(x,u,d,t,w):
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
    
        err1=1.4
        err2=1.2
        err3=0.
        
        Qb0 = u[0]
        Tr = u[1]
            
        # Definitions
        
        Ca=x[0]
        Cb=x[1]
        Cc=x[2]
        Ce=x[3]
        Cg=x[4]
        Cp=x[5]
        
        Qr=pdt.Qa0+Qb0 
            # Reaction rates
        r1=Ca*Cb*ki(pdt.Ea1,pdt.k10mol,Tr)*err1
        r2=Cb*Cc*ki(pdt.Ea2,pdt.k20mol,Tr)*err2
        r3=Cc*Cp*ki(pdt.Ea3,pdt.k30mol,Tr)*err3
    
        x_model = vertcat( pdt.Qa0*pdt.Ca0/pdt.Vr-Qr*Ca/pdt.Vr- r1, \
                       Qb0*pdt.Cb0/pdt.Vr-Qr*Cb/pdt.Vr- r1- r2, \
                       -Qr*Cc/pdt.Vr +  r1- r2- r3, \
                       -Qr*Ce/pdt.Vr +  r2, \
                       -Qr*Cg/pdt.Vr +   r3, \
                       -Qr*Cp/pdt.Vr + r2 -r3)            
        return x_model
        
    def User_fobj_mhe(w,v,t):
        """
        SUMMARY:
        It constructs the fobjective unction for the MHE optimization problem
        
        SYNTAX:
        assignment = User_fxm_Cont(x,u,d,t)
      
        ARGUMENTS:
        + w             - State noise
        + v             - Output noise
        
        OUTPUTS:
        + fobj_mhe      - Objective function     
        """ 
        Qx = 1.0*DM.eye(nx)
        Qd = 1.0*DM.eye(nd)
        Q= DM(scla.block_diag(Qx, Qd))
        R = 1.0e0*DM.eye(ny)
        
        fobj_mhe = 0.5*(xQx(w,inv(Q))+xQx(v,inv(R)))
        
        return fobj_mhe
    
    
Sol_itmax = 300
