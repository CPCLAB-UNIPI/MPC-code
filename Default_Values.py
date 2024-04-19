# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:33:23 2016

@author: marcovaccari
"""

from casadi import *
from casadi.tools import *
from matplotlib import pylab as plt
import math
import scipy.linalg as scla
import numpy as np

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
estimating = False # Set to True if you want to do only the estimation problem
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
ssjacid = False # Set to True if you want a linearization of the process in ss point
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
StateFeedback = False # Set to True if you have all the states measured 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
Fp_nominal = False # Define the nominal case: fp = fx_model e hp = fy_model
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
offree = 'no' # Set to 'lin'/'nl' to have a linear/non linear disturbance model
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

## BOUND CONSTRAINT

# Input bounds
umin = None
umax = None

# State bounds
xmin = None
xmax = None

# Output bounds
ymin = None
ymax = None

# Input bounds on the target problem
umin_ss = None
umax_ss = None

# State bounds on the target problem
xmin_ss = None
xmax_ss = None

# Output bounds on the target problem
ymin_ss = None
ymax_ss = None

# Input bounds on the dynamic problem
umin_dyn = None
umax_dyn = None

# State bounds on the dynamic problem
xmin_dyn = None
xmax_dyn = None

# Output bounds on the dynamic problem
ymin_dyn = None
ymax_dyn = None

# Disturbance bounds
dmin = None
dmax = None

# DeltaInput bounds
Dumin = None
Dumax = None

# State noise bounds
wmin = None
wmax = None

# Ouput noise bounds
vmin = None
vmax = None

## OBJECTIVE FUNCTION

## Steady-state optimization 
QForm_ss = False # Set true if you have setpoint and you want y-y_sp and u-u_sp as optimization variables

DUssForm = False # Set true if you want DUss = u_s[k]-u_s[k-1] as optimization variables rather than u_s[k]

Adaptation = False # Set true to have the modifiers-adaption method

## Dynamic optimization 
ContForm = False # Set true if you want to integrate the objective function in continuous form

TermCons = False # Set true if you want the terminal constraint

QForm = False # Set true if you want x[k]-xs and u[k]-us as optimization variables

DUForm = False # Set true if you want DU = u[k]-u[k-1] as optimization variables rather than u[k]

DUFormEcon = False # Set true if you want DU = u[k]-u[k-1] as optimization variables added to u[k] in economic MPC

## Solver
Sol_itmax = 100
Sol_Hess_constss = 'no' # Set True if you want to calculate Hessian matrices every iteration
Sol_Hess_constdyn = 'no' # Set True if you want to calculate Hessian matrices every iteration
Sol_Hess_constmhe = 'no'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
## Estimator
#### Steady-state Kalman filter tuning params #################################
kalss = False # Set True if you want the Steady-state Kalman filter

#### Luemberger observer tuning params ######################################
lue = False # Set True if you want the Luemberger observer

#### Kalman filter tuning params  #############################################
kal = False # Set True if you want the Kalman filter

#############################################################################
#### Extended Kalman filter tuning params ###################################
ekf = False # Set True if you want the Kalman filter
 
#### Moving Horizon Estimation params ###################################
mhe = False # Set True if you want the MHE

Collocation = False # Set True if want the Collocation Method

LinPar = True

slacks = False

slacksG = True
slacksH = True
