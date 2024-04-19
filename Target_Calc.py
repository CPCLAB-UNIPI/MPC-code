# -*- coding: utf-8 -*-
"""
Created on December 3, 2015

@author: Marco, Mirco, Gabriele

Target calculation
"""

from casadi import *
from casadi.tools import *
from matplotlib import pylab as plt
import math
from math import *
import scipy.linalg as scla
import numpy as np
from Utilities import*

# Target calculation: model constraints and objective function
def opt_ss(n, m, p, nd, npx, npy, Fx_model,Fy_model,Fss_obj,QForm_ss,DUssForm, sol_opts, G_ineq_SS, H_eq_SS, umin = None, umax = None, w_s = None, z_s = None, ymin = None, ymax = None, xmin = None, xmax = None, h = None):
    """
    SUMMARY:
    It builds the target optimization problem
    """
    nxu = n+m 
    nxuy = nxu + p
    
    # Define symbolic optimization variables
    wss = MX.sym("wss",nxuy) 
    
    # Get states
    Xs = wss[0:n]
    
    # Get controls
    Us = wss[n:nxu]
    
    # Get output
    Ys = wss[nxu:nxuy]
    
    # Define parameters
    par_ss = MX.sym("par_ss", 2*m+p+nd+p*m+n+1+npx+npy)
    usp = par_ss[0:m]   
    ysp = par_ss[m:m+p]
    xsp = par_ss[m+p:m+p+n]
    d = par_ss[m+p+n:m+p+n+nd]
    Us_prev = par_ss[m+p+n+nd:2*m+p+n+nd]
    lambdaT_r = par_ss[2*m+p+n+nd:2*m+p+n+nd+p*m]
    t = par_ss[2*m+p+nd+n+p*m:2*m+p+nd+n+p*m+1]
    px = par_ss[2*m+p+nd+n+p*m+1:2*m+p+nd+n+p*m+1+npx]
    py = par_ss[2*m+p+nd+n+p*m+1+npx:2*m+p+nd+n+p*m+1+npx+npy]
    
    lambdaT = lambdaT_r.reshape((p,m)) #shaping lambda_r vector in order to reconstruct the matrix
        
    # Defining constraints
    if ymin is None:
        ymin = -DM.inf(p)
    if ymax is None:
        ymax = DM.inf(p)
    if xmin is None:
        xmin = -DM.inf(n)
    if xmax is None:
        xmax = DM.inf(n)
    if umin is None:
        umin = -DM.inf(m)
    if umax is None:
        umax = DM.inf(m) 
    
    if h is None:
        h = .1 #Defining integrating step if not provided from the user
            
    gss = []
    gss1 = [] 
    gss2 = []
    
    Xs_next = Fx_model( Xs, Us, h, d, t,px) 
    
    gss.append(Xs_next - Xs)
    gss = vertcat(*gss)
    
    Ys_next = Fy_model( Xs, Us, d, t, py) + mtimes(lambdaT,(Us - Us_prev))
    gss = vertcat(gss , Ys_next- Ys)
    
    # initialization of parameters as symbolic variables
    xSX = SX.sym("xSX", n); uSX = SX.sym("uSX", m); dSX = SX.sym("dSX", nd)
    tSX = SX.sym("tSX", 1); pxSX = SX.sym("pxSX", npx); pySX = SX.sym("pySX",npy); ySX = SX.sym("ySX", p)
    
    if G_ineq_SS != None:
        g_ineq_SS = G_ineq_SS(xSX,uSX,ySX,dSX,tSX,pxSX, pySX)
        G_ineqSS_SX = Function('G_ineqSS_SX', [xSX,uSX,ySX,dSX,tSX,pxSX,pySX], [g_ineq_SS])
        
    if H_eq_SS != None:
        h_eq_SS = H_eq_SS(xSX,uSX,ySX,dSX,tSX,pxSX,pySX)
        H_eqSS_SX = Function('H_eqSS_SX', [xSX,uSX,ySX,dSX,tSX,pxSX,pySX], [h_eq_SS])
    
    if G_ineq_SS != None:
        G_ss = G_ineqSS_SX(Xs, Us, Ys, d, t, px, py)
    else:
        G_ss = []
        
    if H_eq_SS != None:
        H_ss = H_eqSS_SX(Xs, Us, Ys, d, t, px, py)
    else:
        H_ss = []
    
    gss1.append(G_ss)
    gss2.append(H_ss)
    
    gss1 = vertcat(*gss1)
    gss2 = vertcat(*gss2)

    # Defining obj_fun
    dy = Ys
    du = Us
    dx = Xs
    
    if QForm_ss is True:   #Checking if the OF is quadratic
        dx = dx - xsp
        dy = dy - ysp
        du = du - usp
        
    if DUssForm is True:
        du = Us - Us_prev #Adding weight on du
            
    fss_obj = Fss_obj( dx, du, dy, xsp, usp, ysp) 
    
    #Defining bound constraint
    wss_lb = -DM.inf(nxuy)
    wss_ub = DM.inf(nxuy)
    wss_lb[0:n] = xmin 
    wss_ub[0:n] = xmax
    wss_lb[n: nxu] = umin
    wss_ub[n: nxu] = umax
    wss_lb[nxu: nxuy] = ymin 
    wss_ub[nxu: nxuy] = ymax
    
    try: 
        ng = gss.size1()
    except AttributeError:
        ng = gss.__len__()
    try: 
        ng1 = gss1.size1()
    except AttributeError:
        ng1 = gss1.__len__()    
    try: 
        ng2 = gss2.size1()
    except AttributeError:
        ng2 = gss2.__len__() 
        
    gss_lb = DM.zeros(ng+ng1+ng2,1)   # Equalities identification 
    gss_ub = DM.zeros(ng+ng1+ng2,1)
    
    if ng1 != 0:
        gss_lb[ng:ng+ng1] = -DM.inf(ng1)
    
    gss = vertcat (gss, gss1, gss2)
    
    nlp_ss = {'x':wss, 'p':par_ss, 'f':fss_obj, 'g':gss}
    
    solver_ss = nlpsol('solver','ipopt', nlp_ss, sol_opts)
    
    return [solver_ss, wss_lb, wss_ub, gss_lb, gss_ub]