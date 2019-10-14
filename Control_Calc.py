# -*- coding: utf-8 -*-
"""
Created on December 3, 2015

@author: Marco, Mirco, Gabriele

MPC control calculation
"""

from builtins import range
from casadi import *
from casadi.tools import *
from matplotlib import pylab as plt
import math
import scipy.linalg as scla
import numpy as np
from Utilities import*


def opt_dyn(xSX , uSX, ySX, dSX, tSX, n, m, p, nd, Fx_model, Fy_model, F_obj, Vfin, N, QForm, DUForm, DUFormEcon, ContForm, TermCons, nw, sol_opts, umin = None, umax = None,  W = None, Z = None, ymin = None, ymax = None, xmin = None, xmax = None, Dumin = None, Dumax = None, h = None, fx = None, xstat = None, ustat = None):
    """
    SUMMARY:
    It builds the dynamic optimization problem
    """
    # Extract dimensions
    nxu = n+m 
    nxuy = nxu + p
    
    # Define symbolic optimization variables
    w = MX.sym("w",nw)  # w = [x[0],u[0], ... ,u[N-1],x[N],d,xs,us]
        
    # Get states
    X = [w[nxu*k : nxu*k+n] for k in range(N+1)]
     
    # Get controls
    U = [w[nxu*k+n : nxu*k + nxu] for k in range(N)]
    
    # Define parameters
    par = MX.sym("par", 2*nxu+nd+1+p*m)
    x0 = par[0:n]
    xs = par[n:2*n]
    us = par[2*n:n+nxu]
    d = par[n+nxu:n+nxu+nd]
    um1 = par[n+nxu+nd:2*nxu+nd]
    t = par[2*nxu+nd:2*nxu+nd+1]
    lambdayT_r = par[2*nxu+nd+1:2*nxu+nd+1+p*m]
    
    lambdayT = lambdayT_r.reshape((p,m)) #shaping lambda_r vector in order to reconstruct the matrix
    
    

    # Defining bound constraint 
    if ymin is None and ymax is None:
        yFree = True
    else:
        yFree = False
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
    if Dumin is None and Dumax is None:
        DuFree = True
    else:
        DuFree = False
        if Dumin is None:
            Dumin = -DM.inf(m)
        if Dumax is None:
            Dumax = DM.inf(m)  
        
    if h is None:
        h = .1 #Defining integrating step if not provided from the user
    
    if ContForm is True:
        xdot = fx(xSX,uSX,dSX,tSX)
        y = Fy_model( xSX, dSX, tSX) 
        ystat = Fy_model( xstat, dSX, tSX)
        F_obj1 = F_obj(xSX, uSX, y, xstat, ustat, ystat)
        
        # Create an integrator
        dae = {'x':xSX, 'p':vertcat(uSX,dSX,tSX,xstat,ustat), 'ode':xdot, 'quad':F_obj1}
        opts = {'tf':h, 't0':0.0} # final time
        F = integrator('F', 'idas', dae, opts)
    
    # Initializing constraints vectors and obj fun
    g = []
    g1 = [] # Costraint vector for y bounds
    g2 = [] # Costraint vector for Du bounds
    f_obj = 0.0;
    

    ys = Fy_model( xs, d, t) #Calculating steady-state output if necessary
    
    g.append(x0 - X[0]) #adding initial contraint to the current xhat_k|k   

    for k in range(N):
        # Correction for dynamic KKT matching
        Y_k = Fy_model( X[k], d, t) + mtimes(lambdayT,(U[k] - us))
        
        if yFree is False:
            g1.append(Y_k) #bound constraint on Y_k

        if ContForm is True: 
            Fk = F(x0=X[k], p=vertcat(U[k],d,t, xs, us))
            g.append(X[k+1] - Fk['xf'])

            # Add contribution to the objective
            f_obj += Fk['qf']

        else:
            X_next = Fx_model( X[k], U[k], h, d, t)

            if k == 0:
                DU_k = U[k] - um1
            else:
                DU_k = U[k] - U[k-1]
            
            if DuFree is False:
                g2.append(DU_k) #bound constraint on DU_k
                
            g.append(X_next - X[k+1])
            # Defining variable entering the objective function
            dx = X[k] 
            du = U[k]
            dy = Y_k
            if QForm is True:   #Checking if the OF is quadratic
                dx = dx - xs
                du = du - us
                dy = dy - ys
            if DUForm is True:  #Checking if the OF requires DU instead of u
                du = DU_k
#            if DUFormEcon is True:  #Checking if the OF requires DU instead of u
            us_obj = DU_k if DUFormEcon is True else us
        
            f_obj_new = F_obj( dx, du, dy, xs, us_obj, ys)        
            f_obj += f_obj_new
        
    dx = X[N]
    if QForm is True:   #Checking if the OF is quadratic
        dx = dx - xs
    if TermCons is True: #Adding the terminal constraint
        g.append(dx) 
        
    g = vertcat(*g)
    g1 = vertcat(*g1) #bound constraint on Y_k
    g2 = vertcat(*g2) #bound constraint on Du_k
    
    vfin = Vfin(dx,xs)
    f_obj += vfin #adding the final weight
    
    #Defining bound constraint
    w_lb = -DM.inf(nw)
    w_ub = DM.inf(nw)
    w_lb[0:n] = xmin
    w_ub[0:n] = xmax
    
    ng = g.size1()
    ng1 = g1.size1()
    ng2 = g2.size1()
    g_lb = DM.zeros(ng+ng1+ng2,1)
    g_ub = DM.zeros(ng+ng1+ng2,1)
    
    for k in range(1,N+1,1):
        w_lb[k*nxu:k*nxu+n] = xmin
        w_ub[k*nxu:k*nxu+n] = xmax
        w_lb[k*nxu-m:k*nxu] = umin
        w_ub[k*nxu-m:k*nxu] = umax
        
        if yFree is False:
            g_lb[ng+(k-1)*p: ng+k*p] = ymin
            g_ub[ng+(k-1)*p: ng+k*p] = ymax
            if DuFree is False:
                g_lb[ng+ng1+(k-1)*m: ng+ng1+k*m] = Dumin
                g_ub[ng+ng1+(k-1)*m: ng+ng1+k*m] = Dumax
        else:
            if DuFree is False:
                g_lb[ng+(k-1)*m: ng+k*m] = Dumin
                g_ub[ng+(k-1)*m: ng+k*m] = Dumax
    
    g = vertcat(g, g1, g2)
    
    nlp = {'x':w, 'p':par, 'f':f_obj, 'g':g}

    solver = nlpsol('solver', 'ipopt', nlp, sol_opts)
    
    return [solver, w_lb, w_ub, g_lb, g_ub]