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


def opt_dyn(xSX , uSX, ySX, dSX, tSX, pxSX, pySX, n, m, p, nd, npx, npy, ng_v, nh_v, Fx_model, Fy_model, F_obj, Vfin, N, QForm, DUForm, DUFormEcon, ContForm, TermCons, slacks, slacksG, slacksH, nw, sol_opts, G_ineq, H_eq, umin = None, umax = None,  W = None, Z = None, ymin = None, ymax = None, xmin = None, xmax = None, Dumin = None, Dumax = None, h = None, fx = None, xstat = None, ustat = None, Ws = None):
    """
    SUMMARY:
    It builds the dynamic optimization problem
    """
    # Extract dimensions
    nxu = n+m 
    nxuy = nxu + p
    ns = nw - (nxu*N+n)
    
    # Define symbolic optimization variables
    w = MX.sym("w",nw)  # w = [x[0],u[0], ... ,u[N-1],x[N]] , w = [x[0],u[0], ... ,u[N-1],x[N],Sl]
        
    # Get states
    X = [w[nxu*k : nxu*k+n] for k in range(N+1)]
     
    # Get controls
    U = [w[nxu*k+n : nxu*k + nxu] for k in range(N)]
    
    if slacks == True:
      Sl = w[nw-ns:nw] # 2*ny (+ng) (+nh)
    
    # Define parameters
    par = MX.sym("par", 2*nxu+nd+1+p*m+npx*N+npy*N)
    x0 = par[0:n]
    xs = par[n:2*n]
    us = par[2*n:n+nxu]
    d = par[n+nxu:n+nxu+nd]
    um1 = par[n+nxu+nd:2*nxu+nd]
    t = par[2*nxu+nd:2*nxu+nd+1]
    lambdayT_r = par[2*nxu+nd+1:2*nxu+nd+1+p*m]
    par_xmk_r = par[2*nxu+nd+1+p*m:2*nxu+nd+1+p*m+npx*N]
    par_ymk_r = par[2*nxu+nd+1+p*m+npx*N:2*nxu+nd+1+p*m+npx*N+npy*N]
    
    lambdayT = lambdayT_r.reshape((p,m)) #shaping lambda_r vector in order to reconstruct the matrix
    
    par_xmk = reshape(par_xmk_r,npx,N) #shaping par_xmk_r vector in order to reconstruct the matrix
    par_ymk = reshape(par_ymk_r,npy,N) #shaping par_xmk_r vector in order to reconstruct the matrix

    # Defining bound constraint 
    if ymin is None and ymax is None:
        yFree = True
    else:
        yFree = False
        if ymin is None:
            if slacks == False:
                ymin = -DM.inf(p)
            else:
                ymin = -1e12*DM.ones(p)
        if ymax is None:
            if slacks == False:
                ymax = DM.inf(p)
            else:
                ymax = 1e12*DM.ones(p)    
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
    
    if G_ineq != None:
        g_ineq = G_ineq(xSX,uSX,ySX,dSX,tSX,pxSX,pySX)
        G_ineqSX = Function('G_ineqSX', [xSX,uSX,ySX,dSX,tSX,pxSX,pySX], [g_ineq])
        
    if H_eq != None:
        h_eq = H_eq(xSX,uSX,ySX,dSX,tSX,pxSX,pySX)
        H_eqSX = Function('H_eqSX', [xSX,uSX,ySX,dSX,tSX,pxSX,pySX], [h_eq])
        
    if ContForm is True:
        xdot = fx(xSX,uSX,dSX,tSX,pxSX) + pxSX
        y = Fy_model( xSX,uSX, dSX, tSX, pySX) 
        ystat = Fy_model( xstat, ustat, dSX, tSX, pySX)
        F_obj1 = F_obj(xSX, uSX, y, xstat, ustat, ystat)
        
        # Create an integrator
        dae = {'x':xSX, 'p':vertcat(uSX,dSX,tSX,xstat,ustat,pxSX,pySX), 'ode':xdot, 'quad':F_obj1}
        opts = {'tf':h, 't0':0.0} # final time
        F = integrator('F', 'idas', dae, opts)
    
    # Initializing constraints vectors and obj fun
    g = []
    g1 = [] # Costraint vector for y bounds
    g2 = [] # Costraint vector for Du bounds
    g4 = [] # User defined inequality constraints
    g5 = [] # User defined equality constraints
    f_obj = 0.0;
    sl_ub = []
    sl_lb = []
    

    ys = Fy_model( xs, us, d, t, par_ymk[:,0]) #Calculating steady-state output if necessary
    
    g.append(x0 - X[0]) #adding initial contraint to the current xhat_k|k   

    for k in range(N):
        # Correction for dynamic KKT matching
        Y_k = Fy_model( X[k], U[k], d, t, par_ymk[:,k]) + mtimes(lambdayT,(U[k] - us))
        
        if G_ineq != None:
            if slacks == True and slacksG == True:
                G_k = G_ineqSX(X[k], U[k], Y_k, d, t, par_xmk[:,k], par_ymk[:,k]) - Sl[2*p:2*p+ng_v]
            else:
                G_k = G_ineqSX(X[k], U[k], Y_k, d, t, par_xmk[:,k], par_ymk[:,k])
        else:
            G_k = []
        if H_eq != None:
            if slacks == True and slacksH == True:
                H_k = H_eqSX(X[k], U[k], Y_k, d, t, par_xmk[:,k], par_ymk[:,k]) - Sl[2*p+ng_v:2*p+ng_v+nh_v]
            else:
                H_k = H_eqSX(X[k], U[k], Y_k, d, t, par_xmk[:,k], par_ymk[:,k])
        else:
            H_k = []
        
        g4.append(G_k)
        g5.append(H_k)
        
        if yFree is False:
            g1.append(Y_k) #bound constraint on Y_k

        if ContForm is True: 
            Fk = F(x0=X[k], p=vertcat(U[k],d,t, xs, us, par_xmk[:,k], par_ymk[:,k]))
            g.append(X[k+1] - Fk['xf'])

            # Add contribution to the objective
            f_obj += Fk['qf']

        else:
            X_next = Fx_model( X[k], U[k], h, d, t, par_xmk[:,k])

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
            if slacks == True:
                f_obj_new = F_obj( dx, du, dy, xs, us_obj, ys) + mtimes(Sl.T,mtimes(Ws,Sl))
            f_obj += f_obj_new
            
            if slacks == True:
                sl_ub.append(Sl[0:p])
                sl_lb.append(Sl[p:2*p])
        
    dx = X[N]
    if QForm is True:   #Checking if the OF is quadratic
        dx = dx - xs
    if TermCons is True: #Adding the terminal constraint
        g.append(dx) 
        
    g = vertcat(*g)
    g1 = vertcat(*g1) #bound constraint on Y_k
    g2 = vertcat(*g2) #bound constraint on Du_k
    g4 = vertcat(*g4) 
    g5 = vertcat(*g5)
    if slacks == True:
        sl_ub = vertcat(*sl_ub)
        sl_lb = vertcat(*sl_lb)
        
    vfin = Vfin(dx,xs)
    f_obj += vfin #adding the final weight
    
    #Defining bound constraint
    w_lb = -DM.inf(nw)
    w_ub = DM.inf(nw)
    w_lb[0:n] = xmin
    w_ub[0:n] = xmax
    w_lb[nw-ns:nw] = DM.zeros(ns) # sl > 0
    
    ng = g.size1()
    ng1 = g1.size1()
    ng2 = g2.size1()
    ng4 = g4.size1()
    ng5 = g5.size1()
    g_lb = DM.zeros(ng+ng1+ng2+ng4+ng5,1)
    g_ub = DM.zeros(ng+ng1+ng2+ng4+ng5,1)
    
    if ng1 != 0:
       if slacks == False: 
           g_lb[ng:ng+ng1] = mtimes(ymin,DM.ones(N).T).reshape((ng1,1))
           g_ub[ng:ng+ng1] = mtimes(ymax,DM.ones(N).T).reshape((ng1,1))
       else:
           g_lb = DM.zeros(ng+2*ng1+ng2+ng4+ng5,1)
           g_ub = DM.zeros(ng+2*ng1+ng2+ng4+ng5,1)
           g1_old = g1
           g1 = MX.zeros(2*ng1)
           g1[0:ng1] = mtimes(ymin,DM.ones(N).T).reshape((ng1,1)) - g1_old - sl_lb
           g1[ng1:2*ng1] = -mtimes(ymax,DM.ones(N).T).reshape((ng1,1)) + g1_old - sl_ub
           g_lb[ng:ng+2*ng1] = -DM.inf(2*ng1)
           ng1 = g1.size1()
    
    if ng2 != 0:
       g_lb[ng+ng1:ng+ng1+ng2] = mtimes(Dumin,DM.ones(N).T).reshape((ng2,1))
       g_ub[ng+ng1:ng+ng1+ng2] = mtimes(Dumax,DM.ones(N).T).reshape((ng2,1))
              
    if ng4 != 0:
       g_lb[ng+ng1+ng2:ng+ng1+ng2+ng4] = -DM.inf(ng4)
    
    for k in range(1,N+1,1):
        w_lb[k*nxu:k*nxu+n] = xmin
        w_ub[k*nxu:k*nxu+n] = xmax
        w_lb[k*nxu-m:k*nxu] = umin
        w_ub[k*nxu-m:k*nxu] = umax
        
    g = vertcat(g, g1, g2, g4, g5)
    
    nlp = {'x':w, 'p':par, 'f':f_obj, 'g':g}

    solver = nlpsol('solver', 'ipopt', nlp, sol_opts)
    
    return [solver, w_lb, w_ub, g_lb, g_ub]



def opt_dyn_CM(xSX , uSX, ySX, dSX, tSX, pxSX, pySX, n, m, p, nd, npx, npy, ng_v, nh_v, Fx_model, Fy_model, F_obj, Vfin, N, QForm, DUForm, DUFormEcon, ContForm, TermCons, slacks, slacksG, slacksH, nw, sol_opts, G_ineq , H_eq,  umin = None, umax = None,  W = None, Z = None, ymin = None, ymax = None, xmin = None, xmax = None, Dumin = None, Dumax = None, h = None, fx = None, xstat = None, ustat = None, Mx = None, Ws = None):

    """
    SUMMARY:
    It builds the dynamic optimization problem
    """
    # Extract dimensions
    nxu = n+m
    nxuk = 3*n+m
    nxuy = nxu + p
    ns = nw - (nxu*N+n)
    
    nw = nw + 2*n*N #number of optimization variables for collocation methods
    
    # Define symbolic optimization variables
    w = MX.sym("w",nw)  # w = [x[0],s1[0],s2[0],u[0], ... ,u[N-1],x[N],d,xs,us]
    
        
    # Get states
    X = [w[nxuk*k : nxuk*k+n] for k in range(N+1)]
    # Get internal states
    S1 = [w[nxuk*k+n : nxuk*k+2*n] for k in range(N)]
    S2 = [w[nxuk*k+2*n : nxuk*k+3*n] for k in range(N)]
    # Get controls
    U = [w[nxuk*k+3*n : nxuk*k +3*n+m] for k in range(N)]
    
    if slacks == True:
      Sl = w[nw-ns:nw] # 2*ny (+ng) (+nh)
    
    # Define parameters
    par = MX.sym("par", 2*nxu+nd+1+p*m+npx*N+npy*N)
    x0 = par[0:n]
    xs = par[n:2*n]
    us = par[2*n:n+nxu]
    d = par[n+nxu:n+nxu+nd]
    um1 = par[n+nxu+nd:2*nxu+nd]
    t = par[2*nxu+nd:2*nxu+nd+1]
    lambdayT_r = par[2*nxu+nd+1:2*nxu+nd+1+p*m]
    par_xmk_r = par[2*nxu+nd+1+p*m:2*nxu+nd+1+p*m+npx*N]
    par_ymk_r = par[2*nxu+nd+1+p*m+npx*N:2*nxu+nd+1+p*m+npx*N+npy*N]
    
    lambdayT = lambdayT_r.reshape((p,m)) #shaping lambda_r vector in order to reconstruct the matrix
    
    par_xmk = reshape(par_xmk_r,npx,N) #shaping par_xmk_r vector in order to reconstruct the matrix
    par_ymk = reshape(par_ymk_r,npy,N) #shaping par_xmk_r vector in order to reconstruct the matrix
    
    # Defining bound constraint 
    if ymin is None and ymax is None:
        yFree = True
    else:
        yFree = False
        if ymin is None:
            if slacks == False:
                ymin = -DM.inf(p)
            else:
                ymin = -1e12*DM.ones(p)
        if ymax is None:
            if slacks == False:
                ymax = DM.inf(p)
            else:
                ymax = 1e12*DM.ones(p)
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
        
    hSX = SX.sym("h_SX", 1) 
    
    if G_ineq != None:
        g_ineq = G_ineq(xSX,uSX,ySX,dSX,tSX,pxSX,pySX)
        G_ineqSX = Function('G_ineqSX', [xSX,uSX,ySX,dSX,tSX,pxSX,pySX], [g_ineq])
        
    if H_eq != None:
        h_eq = H_eq(xSX,uSX,ySX,dSX,tSX,pxSX,pySX)
        H_eqSX = Function('H_eqSX', [xSX,uSX,ySX,dSX,tSX,pxSX,pySX], [h_eq])
        
    
    fx_SX = fx(xSX,uSX,dSX,tSX,pxSX)
    Fx_SX = Function('Fx_SX', [xSX,uSX,hSX,dSX,tSX,pxSX], [fx_SX])
    
    if ContForm is True:
        xdot = fx(xSX,uSX,dSX,tSX) + pxSX
        y = Fy_model( xSX, uSX, dSX, tSX, pySX) 
        ystat = Fy_model( xstat, ustat, dSX, tSX, pySX)
        F_obj1 = F_obj(xSX, uSX, y, xstat, ustat, ystat)
        
        # Create an integrator
        dae = {'x':xSX, 'p':vertcat(uSX,dSX,tSX,xstat,ustat,pxSX,pySX), 'ode':xdot, 'quad':F_obj1}
        opts = {'tf':h, 't0':0.0} # final time
        F = integrator('F', 'idas', dae, opts)
    
    ### INTEGRATION WITH COLLOCATION ####
    #coefficient of Butcher tableau with Gauss-Legendre method
    c1 = 1/2-(3**0.5)/6 ; c2 = 1/2+(3**0.5)/6
    a11 = 1/4 ; a12 = 1/4-(3**0.5)/6
    a21 = 1/4+(3**0.5)/6; a22 = 1/4
    b1 = 1/2 ; b2 = 1/2
    b = [b1,b2]
    
    A = [[a11,a12],[a21,a22]]
    D = np.linalg.inv(A)
    D11 = D[0,0] ; D12 = D[0,1]
    D21 = D[1,0] ; D22 = D[1,1]
    b_t = np.dot(D.T,b)
    b1_t = b_t[0] ; b2_t = b_t[1]
    
    # Initializing constraints vectors and obj fun
    g = []
    g1 = [] # Costraint vector for y bounds
    g2 = [] # Costraint vector for Du bounds
    g3 = [] # Constraint vector for S internal states
    g4 = [] # User defined inequality constraints
    g5 = [] # User defined equality constraints
    f_obj = 0.0;
    sl_ub = []
    sl_lb = []
    
    

    ys = Fy_model( xs, us, d, t, par_ymk[:,0]) #Calculating steady-state output if necessary
    
    g.append(x0 - X[0]) #adding initial contraint to the current xhat_k|k   

    for k in range(N):
        
        # Correction for dynamic KKT matching
        Y_k = Fy_model( X[k], U[k], d, t, par_ymk[:,k]) + mtimes(lambdayT,(U[k] - us))
        
        if G_ineq != None:
            if slacks == True and slacksG == True:
                G_k = G_ineqSX(X[k], U[k], Y_k, d, t, par_xmk[:,k], par_ymk[:,k]) - Sl[2*p:2*p+ng_v]
            else:
                G_k = G_ineqSX(X[k], U[k], Y_k, d, t, par_xmk[:,k], par_ymk[:,k])
        else:
            G_k = []
        if H_eq != None:
            if slacks == True and slacksH == True:
                H_k = H_eqSX(X[k], U[k], Y_k, d, t, par_xmk[:,k], par_ymk[:,k]) - Sl[2*p+ng_v:2*p+ng_v+nh_v]
            else:
                H_k = H_eqSX(X[k], U[k], Y_k, d, t, par_xmk[:,k], par_ymk[:,k])
        else:
            H_k = []
        
        g4.append(G_k)
        g5.append(H_k)
        
        if yFree is False:
            g1.append(Y_k) #bound constraint on Y_k

        if ContForm is True: 
            Fk = F(x0=X[k], p=vertcat(U[k],d,t, xs, us, par_xmk[:,k], par_ymk[:,k]))
            g.append(X[k+1] - Fk['xf'])

            # Add contribution to the objective
            f_obj += Fk['qf']
            

        else:
            X_next = X[k] + b1_t*(S1[k] - X[k]) + b2_t*(S2[k] - X[k])#transition to the next state wìfor collocation method in state representation
            #X_next = X[k]+h*(b1*S1[k]+b2*S2[k]) #transition to the next state for collocation method in derivative state


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
            ds1 = S1[k]
            ds2 = S2[k]
            ds = vertcat(ds1,ds2)
            
            if QForm is True:   #Checking if the OF is quadratic
                dx = dx - xs
                du = du - us
                dy = dy - ys
                ds1 = ds1 - xs
                ds2 = ds2 - xs
                ds = vertcat(ds1,ds2)
                
                
            if DUForm is True:  #Checking if the OF requires DU instead of u
                du = DU_k
#            if DUFormEcon is True:  #Checking if the OF requires DU instead of u
            us_obj = DU_k if DUFormEcon is True else us
              
            #equality constrain in state rappresentation
            rg1 = 1/h*(D11*(S1[k]-X[k])+D12*(S2[k]-X[k]))-Fx_SX(S1[k],U[k],h,d,t,par_xmk[:,0])
            rg2 = 1/h*(D21*(S1[k]-X[k])+D22*(S2[k]-X[k]))-Fx_SX(S2[k],U[k],h,d,t,par_xmk[:,0])
            
            #equality constrain in derivative states S==K
            #rg1 = S1[k] - fx(X[k]+h*(a11*S1[k]+a12*S2[k]),U[k],h,d,t,dxm)
            #rg2 = S2[k] - fx(X[k]+h*(a21*S1[k]+a22*S2[k]),U[k],h,d,t,dxm)
            
            g3.append(rg1) #internal states
            g3.append(rg2) 
            
            f_obj_new = F_obj( dx, du, dy, xs, us_obj, ys, ds)
            if slacks == True:
                f_obj_new = F_obj( dx, du, dy, xs, us_obj, ys, ds) + mtimes(Sl.T,mtimes(Ws,Sl))
            f_obj += f_obj_new
            
            if slacks == True:
                sl_ub.append(Sl[0:p])
                sl_lb.append(Sl[p:2*p])
        
    dx = X[N]
    if QForm is True:   #Checking if the OF is quadratic
        dx = dx - xs
    if TermCons is True: #Adding the terminal constraint
        g.append(dx) 
        
    g = vertcat(*g)
    g1 = vertcat(*g1) #bound constraint on Y_k
    g2 = vertcat(*g2) #bound constraint on Du_k
    g3 = vertcat(*g3) #bound constraint on S_k
    g4 = vertcat(*g4)
    g5 = vertcat(*g5)
    if slacks == True:
        sl_ub = vertcat(*sl_ub)
        sl_lb = vertcat(*sl_lb)
        
    
    vfin = Vfin(dx,xs)
    f_obj += vfin #adding the final weight
    
    #Defining bound constraint
    w_lb = -DM.inf(nw)
    w_ub = DM.inf(nw)
    w_lb[0:n] = xmin
    w_ub[0:n] = xmax
    w_lb[nw-ns:nw] = DM.zeros(ns) # sl > 0
    
    ng = g.size1() #x
    ng1 = g1.size1() #y
    ng2 = g2.size1() #u
    ng3 = g3.size1()
    ng4 = g4.size1()
    ng5 = g5.size1()
    g_lb = DM.zeros(ng+ng1+ng2+ng3+ng4+ng5,1)
    g_ub = DM.zeros(ng+ng1+ng2+ng3+ng4+ng5,1)
    
    
    if ng1 != 0:  # yFree = False
       if slacks == False: 
           g_lb[ng:ng+ng1] = mtimes(ymin,DM.ones(N).T).reshape((ng1,1))
           g_ub[ng:ng+ng1] = mtimes(ymax,DM.ones(N).T).reshape((ng1,1))
       else:
           # Ridefine g1 (- inf < g1 < 0 )
           g_lb = DM.zeros(ng+2*ng1+ng2+ng3+ng4+ng5,1) 
           g_ub = DM.zeros(ng+2*ng1+ng2+ng3+ng4+ng5,1)
           g1_old = g1
           g1 = MX.zeros(2*ng1)
           g1[0:ng1] = mtimes(ymin,DM.ones(N).T).reshape((ng1,1)) - g1_old - sl_lb
           g1[ng1:2*ng1] = -mtimes(ymax,DM.ones(N).T).reshape((ng1,1)) + g1_old - sl_ub
           g_lb[ng:ng+2*ng1] = -DM.inf(2*ng1)
           ng1 = g1.size1()
    
    if ng2 != 0:
       g_lb[ng+ng1:ng+ng1+ng2] = mtimes(Dumin,DM.ones(N).T).reshape((ng2,1))
       g_ub[ng+ng1:ng+ng1+ng2] = mtimes(Dumax,DM.ones(N).T).reshape((ng2,1))
              
    if ng4 != 0:
       g_lb[ng+ng1+ng2+ng3:ng+ng1+ng2+ng3+ng4] = -DM.inf(ng4)
       
                    
    for k in range(1,N+1,1):
        w_lb[k*nxuk:k*nxuk+n] = xmin
        w_ub[k*nxuk:k*nxuk+n] = xmax
        w_lb[k*nxuk-m-2*n:k*nxuk-m] = vertcat(xmin,xmin)
        w_ub[k*nxuk-m-2*n:k*nxuk-m] = vertcat(xmax,xmax)
        w_lb[k*nxuk-m:k*nxuk] = umin
        w_ub[k*nxuk-m:k*nxuk] = umax
        
    
    g = vertcat(g, g1, g2, g3, g4, g5)
    
    nlp = {'x':w, 'p':par, 'f':f_obj, 'g':g}

    solver = nlpsol('solver', 'ipopt', nlp, sol_opts)
    
    return [solver, w_lb, w_ub, g_lb, g_ub]
