# -*- coding: utf-8 -*-
"""
Created on December 3, 2015

@author: Marco, Mirco, Gabriele

Estimator envelope file
"""
from __future__ import division

from builtins import range
from past.utils import old_div
from casadi import *
from casadi.tools import *
from matplotlib import pylab as plt
import math
import scipy.linalg as scla
import numpy as np
from Utilities import*

def defEstimator(Fx,Fy,y_k,u_k, estype,xhat_min, t_k, p_x, p_y, **kwargsin):
    if estype == 'kal':
        Q = kwargsin['Q']
        R = kwargsin['R']
        P_min = kwargsin['P_min']
        [P_plus, P_corr, xhat_corr] = kalman(Fx,Fy,y_k,u_k,Q,R,P_min,xhat_min,t_k,p_y)
        kwargsout = {"P_plus": P_plus, "P_corr": P_corr}
    elif estype == 'ekf':
        Q = kwargsin['Q']
        R = kwargsin['R']
        P_min = kwargsin['P_min']
        ts = kwargsin['ts']
        [P_plus, P_corr, xhat_corr] = ekf(Fx,Fy,y_k,u_k,Q,R,P_min,xhat_min,ts,t_k,p_y,p_x)
        kwargsout = {"P_plus": P_plus, "P_corr": P_corr}
    elif estype == 'kalss':
        K = kwargsin['K']
        [xhat_corr] = kalss(Fx,Fy,y_k,u_k,K,xhat_min,t_k,p_y)
        kwargsout = {}
    elif estype == 'mhe':
        Fobj = kwargsin['Fobj']
        P_k = kwargsin['P_min']
        
        sol = kwargsin['sol']
        solwlb = kwargsin['solwlb']
        solwub = kwargsin['solwub']
        solglb = kwargsin['solglb']
        solgub = kwargsin['solgub']
        N = kwargsin['N']
        ts = kwargsin['ts']
        v_k = kwargsin['vk']
        w_k = kwargsin['wk']
        U = kwargsin['U']
        X = kwargsin['X']
        Xm = kwargsin['Xm']
        Y = kwargsin['Y']
        T = kwargsin['T']
        V = kwargsin['V']
        W = kwargsin['W']
        xb = kwargsin['xb']
        up = kwargsin['up']
        Nmhe = kwargsin['Nmhe']
        C = kwargsin['C']
        G = kwargsin['G']
        A = kwargsin['A']
        B = kwargsin['B']
        f = kwargsin['f']
        h = kwargsin['h']
        Qk = kwargsin['Qk']
        Rk = kwargsin['Rk']
        Sk = kwargsin['Sk']
        Q = kwargsin['Q']
        bU = kwargsin['bU']
        P = kwargsin['P']
        Pc = kwargsin['Pc']
        P_kal = kwargsin['P_kal']
        P_c_kal = kwargsin['P_c_kal']
        pH = kwargsin['pH']
        pO = kwargsin['pO']
        pPyx = kwargsin['pPyx']
        xm_kal = kwargsin['xm_kal']
        PX = kwargsin['PX']
        PY = kwargsin['PY']
        nd = kwargsin['nd']
        
        
                     
        [P_k, xhat_corr, w_k,v_k,U,Y,T,Xm,X,V,W,xb,C,G, A, B,\
         f, h, Qk, Rk, Sk, Q, bU,P, Pc, P_kal, P_c_kal, pH,pO,pPyx, xm_kal, xc_kal, PX, PY] = \
         mhe(Fx,Fy,y_k,u_k,P_k,\
        xhat_min,Fobj,ts,t_k,p_x,p_y,U,Y,T,Xm,X,V,W,w_k,v_k,xb,N,up,Nmhe,sol,solwlb,solwub,solglb,solgub,\
        C, G, A, B, f, h, Qk, Rk, Sk, Q, bU,\
        P, Pc, P_kal, P_c_kal, pH,pO,pPyx, xm_kal, PX, PY, nd)
        kwargsout = {"P_plus": P_k, "U_mhe" : U, "X_mhe" : X, "Xm_mhe" : Xm,\
                     "Y_mhe" : Y, "T_mhe" : T,"V_mhe" : V, "W_mhe" : W, "wk" : w_k, "vk" : v_k, "xb" : xb,\
                     "C_mhe" : C,  "G_mhe" : G, "A_mhe" : A, "B_mhe" : B, "f_mhe" : f,\
                     "h_mhe" : h, "Qk_mhe" : Qk, "Rk_mhe" : Rk, "Sk_mhe" : Sk,\
                     "Q_mhe" : Q, "bigU_mhe" : bU, "P_mhe" : P, "Pc_mhe" : Pc, \
                     "P_kal_mhe" : P_kal, "P_c_kal_mhe" : P_c_kal, "pH_mhe" : pH, \
                     "pO_mhe" : pO, "pPyx_mhe" : pPyx, "xm_kal_mhe" : xm_kal,"xc_kal_mhe" : xc_kal,\
                         "PX_mhe":PX, "PY_mhe":PY}
    return [xhat_corr, kwargsout]

def Kkalss(ny, nd, nx, Q_kf, R_kf, offree, linmod, *var, **kwargs):
    """
    SUMMARY:
    Discrete-time steady-state Kalman filter gain calculation for the given
    linear system in state space form.
    
    SYNTAX:
    assignment = kalman(ny, nd, nx, Q_kf, R_kf, offree, linmod, *var, **kwargs)
  
    ARGUMENTS:
    + nx, ny , nd  - State, output and disturbance dimensions
    + Q_kf - Process noise covariance matrix
    + R_kf - Measurements noise covariance matrix
    + offree - Offset free tag
    + linmod - Lineaity of the model tag
    + var - Positional variables
    + kwargs - Model and Disturbance matrices
       
    OUTPUTS:
    + Kaug  - Steady-state Kalman filter gain 
    """    
    if linmod == 'onlyA' or linmod == 'full':
        A = kwargs['A']
    
    if linmod == 'onlyC' or linmod == 'full':
        C = kwargs['C']
     
    try:
        A
    except NameError:
        Fx_model = kwargs['Fx']
        x = var[0]
        u = var[1]
        k = var[2]
        d = var[3]
        t = var[4]
        h = var[5]
        px = var[6]
        py= var[7]
        x_ss = var[8]
        u_ss= var[9]
        px_ss= var[10]
        py_ss= var[11]
        
         # get the system matrices
        Fun_in = SX.get_input(Fx_model)
        Adummy = jacobian(Fx_model.call(Fun_in)[0], Fun_in[0])    
        
        d_ss = DM.zeros(nd)
        if offree == 'nl':
            xnew = vertcat(x,d)
            x_ss_p = vertcat(x_ss,d_ss)
        else:
            xnew = x
            x_ss_p = x_ss
            
        A_dm = Function('A_dm', [xnew,u,k,t,px], [Adummy])
        
        A = A_dm(x_ss_p, u_ss, h, 0, px_ss)
        
    try:
        C
    except NameError:
        Fy_model = kwargs['Fy']
        Fun_in = SX.get_input(Fy_model)
        Cdummy = jacobian(Fy_model.call(Fun_in)[0], Fun_in[0]) 
        
        if 'Fx_model' not in locals:
            x = var[0]
            u = var[1]
            k = var[2]
            d = var[3]
            t = var[4]
            h = var[5]
            px = var[6]
            py= var[7]
            x_ss = var[8]
            u_ss= var[9]
            px_ss= var[10]
            py_ss= var[11]
        
        C_dm = Function('C_dm', [x,u,d,t,py], [Cdummy])
        
        C = C_dm(x_ss, u_ss, d_ss, 0.0, py_ss)

    
    Aaug = DM.eye(nx+nd)
    Caug = DM.zeros(ny, nx+nd)
    
    if offree == 'nl':
        if A.size2() < nx+nd:
            Aaug[0:nx,0:nx] = A
        else:
            Aaug = A
            
        if C.size2() < nx+nd:
            Caug[0:ny,0:nx] = C
        else:
            Caug = C
    else:
        Aaug[0:nx,0:nx] = A
        Caug[0:ny,0:nx] = C
    
    if offree == "lin":
        Bd = kwargs["Bd"]
        Cd = kwargs["Cd"]
        
        Aaug[0:nx,nx:nx+nd] = Bd
        Caug[0:ny,nx:nx+nd] = Cd
    
    Ae = np.array(Aaug.T)
    Be = np.array(Caug.T)
    Qe = np.array(Q_kf)
    Re = np.array(R_kf)
    Pe = scla.solve_discrete_are(Ae,Be,Qe,Re)
    MAT = np.dot(Be.transpose(), Pe)
    MAT = np.dot(MAT, Be) + Re
    invMAT = np.linalg.inv(MAT)
    Ke = np.dot(Pe,Be)
    Ke = np.dot(Ke, invMAT)
    Kaug = DM(Ke)

    # Eigenvalue checks
    Aobs = Aaug - mtimes(mtimes(Aaug, Kaug), Caug)
    eigvals, eigvecs = scla.eig(Aobs)
    
    return (Kaug)
    
def kalss(Fx,Fy,y_act,u_k,K,xhat_min,t_k,p_y):
    """
    SUMMARY:
    Steady-state Discrete-time Kalman filter for the given linear system 
    in state space form.
    
    SYNTAX:
    assignment = kalman(Fx,Fy,y_act,u_k,K,xhat_min,t_k)
  
    ARGUMENTS:
    + Fx - State correlation function  
    + Fy - Output correlation function    
    + y_act - Measurements, i.e. y(k)
    + u_k - Input, i.e. u(k)
    + K - Kalman filter gain
    + xhat_min - Predicted mean of the state, i.e. x(k|k-1)
    + t_k - Current time index
        
    OUTPUTS:
    + xhat_corr - Estimated mean of the state, i.e. x(k|k) 
    """    
    # predicted output: y(k|k-1) 
    yhat = Fy(xhat_min,u_k,t_k,p_y) 
    
    # estimation error
    e_k = y_act - yhat
    
    # estimated mean of the state: x(k|k) 
    xhat_corr = DM(xhat_min + mtimes(K, e_k))
    
    return [xhat_corr]
    
def kalman(Fx,Fy,y_act,u_k,Q,R,P_min,xhat_min,t_k,p_y):
    """
    SUMMARY:
    Discrete-time Kalman filter for the given linear system in state space form.
    
    SYNTAX:
    assignment = kalman(Fx_model,Fy_model,y_act,u,Q,R,P_min,xhat_min)
  
    ARGUMENTS:
    + Fx - State correlation function  
    + Fy - Output correlation function    
    + Q - Process noise covariance matrix
    + R - Measurements noise covariance matrix
    + y_act - Measurements, i.e. y(k)
    + u - Input, i.e. u(k)
    + P_min - Predicted covariance of the state, i.e. P(k|k-1)
    + xhat_min - Predicted mean of the state, i.e. x(k|k-1)
    + t_k - Current time index
        
    OUTPUTS:
    + P_plus - Predicted covariance of the state, i.e. P(k+1|k)
    + P_corr - Estimated covariance of the state, i.e. P(k|k) 
    + xhat_corr - Estimated mean of the state, i.e. x(k|k) 
    """    
    # get the system matrices
    Fun_in = SX.get_input(Fx)
    A_dm = jacobian(Fx.call(Fun_in)[0], Fun_in[0])    
    Fun_in = SX.get_input(Fy)
    C_dm = jacobian(Fy.call(Fun_in)[0], Fun_in[0]) 
    
    # predicted output: y(k|k-1) 
    yhat = Fy(xhat_min,u_k,t_k,p_y) #mtimes(C_dm,xhat_min) 
    
    # filter gain
    K = (solve((mtimes(mtimes(C_dm,P_min),C_dm.T) + R).T,(mtimes(P_min,C_dm.T)).T)).T
    
    # estimated covariance of the state: P(k|k)
    P_corr = mtimes(DM.eye(A_dm.shape[0]) - mtimes(K,C_dm), P_min)
    
    # estimation error
    e_k = y_act - yhat
    
    # estimated mean of the state: x(k|k) 
    xhat_corr = DM(xhat_min + mtimes(K, e_k))
        
    # predicted covariance of the state: P(k+1|k) 
    P_plus = mtimes(mtimes(A_dm, P_corr),A_dm.T) + Q
    
    return [P_plus, P_corr, xhat_corr]
    
def ekf(Fx,Fy,y_act,u_k,Q,R,P_min,xhat_min,ts,t_k,p_y,p_x):
    """
    SUMMARY:
    Discrete-time extended Kalman filter for the given nonlinear system.
    
    SYNTAX:
    assignment = ekf(Fx_model,Fy_model,y_act,u,Q,R,P_min,xhat_min,h)
  
    ARGUMENTS:
    + Fx - State correlation function  
    + Fy - Output correlation function    
    + Q - Process noise covariance matrix
    + R - Measurements noise covariance matrix
    + y_act - Measurements, i.e. y(k)
    + u - Input, i.e. u(k)
    + P_min - Predicted covariance of the state, i.e. P(k|k-1)
    + xhat_min - Predicted mean of the state, i.e. x(k|k-1)
    + ts - Time step
    + t_k - Current time index

        
    OUTPUTS:
    + P_plus - Predicted covariance of the state, i.e. P(k+1|k)
    + P_corr - Estimated covariance of the state, i.e. P(k|k) 
    + xhat_corr - Estimated mean of the state, i.e. x(k|k) 
    """    
    # predicted output: y(k|k-1) 
    yhat = Fy(xhat_min,u_k,t_k,p_y) 
    
    # get linearization of measurements
    Fun_in = SX.get_input(Fy)
    C_dm = jacobian(Fy.call(Fun_in)[0], Fun_in[0])  # jac Fy_x
    C = Function('C', [Fun_in[0],Fun_in[1],Fun_in[2],Fun_in[3]], [C_dm])
    
    # filter gain 
    C_k = C(xhat_min,u_k,t_k,p_y)
    
    C_k = C_k.full()
    P_min = P_min.full() if hasattr(P_min, 'full') else P_min
    R = R.full() if hasattr(R, 'full') else R
    
    inbrackets = scla.inv(np.linalg.multi_dot([C_k,P_min,C_k.T]) + R)
    K_k = np.linalg.multi_dot([P_min,C_k.T,inbrackets])
    
    # estimated covariance of the state: P(k|k)
    P_corr = P_min - np.linalg.multi_dot([K_k,C_k,P_min])
    
   # estimation error
    e_k = y_act - yhat
    
    e_k = e_k.full()
    xhat_min = xhat_min.full()
    
    # estimated mean of the state: x(k|k)
    xhat_corr = xhat_min + np.dot(K_k, e_k)
        
    # get linearization of states
    Fun_in = SX.get_input(Fx)
    
    jac_Fx = jacobian(Fx.call(Fun_in)[0],Fun_in[0])
    A = Function('A', [Fun_in[0],Fun_in[1],Fun_in[2],Fun_in[3],Fun_in[4]], [jac_Fx])
    
    # next predicted covariance of the state: P(k+1|k) 
    A_k = A(xhat_corr,u_k,ts,t_k,p_x)
    
    A_k = A_k.full()
    Q = Q.full() if hasattr(Q, 'full') else Q
    
    P_plus = np.linalg.multi_dot([A_k,P_corr,A_k.T]) + Q
    
    P_plus = DM(P_plus)
    xhat_corr = DM(xhat_corr)
    
    return [P_plus, P_corr, xhat_corr]
    
def mhe(Fx,Fy,y_act,u_k,P_k,xhat_min,F_obj,ts,t_k,px,py,U,Y,T,Xmin,X,V,W,w_k,v_k,x_bar,\
        N,mhe_up,N_mhe, solver, w_lb, w_ub, g_lb, g_ub,\
        bigC, bigG, bigA, bigB, bigf, bigh, bigQk, bigRk, bigSk, bigQ, bigU,\
        bigP, bigPc,P_k_kal,P_corr_kal,Hbig,Obig_r,Pycondx_inv_r,xm_kal,PX,PY,nd):
    
    """
    SUMMARY:
    Moving horizon estimation method for the given nonlinear system.
    
    SYNTAX:
    assignment = mhe(Fx,Fy,y_act,u_k,P_k,xhat_min,F_obj,ts,t_k,px,py,U,Y,T,Xmin,X,V,W,w_k,v_k,x_bar,\
        N,mhe_up,N_mhe, solver, w_lb, w_ub, g_lb, g_ub,\
        bigC, bigG, bigA, bigB, bigf, bigh, bigQk, bigRk, bigSk, bigQ, bigU,\
        bigP, bigPc,P_k_kal,P_corr_kal,Hbig,Obig_r,Pycondx_inv_r,xm_kal,PX,PY)
  
    ARGUMENTS:
    + Fx - State correlation function  
    + Fy - Output correlation function
    + y_act - Measurements, i.e. y(k)
    + u_k - Input, i.e. u(k)
    + P_k - Predicted covariance of the state, i.e. P(k|k-1) 
    + xhat_min - Predicted mean of the state, i.e. x(k|k-1)
    + F_obj - MHE problem objective function
    + ts - Time step
    + t_k - Current time index
    + px,py - Measurable parameters
    + U,Y,T,Xmin,X,V,W,DXM,DYM - Data vectors for inputs, measurements, time indeces, state, noises and measurable disturbances
    + w_k,v_k - Current process and measurement noises
    + x_bar - A priori state estimate
    + N - Growing MHE horizon length (once N = N_mhe it does not change anymore) 
    + mhe_up - Updating prior weight method (choose between "filter" or "smooth")
    + N_mhe - MHE horizon length
    + solver, w_lb, w_ub, g_lb, g_ub - MHE optimization problem definition 
    + bigC, bigG, bigA, bigB, bigf, bigh, bigQk, bigRk, bigSk, bigQ, bigU - System matrices/vectors used for smoothing update
    + bigP, bigPc,P_k_kal,P_corr_kal, xm_kal - Kalman filter quantities used for smoothing update
    + Hbig,Obig_r,Pycondx_inv_r - Inverse matrix values for prior weight calculation in MHE problem in case of smoothing update
    

        
    OUTPUTS:
    + P_plus - Predicted covariance of the state, i.e. P(k+1|k)
    + P_corr - Estimated covariance of the state, i.e. P(k|k) 
    + xhat_corr - Estimated mean of the state, i.e. x(k|k) 
    + U,Y,T,Xmin,X,V,W,DXM,DYM  - Data vectors for inputs, measurements, time indeces, state, noises and measurable disturbances
    + bigC, bigG, bigA, bigB, bigf, bigh, bigQk, bigRk, bigSk, bigQ, bigU - System matrices/vectors used for smoothing update
    + bigP, bigPc,P_k_kal,P_corr_kal, xm_kal - Kalman filter quantities used for smoothing update
    + Hbig,Obig_r,Pycondx_inv_r - Inverse matrix values for prior weight calculation in MHE problem in case of smoothing update
    
    """  
    
    ksim = int(round(old_div(t_k,ts)))
    n = xhat_min.size1()
    m = u_k.size1()
    p = y_act.size1()
    npx = px.size1()
    npy = py.size1()
    
   
    # get linearization of measurements
    Fun_in = SX.get_input(Fy)
    C_dm = jacobian(Fy.call(Fun_in)[0], Fun_in[0])  # jac Fy_x
    C = Function('C', [Fun_in[0],Fun_in[1],Fun_in[2],Fun_in[3]], [C_dm])
    
    # get linearization of states
    Fun_in = SX.get_input(Fx)
    A_dm = jacobian(Fx.call(Fun_in)[0],Fun_in[0]) # jac Fx_x
    A = Function('A', [Fun_in[0],Fun_in[1],Fun_in[2],Fun_in[3],Fun_in[4],Fun_in[5]], [A_dm])
    
    B_dm = jacobian(Fx.call(Fun_in)[0],Fun_in[1]) # jac Fx_u
    B = Function('B', [Fun_in[0],Fun_in[1],Fun_in[2],Fun_in[3],Fun_in[4],Fun_in[5]], [B_dm])
    
    G_dm = jacobian(Fx.call(Fun_in)[0],Fun_in[4]) # jac Fx_w
    G = Function('G', [Fun_in[0],Fun_in[1],Fun_in[2],Fun_in[3],Fun_in[4],Fun_in[5]], [G_dm])
    
    n_w = G_dm.size2() #get w dimension
    
    nxv = n+p 
    nxvw = nxv + n_w
    n_opt = N*nxvw + n  # total # of variables   
    
    # get linearization of objective function
    Fun_in = SX.get_input(F_obj)
    FobjIn = vertcat(Fun_in[0],Fun_in[1])
    [H_dm,_] = hessian(F_obj.call(Fun_in)[0], FobjIn)
    H = Function('H', [Fun_in[0],Fun_in[1],Fun_in[2]], [H_dm])
    
    ## Stacking data
    if ksim < N_mhe:
        if ksim == 0:
            U = vertcat(U,u_k)
        else:
            U = vertcat(U,u_k,u_k) #doubling u_k to maintain the lenght for U (last component is fictuous)
        Y = vertcat(Y,y_act)
        T = vertcat(T,t_k)
        Xmin = vertcat(Xmin,xhat_min)
        Yold = Y
        PX = vertcat(PX,px)
        PY = vertcat(PY,py)
    else:
        if N_mhe == 1:
            U = u_k
            Y = y_act
            T = t_k
            Xmin = xhat_min
            PX = px
            PY = py
        else:
            Yold = Y
            U = vertcat(U[m:],u_k,u_k) #doubling u_k to maintain the lenght for U (last component is fictuous)
            Y = vertcat(Y[p:],y_act)
            T = vertcat(T[1:],t_k)
            Xmin = vertcat(Xmin[n:],xhat_min)
            PX = vertcat(PX[npx:],px)
            PY = vertcat(PY[npy:],py)
        
    ## Initial guess (on the first NLP run)
    w_guess = DM.zeros(n_opt)
    for key in range(N):
        if key == 0: 
            w_guess[key*nxvw:key*nxvw+n] = x_bar
        else:
            w_guess[key*nxvw:key*nxvw+n] = Fx(w_guess[(key-1)*nxvw:(key-1)*nxvw+n],U[(key-1)*m:(key-1)*m+m],ts,T[key-1],np.zeros((n_w,1)),PX[(key-1)*(npx):(npx)*key])
        w_guess[key*nxvw+n:key*nxvw+nxv] = np.zeros((p,1))#v_k
        w_guess[key*nxvw+nxv:(key+1)*nxvw] = np.zeros((n_w,1))#w_k
    w_guess[N*nxvw:N*nxvw+n] = Fx(w_guess[key*nxvw:key*nxvw+n],U[key*m:key*m+m],ts,T[key],np.zeros((n_w,1)),PX[key*(npx):(npx)*(key+1)])#xhat_min  #x_N

   
    ## Inverting P_k matrix for optimization solver
    pk = P_k.full() if hasattr(P_k, 'full') else P_k
    P_k_inv = scla.inv(pk)
       
    P_k_inv_r = DM(P_k_inv).reshape((n*n,1)) #The DM is needed to avoid error in reshape inside solver definition
    
    ## Set parameter for dynamic optimisation
    par = vertcat(U,Y,x_bar,P_k_inv_r,T,Pycondx_inv_r,Hbig,Obig_r,PX,PY)

    # Optimization problem
    sol = solver(lbx = w_lb,
                 ubx = w_ub,
                 x0  = w_guess,
                 p = par,
                 lbg = g_lb,
                 ubg = g_ub)

    w_opt = sol["x"]
    xkp1k = w_opt[-n:] 
    xhat_corr = w_opt[-n-nxvw:-nxvw] 
    
    v_k = w_opt[-nxvw:-n-n_w] 
    if ksim != 0 and N_mhe != 1:
        w_k = w_opt[-n-n_w:-n] 

    
    ## Stacking data
    if ksim < N_mhe:
        X = vertcat(X,xkp1k)
        V = vertcat(V,v_k)
        W = vertcat(W,w_k)
        
    else:
        if N_mhe == 1:
            X = xkp1k
            V = v_k
            W = w_k          
        else:
            X = vertcat(X[n:],xkp1k)
            V = vertcat(V[p:],v_k)
            W = vertcat(W[n_w:],w_k)
            
    
    if mhe_up == 'smooth' or mhe_up == 'filter':
        H_k = scla.inv(H(w_k,v_k,t_k).full())    
        Q_k = H_k[0:n_w,0:n_w]
        R_k = H_k[-p:,-p:]
        S_k = H_k[0:n_w,-p:]
        
        
        R = Function('R', [Fun_in[0],Fun_in[1],Fun_in[2]], [inv(H_dm[-p:,-p:])])
        R_kk = R(w_k,v_k,t_k).full() 
        
        C_k = C(xhat_corr,u_k,t_k,py).full()
        h_k = Y[-p:] - np.dot(C_k,xhat_corr) - v_k 
        A_k = A(xhat_corr,u_k,ts,t_k,w_k,px).full()
        B_k = B(xhat_corr,u_k,ts,t_k,w_k,px).full()
        G_k = G(xhat_corr,u_k,ts,t_k,w_k,px).full()
        f_k = xkp1k - np.dot(A_k,xhat_corr) - np.dot(B_k,u_k) - np.dot(G_k,w_k) 
        
        
        # Bulding the Kalman Filter covariance
        inbrackets = scla.inv(np.linalg.multi_dot([C_k,P_k_kal,C_k.T]) + R_k)
        K_k = np.linalg.multi_dot([P_k_kal,C_k.T,inbrackets])
        
        # estimated covariance of the state: P(k|k)
        P_corr_kal = P_k_kal - np.linalg.multi_dot([K_k,C_k,P_k_kal])
        
        # storing the right value of P_k_kal
        Pi = P_k_kal
        
        # predicted output: y(k|k-1) 
        yhat = Fy(xm_kal,u_k,t_k,py) 
    
        # estimation error
        e_k = y_act - yhat
        
        e_k = e_k.full()
        xm_kal = xm_kal.full()
        
        # estimated mean of the state: x(k|k) 
        xc_kal = xm_kal + np.dot(K_k, e_k)
        
        # estimated mean of the state: x(k+1|k) 
        xm_kal = Fx(xc_kal,u_k,ts,t_k,w_k,px)
        
        # next predicted covariance of the state: P(k+1|k) 
        M_k = np.dot(-K_k,S_k.T)
        
        P_k_kal = np.linalg.multi_dot([A_k,P_corr_kal,A_k.T]) + \
                  np.linalg.multi_dot([G_k,Q_k,G_k.T]) + \
                  np.linalg.multi_dot([A_k,M_k,G_k.T]) + \
                  np.linalg.multi_dot([G_k,M_k,A_k.T])
        
        # Storing data for covariance update
        bigC.append(C_k)
        bigG.append(G_k)
        bigA.append(A_k)
        bigB.append(B_k)
        bigf.append(f_k)
        bigh.append(h_k)
        bigQk.append(Q_k)
        bigRk.append(R_k)
        bigSk.append(S_k)
        bigQ.append(H_k)
        bigU.append(u_k)
        bigP.append(Pi)
        bigPc.append(P_corr_kal)
        

    # Update prior weight 
    if ksim >= N_mhe-1:
        if mhe_up == 'filter': #Filtering
            # Calculating linearization of objective function    
            
            #################
            H_k = scla.inv(H(W[0:n_w],V[0:p],T[0]).full())  
            Q_k = H_k[0:n_w,0:n_w]
            R_k = H_k[-p:,-p:]
            S_k = H_k[0:n_w,-p:]
            ############################
            C_k = C(Xmin[0:n],U[0:m],T[0],PY[0:npy]).full()
            inbrackets = scla.inv(np.linalg.multi_dot([C_k,P_k,C_k.T]) + R_k)
            K_k = np.linalg.multi_dot([P_k,C_k.T,inbrackets])
            P_corr = P_k - np.linalg.multi_dot([K_k,C_k,P_k])
            
            # next predicted covariance of the state: P(k+1|k) 
            A_k = A(X[0:n],U[0:m],ts,T[0],W[0:n_w],PX[0:npx]).full()
            G_k = G(X[0:n],U[0:m],ts,T[0],W[0:n_w],PX[0:npx]).full()
            #The following terms comes from the correlation between v and w (Feng et al. 2013)
            M_k = np.dot(-K_k,S_k.T)
            
            P_k = np.linalg.multi_dot([A_k,P_corr,A_k.T]) + \
                  np.linalg.multi_dot([G_k,Q_k,G_k.T]) + \
                  np.linalg.multi_dot([A_k,M_k,G_k.T]) + \
                  np.linalg.multi_dot([G_k,M_k,A_k.T])
                        
        else: #Smoothing
       
            ## Backward Riccati iteration for smoothed covariances.
            Pisl = np.zeros((n*(N_mhe),n))
            Pis = [Pisl[n*k:n*(k+1),:] for k in range(N_mhe)]
            Pis[N_mhe-1] = bigPc[N_mhe-1]
            for i in range(N_mhe-2, -1, -1):
                Piminv = scla.inv(bigP[i+1])
                
                Pis[i] = bigPc[i] + np.linalg.multi_dot([bigPc[i],\
                   bigA[i].T,Piminv,(Pis[i+1] - bigP[i+1]),Piminv,bigA[i],bigPc[i]])
                
            P_k = Pis[1]
            
            ############ Running again to build the Pycondx matrix #############
            nvars = n + (N_mhe-2)*n_w + (N_mhe-1)*p #Sequence [x_{k-N_T}, w_{k-N_T}, v_{k-N_T}, ..., w_{k-1}, v_{k-1},v_{k}] #changed
            idx = N_mhe-1
           
            
            ## Shifting one step forward
            bigC = bigC[1:]
            bigG = bigG[1:]
            bigA = bigA[1:]
            bigB = bigB[1:]
            bigf = bigf[1:]
            bigh = bigh[1:]
            bigQk = bigQk[1:]
            bigRk = bigRk[1:]
            bigSk = bigSk[1:]
            bigQ = bigQ[1:]
            bigU = bigU[1:]
            bigP = bigP[1:]
            bigPc = bigPc[1:]
            
            ## Building Abig, Cbig, Hbig, Qbig for the smoothing problem
            Qbig = P_k
            Hbig = np.zeros((p*(idx),1))
            Abig = np.zeros((n*(idx), nvars))
            Arow = np.eye(n)
            Abig[0:n,0:n] = Arow      
            Cbig = np.zeros((p*(idx), nvars))
            
            Cbig[0:p, 0:(n+n_w+p)]  = np.column_stack([bigC[0], np.zeros((p,n_w)), np.eye(p)])
            Hbig[:p,:]  = bigh[0] 

            for i in range((N_mhe-2)):
                if i == 0:
                    Apad = np.zeros((n,0))
                else:
                    Apad = np.zeros((n,p))
                Arow = np.column_stack([np.dot(bigA[i],Arow), Apad , bigG[i]])
                    
                Abig[(i+1)*n:(i+2)*n, 0:Arow.shape[1]] = Arow;
                
                if i == N_mhe-3:
                    Cpad = np.zeros((p,p))
                else:
                    Cpad = np.zeros((p, n_w+p))
                
                Crow = np.column_stack([np.dot(bigC[i+1],Arow), Cpad, np.eye(p)])
                Cbig[(i+1)*p:(i+2)*p, 0:Crow.shape[1]] = Crow
                
                Qbig = scla.block_diag(Qbig, bigQ[i])
                
                if i == 0: 
                    Hrow = np.dot(bigB[i], bigU[i]) + bigf[i]
                else:
                    Hrow = np.dot(bigA[i],Hrow) + np.dot(bigB[i], bigU[i]) + bigf[i]
                Hbig[(i+1)*p:(i+2)*p, :]  = np.dot(bigC[i+1],Hrow) + bigh[i+1] 
            
            # Adding the last component of Qbig
            Qbig = scla.block_diag(Qbig, R_kk)

            
            Obig = Cbig[:,0:n]
            Gbig = Cbig[:,n:]
            
            QRbig = Qbig[n:,n:]
            
            Pycondx = np.linalg.multi_dot([Gbig,QRbig,Gbig.T])
            
            Obig_r = DM(Obig).reshape((p*(idx)*n,1))
            Pycondx_inv = scla.inv(Pycondx)
            Pycondx_inv_r = DM(Pycondx_inv).reshape((p*(idx)*p*(idx),1))
            
            
    ### x_bar updating
    if ksim >= N_mhe-1: #permitted only after packing enough information    
        if mhe_up == 'filter': #Filtering
            if N_mhe == 1:
                x_bar = X
                v_bar = V
                w_bar = W
            else:
                x_bar = X[0:n]
                v_bar = V[0:p]
                w_bar = W[0:n_w]
        else: # Smoothing: picking the second component of the optimization sequence
            if N_mhe == 1:
                x_bar = w_opt[:n]
                v_bar = w_opt[n:]
                w_bar = w_k
            else:
                x_bar = w_opt[nxvw:nxvw+n]
                v_bar = w_opt[nxvw+n:nxvw+nxv]
                w_bar = w_opt[nxvw+nxv:2*nxvw]
                
                
    # Eliminating the last fictuous component
    if ksim == 0:
        U = []
    else:
        U = U[:-m] 
        
        
    return [P_k, xhat_corr, w_k,v_k,U,Y,T,Xmin,X,V,W,x_bar, bigC ,bigG,\
            bigA, bigB, bigf, bigh, bigQk, bigRk, bigSk, bigQ, bigU,\
            bigP, bigPc,P_k_kal,P_corr_kal,Hbig,Obig_r,Pycondx_inv_r,xm_kal,xc_kal,PX,PY]