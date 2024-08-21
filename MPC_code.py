# -*- coding: utf-8 -*-
"""
Created on December 3, 2015

@author: Marco, Mirco, Gabriele

MPC code main file
"""
from builtins import range
from casadi import *
from casadi.tools import *
from matplotlib import pylab as plt
import math
import scipy.linalg as scla
import scipy.optimize as scopt
from scipy.integrate import odeint
import numpy as np
import time 
from Utilities import*
from Target_Calc import *
from Estimator import *
from Control_Calc import *
from Default_Values import *

ex_name = __import__('Ex_LMPC_WB') # Insert here your file name
import sys
sys.modules['ex_name'] = ex_name
from ex_name import * #Loading example

########## Dimensions #############################
nx = x.size1()  # model state size                #
nxp = xp.size1()# process state size              #
nu = u.size1()  # control size                    #
ny = y.size1()  # measured output/disturbance size#
nd = d.size1()  # disturbance size                #
if LinPar == False:
    npx = px.size1() # model state parameters size
    npy = py.size1() # model output parameters size
    if Fp_nominal == True:
        npxp = npx  # process state parameters size
        npyp = npy  # process output parameters size
    else:
        npxp = pxp.size1() # process state parameters size
        npyp = pyp.size1() # process output parameters size  
else:
    npx = nx
    npxp = nxp
    npy = npyp = ny                                           #
nxu = nx + nu # state+control                     #
nxuy = nx + nu + ny # state+control               #
nxuk = nx + nu + 2*nx # state + control + internal states
nw = nx*(N+1) + nu*N # total of variables         #
nw_c = nx*(N+1) + nu*N + 2*nx*N # total of variables for collocation methods

if 'Ws' in locals() and slacks == True:
    Ws = Ws
    ns = Ws.shape[0]
else:
    Ws = []
    ns = 0
    
###################################################

########## Fixed symbolic variables #########################################
k = SX.sym("k", 1)  # step integration                                      #
t = SX.sym("t", 1)  # time                                                  #
pxp = SX.sym('pxp',npxp) #process state parameters
pyp = SX.sym('pyp',npyp) #process output parameters
px = SX.sym('px',npx) #model state parameters
py = SX.sym('py',npy) #model output parameter
pxmp = SX.sym('pxmp',npxp)  #process measurable state parameters
pymp = SX.sym('pymp',npyp)  #process measurable output parameters
xs = SX.sym('xs',nx) # stationary state value                               #  
xps = SX.sym('xps',nx) # process stationary state value                     #  
us = SX.sym('us',nu) # stationary input value                               #
ys = SX.sym('ys',ny) # stationary output value                              #
xsp = SX.sym('xsp',nx) # state setpoint value                               #  
usp = SX.sym('usp',nu) # input setpoint value                               #
ysp = SX.sym('ysp',ny) # output setpoint value                              #
lambdaT = SX.sym('lambdaTSX',(ny,nu)) # modifier                            #
s_Coll = SX.sym('s_Coll',2*nx) #internal states for collocation method  
#############################################################################

if ssjacid is True:
    from SS_JAC_ID import *
    [A, B, C, xlin, ulin, ylin] = ss_p_jac_id(ex_name, nx, nu, ny, nd, k, t)
    # Build linearized model.
    if offree == "lin":
        [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, Bd = Bd, Cd = Cd, A = A, B = B, C = C, xlin = xlin, ulin = ulin, ylin = ylin)
    else:
        [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, A = A, B = B, C = C, xlin = xlin, ulin = ulin, ylin = ylin)
else:
    #### Model calculation  #####################################################
    if 'User_fxm_Cont' in locals():
        if StateFeedback is True:
            if offree == "lin":
                [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, Bd = Bd, Cd = Cd, fx = User_fxm_Cont, Mx = Mx, SF = StateFeedback)
            else:
                [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, fx = User_fxm_Cont, Mx = Mx, SF = StateFeedback)
        else:
            if 'User_fym' in locals():
                if offree == "lin":
                    [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, Bd = Bd, Cd = Cd, fx = User_fxm_Cont, Mx = Mx, fy = User_fym)
                else:
                    [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, fx = User_fxm_Cont, Mx = Mx, fy = User_fym)
            else:
                if offree == "lin":
                    [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, Bd = Bd, Cd = Cd, fx = User_fxm_Cont, Mx = Mx, C = C)
                else:
                    [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, fx = User_fxm_Cont, Mx = Mx, C = C)
    elif 'User_fxm_Dis' in locals():
        if StateFeedback is True:
            if offree == "lin":
                [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, Bd = Bd, Cd = Cd, Fx = User_fxm_Dis, SF = StateFeedback)
            else:
                [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, Fx = User_fxm_Dis, SF = StateFeedback)
        else:
            if 'User_fym' in locals():
                if offree == "lin":
                    [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, Bd = Bd, Cd = Cd, Fx = User_fxm_Dis, fy = User_fym)
                else:
                    [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, Fx = User_fxm_Dis, fy = User_fym)
            else:
                if offree == "lin":
                    [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, Bd = Bd, Cd = Cd, Fx = User_fxm_Dis, C = C)
                else:
                    [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, Fx = User_fxm_Dis, C = C)
    elif 'A' in locals():
        if StateFeedback is True:
            if offree == "lin":
                [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, Bd = Bd, Cd = Cd, A = A, B = B, SF = StateFeedback)
            else:
                [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, A = A, B = B, SF = StateFeedback)
        else:
            if 'User_fym' in locals():
                if 'xlin' in locals():
                    if offree == "lin":
                        [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, Bd = Bd, Cd = Cd, A = A, B = B, fy = User_fym, xlin = xlin, ulin = ulin)
                    else:
                        [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, A = A, B = B, fy = User_fym, xlin = xlin, ulin = ulin)
                else:
                    if offree == "lin":
                        [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, Bd = Bd, Cd = Cd, A = A, B = B, fy = User_fym)
                    else:
                        [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, A = A, B = B, fy = User_fym)
            else:
                if 'ylin' in locals():
                    if 'xlin' in locals():
                        if offree == "lin":
                            [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, Bd = Bd, Cd = Cd, A = A, B = B, C = C, xlin = xlin, ulin = ulin, ylin = ylin)
                        else:
                            [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, A = A, B = B, C = C, xlin = xlin, ulin = ulin, ylin = ylin)
                    else:
                        if offree == "lin":
                            [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, Bd = Bd, Cd = Cd, A = A, B = B, C = C, ylin = ylin)
                        else:        
                            [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, A = A, B = B, C = C, ylin = ylin)
                elif 'xlin' in locals():
                        if offree == "lin":
                            [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, Bd = Bd, Cd = Cd, A = A, B = B, C = C, xlin = xlin, ulin = ulin)
                        else:
                            [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, A = A, B = B, C = C, xlin = xlin, ulin = ulin)
                else:
                    if offree == "lin":
                        [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, Bd = Bd, Cd = Cd, A = A, B = B, C = C)
                    else:
                        [Fx_model,Fy_model] = defF_model(x,u,y,d,k,t,px,py,offree,LinPar, A = A, B = B, C = C)

#############################################################################

#### Plant equation  ########################################################
if Fp_nominal is True:
    Fx_p = Fx_model
    Fy_p = Fy_model
else:
    if 'Ap' in locals():
        if StateFeedback is True: # x = Ax + Bu ; y = x 
            [Fx_p,Fy_p] = defF_p(xp,u,y,k,t,pxp,pyp,pxmp,pymp,LinPar, Ap = Ap, Bp = Bp, SF = StateFeedback)
        elif 'User_fyp' in locals(): # x = Ax + Bu ; y = h(x,t)  
            [Fx_p,Fy_p] = defF_p(xp,u,y,k,t,pxp,pyp,pxmp,pymp,LinPar, Ap = Ap, Bp = Bp, fyp = User_fyp)
        else:
            [Fx_p,Fy_p] = defF_p(xp,u,y,k,t,pxp,pyp,pxmp,pymp,LinPar, Ap = Ap, Bp = Bp, Cp = Cp) 
    elif 'User_fxp_Dis' in locals(): 
        if StateFeedback is True: # x = F(x,t) ; y = x
            [Fx_p,Fy_p] = defF_p(xp,u,y,k,t,pxp,pyp,pxmp,pymp,LinPar, Fx = User_fxp_Dis, SF = StateFeedback)
        elif 'User_fyp' in locals():   # x = F(x,t) ; y = h(x,t)          
            [Fx_p,Fy_p] = defF_p(xp,u,y,k,t,pxp,pyp,pxmp,pymp,LinPar, Fx = User_fxp_Dis, fy = User_fyp) 
        elif 'Cp' in locals(): # x = F(x,t)  ; y = Cx 
            [Fx_p,Fy_p] = defF_p(xp,u,y,k,t,pxp,pyp,pxmp,pymp,LinPar, Fx = User_fxp_Dis, Cp = Cp)   
    elif 'User_fxp_Cont' in locals(): 
        if StateFeedback is True: # \dot{x} = f(x,t) ; y = x
            [Fx_p,Fy_p] = defF_p(xp,u,y,k,t,pxp,pyp,pxmp,pymp,LinPar, fx = User_fxp_Cont, Mx = Mx, SF = StateFeedback)
        elif 'User_fyp' in locals(): # \dot{x} = f(x,t) ; y = h(x,t) 
            [Fx_p,Fy_p] = defF_p(xp,u,y,k,t,pxp,pyp,pxmp,pymp,LinPar, fx = User_fxp_Cont, Mx = Mx, fy = User_fyp)
        else:  # \dot{x} = f(x,t)  ; y = Cx 
            [Fx_p,Fy_p] = defF_p(xp,u,y,k,t,pxp,pyp,pxmp,pymp,LinPar, fx = User_fxp_Cont, Mx = Mx, Cp = Cp)
        
#############################################################################

if estimating is False:       
    #### Objective function calculation  ########################################
    if 'rss_y' in locals():
        if 'A' in locals() and 'C' in locals():
            Sol_Hess_constss = 'yes'        
        if 'rss_u' in locals():
            Fss_obj = defFss_obj(x, u, y, xsp, usp, ysp, r_y = rss_y, r_u = rss_u)
        else:
            Fss_obj = defFss_obj(x, u, y, xsp, usp, ysp, r_y = rss_y, r_Du = rss_Du)
            DUssForm = True    
    elif 'Qss' in locals():
        QForm_ss = True
        if 'A' in locals() and 'C' in locals():
            Sol_Hess_constss = 'yes'
        if 'Rss' in locals():
            Fss_obj = defFss_obj(x, u, y, xsp, usp, ysp, Q = Qss, R = Rss)
        else:
            Fss_obj = defFss_obj(x, u, y, xsp, usp, ysp, Q = Qss, S = Sss) 
            DUssForm = True
    elif 'User_fssobj' in locals():
        Fss_obj = defFss_obj(x, u, y, xsp, usp, ysp, f_obj = User_fssobj)
        
    if 'r_x' in locals():
        QForm = True
        if 'A' in locals() and 'C' in locals():
            Sol_Hess_constdyn = 'yes'
        if 'r_u' in locals():
            F_obj = defF_obj(x, u, y, xs, us, ys, r_x = r_x, r_u = r_u)
        else:
            F_obj = defF_obj(x, u, y, xs, us, ys, r_x = r_x, r_Du = r_Du)
            DUForm = True
    elif 'Q' in locals():
        QForm = True
        if 'A' in locals() and 'C' in locals():
            Sol_Hess_constdyn = 'yes'
        if 'R' in locals():
            F_obj = defF_obj(x, u, y, xs, us, ys, Q = Q, R = R)
        else:
            F_obj = defF_obj(x, u, y, xs, us, ys, Q = Q, S = S)
            DUForm = True
    elif 'User_fobj_Cont' in locals():
        ContForm = True
        F_obj = defF_obj(x, u, y, xs, us, ys, f_Cont = User_fobj_Cont)
    elif 'User_fobj_Dis' in locals():
        F_obj = defF_obj(x, u, y, xs, us, ys, f_Dis = User_fobj_Dis)
    elif 'User_fobj_Coll' in locals():
        F_obj = defF_obj(x, u, y, xs, us, ys, f_Coll = User_fobj_Coll, s_Coll = s_Coll)

    if 'User_vfin' in locals():
        Vfin = defVfin(x, xs, vfin_F = User_vfin)
    elif 'A' in locals():    # Linear system
        if 'Q' in locals():      # QP problem
            #Checking S and R matrix for Riccati Equation
            if 'S' in locals():
                R = S
            Vfin = defVfin(x, xs, A = A, B = B, Q = Q, R = R)
    else:
        Vfin = defVfin(x, xs)

    #############################################################################
    
    ### Solver options ##########################################################
    sol_optss = {'ipopt.max_iter':Sol_itmax, 'ipopt.hessian_constant':Sol_Hess_constss,'ipopt.print_level':0,'ipopt.sb':"yes",'print_time':0}#, 'ipopt.tol':1e-10}
    sol_optdyn = {'ipopt.max_iter':Sol_itmax, 'ipopt.hessian_constant':Sol_Hess_constdyn,'ipopt.print_level':0,'ipopt.sb':"yes",'print_time':0}#, 'ipopt.tol':1e-10} 
        
    #### Modifiers Adaptatation gradient definition  ############################
    if Adaptation is True: 
        # Defining eventual new bound contraints when nx != nxp
        if 'xpmin' not in locals(): xpmin = xmin
        if 'xpmax' not in locals(): xpmax = xmax
        
        # Defining the optimization problem to calculate the plant steady state given the input
        (solver_ss_mod, wssp_lb, wssp_ub, gssp_lb, gssp_ub) = opt_ssp(nxp, nu, ny, nd, npx, npy, npxp, Fx_p,Fy_p, sol_optss, xmin = xpmin, xmax = xpmax, h = h)
        
        alpha_mod = 0.2 if 'alpha_mod' not in locals() else alpha_mod
        
        # Defining modifier update filtering euqation
        LambdaT = defLambdaT(xp,x,u,y,d,k,t,pxp,pyp,px,py, Fx_model, Fx_p, Fy_model, Fy_p, alpha_mod)
        
        # Defining auxiliary variable and objective function when nx != nxp
        if nx != nxp:
            xp2 = SX.sym("xp2", nxp)
            Fss_obj2 = Function('Fss_obj2', [xp,u,y,xp2,usp,ysp], [User_fssobj(xp,u,y,xp2,usp,ysp)]) # The economic function has to be non-linear
        else:
            Fss_obj2 = Fss_obj
    
        # Defining the optimization problem to calculate the true plant optimum
        (solver_ss2, wssp2_lb, wssp2_ub, gssp2_lb, gssp2_ub) = opt_ssp2(nxp, nu, ny, nd, npx, npy, npxp, npyp, Fx_p,Fy_p, Fss_obj2, QForm_ss,sol_optss, umin = umin, umax = umax, w_s = None, z_s = None, ymin = ymin, ymax = ymax, xmin = xpmin, xmax = xpmax, h = h)
    #############################################################################
    
    #### Solver definition  #####################################################
    ymin_s = ymin if ymin_ss is None else ymin_ss; ymax_s = ymax if ymax_ss is None else ymax_ss
    xmin_s = xmin if xmin_ss is None else xmin_ss; xmax_s = xmax if xmax_ss is None else xmax_ss
    umin_s = umin if umin_ss is None else umin_ss; umax_s = umax if umax_ss is None else umax_ss
    
    if 'User_g_ineq_SS' not in locals():
        User_g_ineq_SS = None
    if 'User_h_eq_SS' not in locals():
        User_h_eq_SS = None
    
    (solver_ss, wss_lb, wss_ub, gss_lb, gss_ub) = opt_ss(nx, nu, ny, nd, npx,npy, Fx_model,Fy_model, Fss_obj, QForm_ss, DUssForm, sol_optss, User_g_ineq_SS, User_h_eq_SS, umin = umin_s, umax = umax_s, w_s = None, z_s = None, ymin = ymin_s, ymax = ymax_s, xmin = xmin_s, xmax = xmax_s, h = h)
    
    ymin_d = ymin if ymin_dyn is None else ymin_dyn; ymax_d = ymax if ymax_dyn is None else ymax_dyn
    xmin_d = xmin if xmin_dyn is None else xmin_dyn; xmax_d = xmax if xmax_dyn is None else xmax_dyn
    umin_d = umin if umin_dyn is None else umin_dyn; umax_d = umax if umax_dyn is None else umax_dyn
    
    if 'User_g_ineq' not in locals():
        User_g_ineq = None
        ng = 0
    elif 'User_g_ineq' in locals() and slacksG == True: 
        g_ineq = User_g_ineq(x,u,y,d,t,px,py)
        G_ineqSX = Function('G_ineqSX', [x,u,y,d,t,px,py], [g_ineq])
        ng = G_ineqSX(x,u,y,d,t,px,py).size1()
    else:
        ng = 0
        
    if 'User_h_eq' not in locals():
        User_h_eq = None
        nh = 0
    elif 'User_h_eq' in locals() and slacksH == True:
        h_eq = User_h_eq(x,u,y,d,t,px,py)
        H_eqSX = Function('H_eqSX', [x,u,y,d,t,px,py], [h_eq])
        nh = H_eqSX(x,u,y,d,t,px,py).size1()
    else:
        nh = 0
        
    if slacks == True:
        nw = nw + ns
        nw_c = nw_c + ns
    
    if 'User_fobj_Cont' in locals():
        (solver, w_lb, w_ub, g_lb, g_ub) = opt_dyn(x, u, y, d, t, px, py, nx, nu, ny, nd, npx, npy, ng, nh, Fx_model,Fy_model, F_obj,Vfin, N, QForm, DUForm, DUFormEcon, ContForm, TermCons, slacks, slacksG, slacksH,  nw, sol_optdyn, User_g_ineq,  User_h_eq, umin = umin_d, umax = umax_d,  W = None, Z = None, ymin = ymin_d, ymax = ymax_d, xmin = xmin_d, xmax = xmax_d, Dumin = Dumin, Dumax = Dumax, h = h, fx = User_fxm_Cont, xstat = xs, ustat = us, Ws = Ws)
    elif 'User_fobj_Coll' in locals():
        (solver, w_lb, w_ub, g_lb, g_ub) = opt_dyn_CM(x, u, y, d, t, px, py, nx, nu, ny, nd, npx, npy, ng, nh, Fx_model,Fy_model, F_obj,Vfin, N, QForm, DUForm, DUFormEcon, ContForm, TermCons, slacks, slacksG, slacksH,  nw, sol_optdyn, User_g_ineq,  User_h_eq, umin = umin_d, umax = umax_d,  W = None, Z = None, ymin = ymin_d, ymax = ymax_d, xmin = xmin_d, xmax = xmax_d, Dumin = Dumin, Dumax = Dumax, h = h, fx = User_fxm_Cont, xstat = xs, ustat = us, Ws = Ws)
    else:
        (solver, w_lb, w_ub, g_lb, g_ub) = opt_dyn(x, u, y, d, t, px, py, nx, nu, ny, nd, npx, npy, ng, nh, Fx_model,Fy_model, F_obj,Vfin, N, QForm, DUForm, DUFormEcon, ContForm, TermCons, slacks, slacksG, slacksH, nw, sol_optdyn, User_g_ineq,  User_h_eq, umin = umin_d, umax = umax_d,  W = None, Z = None, ymin = ymin_d, ymax = ymax_d, xmin = xmin_d, xmax = xmax_d, Dumin = Dumin, Dumax = Dumax, h = h, Ws = Ws)
    #############################################################################

#### Kalman steady-state gain definition  ###################################
if kalss is True: 
    if 'A' in locals() and 'C' in locals():
        linmod = 'full'    
        if offree == "lin":
            K = Kkalss(ny, nd, nx, Q_kf, R_kf, offree, linmod, Bd = Bd, Cd = Cd, A = A, C = C)
        else:
            K = Kkalss(ny, nd, nx, Q_kf, R_kf, offree, linmod, A = A, C = C)
    elif 'A' in locals():
        linmod = 'onlyA' 
        if offree == "lin":
            K = Kkalss(ny, nd, nx, Q_kf, R_kf, offree, linmod, x, u, k, d, t, h, px, py, x_ss, u_ss, px_ss, py_ss, Bd = Bd, Cd = Cd, A = A, Fy = Fy_model)
        else:
            K = Kkalss(ny, nd, nx, Q_kf, R_kf, offree, linmod, x, u, k, d, t, h, px, py, x_ss, u_ss, px_ss, py_ss, A = A, Fy = Fy_model)
    elif 'C' in locals():
        linmod = 'onlyC' 
        if offree == "lin":
            K = Kkalss(ny, nd, nx, Q_kf, R_kf, offree, linmod, x, u, k, d, t, h, px, py, x_ss, u_ss, px_ss, py_ss, Bd = Bd, Cd = Cd, C = C, Fx = Fx_model)
        else:
            K = Kkalss(ny, nd, nx, Q_kf, R_kf, offree, linmod, x, u, k, d, t, h, px, py, x_ss, u_ss, px_ss, py_ss, C = C, Fx = Fx_model)
    else:
        linmod = 'no' 
        if offree == "lin":
            K = Kkalss(ny, nd, nx, Q_kf, R_kf, offree, linmod, x, u, k, d, t, h, px, py, x_ss, u_ss, px_ss, py_ss, Bd = Bd, Cd = Cd, Fx = Fx_model, Fy = Fy_model)
        else:
            K = Kkalss(ny, nd, nx, Q_kf, R_kf, offree, linmod, x, u, k, d, t, h, px, py, x_ss, u_ss, px_ss, py_ss, Fx = Fx_model, Fy = Fy_model)

#############################################################################

#### MHE optimization problem definition  ###################################
if mhe is True: 
    n_w = w.size1()     # state noise dimension  
    v = SX.sym("v", ny)  # output noise   

    ## Building MHE objective function      
    if 'r_w' in locals():
        if 'A' in locals() and 'C' in locals():
            Sol_Hess_constmhe = 'yes'
        F_obj_mhe = defF_obj_mhe(w, v, t, r_w = r_w, r_v = r_v)
    elif 'Q_mhe' in locals():
        if 'A' in locals() and 'C' in locals():
            Sol_Hess_constmhe = 'yes'
        F_obj_mhe = defF_obj_mhe(w, v, t, Q = Q_mhe, R = R_mhe)
    elif 'User_fobj_mhe' in locals():
        F_obj_mhe = defF_obj_mhe(w, v, t, f_obj = User_fobj_mhe)
    sol_optmhe = {'ipopt.max_iter':Sol_itmax, 'ipopt.hessian_constant':Sol_Hess_constmhe, 'ipopt.tol':1e-10,'ipopt.print_level':0,'ipopt.sb':"yes",'print_time':0} 

    
    ## Building state dynamics for MHE (modified with w variable)
    G_mhe = G_mhe if 'G_mhe' in locals() else DM.eye(nx+nd)
    if 'User_fx_mhe_Cont' in locals():
        if offree == "lin":
            Fx_mhe = defFx_mhe(x,u,w,d,k,t,px,offree, LinPar, Bd = Bd, fx = User_fx_mhe_Cont, Mx = Mx, G = G_mhe)
        else:
            Fx_mhe = defFx_mhe(x,u,w,d,k,t,px,offree,LinPar, fx = User_fx_mhe_Cont, Mx = Mx, G = G_mhe)
    elif 'User_fx_mhe_Dis' in locals():
        if offree == "lin":
            Fx_mhe = defFx_mhe(x,u,w,d,k,t,px,offree,LinPar, Bd = Bd, Fx = User_fx_mhe_Dis, G = G_mhe)
        else:
            Fx_mhe = defFx_mhe(x,u,w,d,k,t,px,offree,LinPar, Fx = User_fx_mhe_Dis, G = G_mhe)
    
    xmi =  -DM.inf(nx) if xmin is None else xmin    
    dmi =  -DM.inf(nd) if dmin is None else dmin
    xmin_mhe = vertcat(xmi,dmi)
    xma = DM.inf(nx) if xmax is None else xmax
    dma = DM.inf(nd) if dmax is None else dmax
    xmax_mhe = vertcat(xma,dma)
    
    ## Initializing iterative variables
    w_k = DM.zeros(n_w,1)
    v_k = DM.zeros(ny,1)
    U_mhe = []
    X_mhe = []
    Xm_mhe = []
    Y_mhe = []
    T_mhe = []
    V_mhe = []
    W_mhe = []
    
    C_mhe = []
    G_mhe = []
    A_mhe = []
    B_mhe = []
    f_mhe = []
    h_mhe = []
    Qk_mhe = []
    Rk_mhe = []
    Sk_mhe = []
    Q_mhe = []
    bigU_mhe = []
    P_mhe = []
    Pc_mhe = []
    P_kal_mhe = P0
    Pc_kal_mhe = P0
    PX_mhe = []
    PY_mhe = []
    
    idx = N_mhe if N_mhe == 1 else N_mhe-1   
    pH_mhe = DM.zeros(ny*(idx),1)
    pO_mhe = DM.zeros(ny*(idx)*(nx+nd),1)
    pPyx_mhe = DM.zeros(ny*(idx)*ny*(idx),1) 
    
    
#############################################################################        
p_xp = DM.zeros((npxp,1)) #Dummy variable when parameter is not present
p_yp = DM.zeros((npyp,1)) #Dummy variable when parameter is not present
p_xk = DM.zeros((npx,N)) #Dummy variable when parameter is not present
p_yk = DM.zeros((npy,N)) #Dummy variable when parameter is not present
p_xmp = DM.zeros((npxp,1)) #Dummy variable when parameter is not present
p_ymp = DM.zeros((npyp,1)) #Dummy variable when parameter is not present
ysp_k = DM.zeros(ny,1); usp_k = DM.zeros(nu,1); xsp_k = DM.zeros(nx,1); xsp_k_p = DM.zeros(nxp,1)
x_k = DM(x0_p)
u_k = DM(u0)
xhat_k = DM(x0_m)
lambdaT_k = np.zeros((ny,nu))
cor_k = 0.00*np.ones((ny,1))
delta_k = np.zeros((nu,ny))
try:
    P_k = P0
except NameError:
    P_k = DM.zeros(nx+nd, nx+nd)

try:
    dhat_k = DM(dhat0)  
except NameError:
    dhat_k = DM.zeros(nd)

Xp = []           
Yp = []           
U = []
XS = []
YS = []
US = []
X_HAT = []
Y_HAT = []
D_HAT = []
COR = []
TIME_SS = []
TIME_DYN = []
Upopt = []; Ypopt = []
LAMBDA = []
P_K = []
Esim = []
X_KF = []
Ysp = []
Sl = []

for ksim in range(Nsim):
    print('Time Iteration ' + str(ksim+1) + ' of ' + str(Nsim))
    t_k = ksim*h #real time updating 
    
    ## Updating the current value of parameters
    
    # Defining variable parameters along the predicition horizon
    if 'def_px' in locals():
        for i in range (N):
            [p_xk[:,i]] = def_px(t_k+i)
    if 'def_py' in locals():
        for i in range (N):
            [p_yk[:,i]] = def_py(t_k+i)
    
    # Taking the first vector of parameters for the current ksim
    p_x_k = p_xk[:,0] 
    p_y_k = p_yk[:,0]        
 
    if 'def_px' in locals() and'def_pxmp' in locals():
        [p_xmp] = def_pxmp(t_k)
    elif 'def_px' in locals():
        p_xmp = p_x_k
    if 'def_py' in locals() and 'def_pymp' in locals():
        [p_ymp] = def_pymp(t_k)
    elif 'def_py' in locals():
        p_ymp = p_y_k
     
    if 'def_pxp' in locals():
        [p_xp] = def_pxp(t_k)
    if 'def_pyp' in locals():
        [p_yp] = def_pyp(t_k)
    
            
    ## Store current state
    Xp.append(x_k)
    X_HAT.append(xhat_k)
    
    
    # Model output
    yhat_k = Fy_model(xhat_k, u_k, dhat_k, t_k, p_y_k)
    
    if (ksim==0):
            y_k = yhat_k
    y_k_prev = y_k
    
    # Actual output 
    if Fp_nominal is True:
        y_k = Fy_p(x_k, u_k,  dhat_k, t_k, p_y_k)
    else: #All the other cases
        y_k = Fy_p(x_k,u_k,p_yp,t_k,p_ymp)         
    
        
    # Introducing white noise on output when present
    if 'R_wn' in locals():
        Rv = scla.sqrtm(R_wn)
        v_wn_k = mtimes(Rv,DM(np.random.normal(0,1,ny)))
        y_k = y_k + v_wn_k
        
    Yp.append(y_k)
    Y_HAT.append(yhat_k)
################ Estimator calling ############################################
    if offree != "no":
        x_es = vertcat(xhat_k,dhat_k)
        
        csi = SX.sym("csi", nx+nd)
        x1 = csi[0:nx]
        d1 = csi[nx:nx+nd]        
        Dx = Function('Dx',[d],[d])
        
        if mhe is False:    
            dummyFx = vertcat(Fx_model(x1,u,k,d1,t,px), Dx(d1))
            Fx_es = Function('Fx_es', [csi,u,k,t,px], [dummyFx])
        else:
            Fx_es = Fx_mhe
            
        dummyFy = Fy_model(x1,u,d1,t,py)
        Fy_es = Function('Fy_es', [csi,u,t,py], [dummyFy])
        
    else:
        if nd != 0.0:
            import sys
            sys.exit("The disturbance dimension is not zero but no disturbance model has been selected")
        x_es = xhat_k
        if mhe is False:
            dummyFx = Fx_model(x,u,k,d,t,px)
            Fx_es = Function('Fx_es', [x,u,k,t,px], [dummyFx])
        else:
            dummyFx = Fx_mhe(x,u,k,t,w,px)
            Fx_es = Function('Fx_es', [x,u,k,t,w,px], [dummyFx])
        dummyFy = Fy_model(x,u,d,t,py)        
        Fy_es = Function('Fy_es', [x,u,t,py], [dummyFy])
    
    if kalss is True or lue is True: 
        estype = 'kalss'
        if StateFeedback is True and offree == 'no': 
            K = DM.eye(x_es.size1())
        [x_es, kwargsout] = defEstimator(Fx_es,Fy_es,y_k,u_k, estype, x_es, t_k, p_x_k, p_y_k, K = K)

    elif mhe is True:
        estype = 'mhe'
        
        if ksim == 0:
            xm_kal_mhe = x_es
        
        X_KF.append(DM(xm_kal_mhe))
    
        if ksim < N_mhe: 
        
            N_mhe_curr = ksim + 1
            
                ## Defining the optimization solver
            (solver_mhe, w_lb_mhe, w_ub_mhe, g_lb_mhe, g_ub_mhe) = mhe_opt(nx+nd, nd, nu, ny,npx,npy, n_w, F_obj_mhe, Fx_es,Fy_es, \
            N_mhe_curr, N_mhe, ksim, h, mhe_up, sol_optmhe, User_g_ineq,  User_h_eq, wmin = wmin, wmax = wmax,  vmin = vmin, vmax = vmax, \
            ymin = ymin, ymax = ymax, xmin = xmin_mhe, xmax = xmax_mhe)
        
        
        [x_es, kwargsout] = defEstimator(Fx_es,Fy_es,y_k,u_k, estype, x_es, t_k, p_x_k, p_y_k, P_min = P_k, Fobj = F_obj_mhe,\
        ts = h, wk = w_k, vk = v_k, U = U_mhe, X = X_mhe, Xm = Xm_mhe, Y = Y_mhe, T = T_mhe, V = V_mhe, W = W_mhe, xb = x_bar,\
        sol = solver_mhe, solwlb = w_lb_mhe, solwub = w_ub_mhe, solglb = g_lb_mhe, solgub = g_ub_mhe,\
        N = N_mhe_curr, up = mhe_up, Nmhe = N_mhe, C = C_mhe, G = G_mhe,  A = A_mhe, B = B_mhe, f = f_mhe,h = h_mhe, Qk = Qk_mhe, Rk = Rk_mhe, Sk = Sk_mhe,\
        Q = Q_mhe, bU = bigU_mhe, P = P_mhe, Pc = Pc_mhe, P_kal = P_kal_mhe, P_c_kal = Pc_kal_mhe, pH = pH_mhe, pO = pO_mhe, pPyx = pPyx_mhe, xm_kal = xm_kal_mhe, \
            PX = PX_mhe, PY = PY_mhe, nd = nd) 
        P_k = kwargsout['P_plus']   
        U_mhe = kwargsout['U_mhe']   
        X_mhe = kwargsout['X_mhe']
        Xm_mhe = kwargsout['Xm_mhe']
        Y_mhe = kwargsout['Y_mhe']
        T_mhe = kwargsout['T_mhe']
        V_mhe = kwargsout['V_mhe']   
        W_mhe = kwargsout['W_mhe'] 
        w_k = kwargsout['wk']   
        v_k = kwargsout['vk']
        x_bar = kwargsout['xb']
        C_mhe = kwargsout['C_mhe']
        G_mhe = kwargsout['G_mhe']
        A_mhe = kwargsout['A_mhe']
        B_mhe = kwargsout['B_mhe']
        f_mhe = kwargsout['f_mhe']
        h_mhe = kwargsout['h_mhe']
        Qk_mhe = kwargsout['Qk_mhe']
        Rk_mhe = kwargsout['Rk_mhe']
        Sk_mhe = kwargsout['Sk_mhe']
        Q_mhe = kwargsout['Q_mhe']
        bigU_mhe = kwargsout['bigU_mhe']
        P_mhe = kwargsout['P_mhe']
        Pc_mhe = kwargsout['Pc_mhe']
        P_kal_mhe = kwargsout['P_kal_mhe']
        Pc_kal_mhe = kwargsout['P_c_kal_mhe']
        pH_mhe = kwargsout['pH_mhe']
        pO_mhe = kwargsout['pO_mhe']
        pPyx_mhe = kwargsout['pPyx_mhe']
        xm_kal_mhe = kwargsout['xm_kal_mhe']
        xc_kal_mhe = kwargsout['xc_kal_mhe']
        PX_mhe = kwargsout['PX_mhe']
        PY_mhe = kwargsout['PY_mhe']
        
        P_K.append(P_k)
    else:
        if kal is True: # only for linear system
            if 'A' not in locals():
                import sys
                sys.exit("You cannot use the kalman filter if the model you have chosen is not linear")
            estype = 'kal'
        elif ekf is True: 
            estype = 'ekf'
        [x_es, kwargsout] = defEstimator(Fx_es,Fy_es,y_k,u_k, estype, x_es, t_k, p_x_k, p_y_k, P_min = P_k, Q = Q_kf, R = R_kf, ts = h)
        P_k = kwargsout['P_plus']   
         
   
    # Extracting x(k|k) and d(k|k) 
    if offree != "no":    
        xhat_k = x_es[0:nx]
        dhat_k = x_es[nx:nx+nd]
        
        # dhat_k saturation
        if dmin is not None:
            for k_d in range(nd):
                if dhat_k[k_d] < dmin[k_d]:
                    dhat_k[k_d] = dmin[k_d]
                elif dhat_k[k_d] > dmax[k_d]:
                    dhat_k[k_d] = dmax[k_d]     
    else:
        xhat_k = x_es
    D_HAT.append(dhat_k)    
###############################################################################    
    # Check for feasibile condition
    if np.any(np.isnan(xhat_k.__array__())):
       import sys
       sys.exit("xhat_k has some components that are NaN. This is caused by an error in the integrator. Please check the white-noise or disturbance input: maybe they are too large!")
         
    if estimating is False:  
        
        if 'defSP' in locals():
            ## Setpoint updating    
            [ysp_k, usp_k, xsp_k] = defSP(t_k)
            Ysp.append(ysp_k)
        
        if (ksim==0):
            us_k = u_k
            xs_k = x0_m
        
        uk_prev = u_k
        us_prev = us_k #just a reminder that this must be us_(k-1)
        xs_prev = xs_k #just a reminder that this must be xs_(k-1)
        
        lambdaT_k_r = DM(lambdaT_k).reshape((nu*ny,1)) #shaping lambda matrix in order to enter par_ss
        
        ### Paramenter for Target optimization 
        par_ss = vertcat(usp_k,ysp_k,xsp_k,dhat_k,us_prev,lambdaT_k_r,t_k,p_x_k,p_y_k)    
        
        ## Target calculation initial guess
        wss_guess = DM.zeros(nxuy)
        wss_guess[0:nx] = x0_m
        wss_guess[nx:nxu] = u0
        y0 = Fy_model(x0_m,u0,dhat_k,t_k,p_y_k)
        wss_guess[nxu:nxuy] = y0
        
        ## Solve the Target calculation problem
        start_time = time.time()    
        sol_ss = solver_ss(lbx = wss_lb,
                        ubx = wss_ub,
                        x0  = wss_guess,
                        p = par_ss,
                        lbg = gss_lb,
                        ubg = gss_ub)
                        
        etimess = time.time()-start_time

        ## Check feasibility: reject infeasible solutions
        if solver_ss.stats()['return_status'] != 'Infeasible_Problem_Detected':
            wss_opt = sol_ss["x"]
            xs_k = wss_opt[0:nx]
            us_k = wss_opt[nx:nxu]
            ys_k_opt = wss_opt[nxu:nxuy]
        
        
        if Adaptation is True:
            # Updating correction for modifiers-adaptation method 
            cor_k = mtimes(lambdaT_k,(us_k - us_prev))
            COR.append(cor_k)
        
        # Storing variables for Target calculation
        XS.append(xs_k)
        US.append(us_k)
        TIME_SS.append(etimess)
        ys_k = Fy_model(xs_k,us_k,dhat_k,t_k,p_y_k)
        YS.append(ys_k)
        
        ## Set current state as initial value
        w_lb[0:nx] = w_ub[0:nx] = xhat_k
        
        ## Set current targets 
        cur_tar = vertcat(xs_k,us_k)
        

        if (ksim==0):
            ## Initial guess (on the first OCP run)
            if Collocation == True:
                w_guess = DM.zeros(nw_c)
            else:
                w_guess = DM.zeros(nw)  
                           
            for key in range(1,N+1,1):
                if Collocation == True:
                    w_guess[key*nxuk-nu-2*nx:key*nxuk-nu] = vertcat(x0_m,x0_m) #internal states
                    w_guess[key*nxuk-nu:key*nxuk] = u0
                    w_guess[key*nxuk:key*nxuk+nx] = x0_m
                else:
                    w_guess[key*nxu-nu:key*nxu] = u0
                    w_guess[key*nxu:key*nxu+nx] = x0_m
                
            w_guess[0:nx] = x0_m #x0
        else:
            ## Initial guess (warm start)
            if Collocation == True:
                if solver.stats()['return_status'] != 'Infeasible_Problem_Detected':    
                    w_guess = vertcat(w_opt[nxuk:nw_c-ns],xs_prev,xs_prev,us_prev,xs_prev,w_opt[nw_c-ns:nw_c])
            else:
                if solver.stats()['return_status'] != 'Infeasible_Problem_Detected':
                    w_guess = vertcat(w_opt[nxu:nw-ns],us_prev,xs_prev,w_opt[nw-ns:nw])
   
            
        ## Set parameter for dynamic optimisation
        # Reshaping parameter matrices along the horizon into vectors
        p_xk_r = reshape(p_xk,npx*N,1)
        p_yk_r = reshape(p_yk,npy*N,1)
        
        par = vertcat(xhat_k,cur_tar,dhat_k,u_k,t_k,lambdaT_k_r,p_xk_r,p_yk_r)
        
        ## Solve the OCP
        start_time = time.time()
        sol = solver(lbx = w_lb,
                     ubx = w_ub,
                     x0  = w_guess,
                     p = par,
                     lbg = g_lb,
                     ubg = g_ub)
    
        etime = time.time()-start_time

        ## Check feasibility: reject infeasible solutions
        if solver.stats()['return_status'] != 'Infeasible_Problem_Detected':
            f_opt = sol["f"]
            w_opt = sol["x"]

            ## Get the optimal input and  the Next predicted state (already including the disturbance estimate) i.e.x(k|k-1)
            if Collocation == True:
                u_k = w_opt[nxuk-nu:nxuk]             
                s1_k = w_opt[nxuk-nu-2*nx:nxuk-nu-nx]
                s2_k = w_opt[nxuk-nu-nx:nxuk-nu]              
                xhat_k = w_opt[nxuk:nxuk+nx]             
                sl_k = w_opt[nw_c-ns:nw_c]
            else:
                u_k = w_opt[nxu-nu:nxu]
                xhat_k = w_opt[nxu:nxu+nx]
                sl_k = w_opt[nw-ns:nw]
                
        # Next predicted state (already including the disturbance estimate) 
        # i.e.x(k|k-1)        
        elif solver.stats()['return_status'] == 'Infeasible_Problem_Detected':
             xhat_k = Fx_model(xhat_k, u_k, h, dhat_k, t_k, p_x_k)

        U.append(u_k)
        if slacks == True:
            Sl.append(sl_k)
        TIME_DYN.append(etime)
       
    ############### Updating variables xp and xhat ###########################
    if Fp_nominal is True:
        x_k = Fx_p(x_k, u_k, h, dhat_k, t_k, p_xmp)
    else: # All the other cases
        x_k = Fx_p(x_k, u_k, p_xp, t_k, h, p_xmp)        
        
    # Check for feasibile condition
    if np.any(np.isnan(x_k.__array__())):
        import sys
        sys.exit("x_k has some components that are NaN. This is caused by an error in the integrator. Please check the white-noise or disturbance input: maybe they are too large!")
         
    # Introducing white noise on state when present
    if 'G_wn' in locals():
        Qw = scla.sqrtm(Q_wn)
        w_wn_k = mtimes(Qw,DM(np.random.normal(0,1,nxp)))    
        x_k = x_k + mtimes(G_wn,w_wn_k)
    
    if estimating is False: 
        
        if Adaptation is True:
            par_ssp = vertcat(t_k,us_k,p_xp,p_xmp,p_ymp)  
            
            ## Target process calculation
            wssp_guess = x0_p
         
            sol_ss_mod = solver_ss_mod(lbx = wssp_lb,
                            ubx = wssp_ub,
                            x0  = wssp_guess,
                            p = par_ssp,
                            lbg = gssp_lb,
                            ubg = gssp_ub) 
            
            xs_kp = sol_ss_mod["x"]        
            fss_p = sol_ss_mod["f"] 
    
            ## Evaluate the updated term for the correction
            lambdaT_k = LambdaT(xs_kp,xs_k,us_k,dhat_k,ys_k,h,t_k,p_xp,p_yp,p_x_k,p_y_k, lambdaT_k)
            LAMBDA.append(lambdaT_k)
            
           
            ## Process economic optimum calculation
            par_ssp2 = vertcat(usp_k,ysp_k,xsp_k_p,p_yp,t_k,p_xp,p_xmp,p_ymp)   
             
            wssp2_guess = DM.zeros(nxp+nu+ny)
            wssp2_guess[0:nxp] = x0_p
            wssp2_guess[nxp:nxp+nu] = u0
            y0_p = Fy_p(x0_p,p_yp,t_k,p_ymp)
            wssp2_guess[nxp+nu:nxp+nu+ny] = y0_p
            
            sol_ss2 = solver_ss2(lbx = wssp2_lb,
                            ubx = wssp2_ub,
                            x0  = wssp2_guess,
                            p = par_ssp2,
                            lbg = gssp2_lb,
                            ubg = gssp2_ub) 
                            
            wssp2_opt = sol_ss2["x"]        
            xs_kp2 = wssp2_opt[0:nxp]
            us_kp2 = wssp2_opt[nxp:nxp+nu]
            ys_kp2 = wssp2_opt[nxp+nu:nxp+nu+ny]
            fss_p2 = sol_ss2["f"]
            Upopt.append(us_kp2)    
            Ypopt.append(ys_kp2)
           
        
Xp = vertcat(*Xp)        
Yp = vertcat(*Yp)        
X_HAT = vertcat(*X_HAT)
Y_HAT = vertcat(*Y_HAT)
D_HAT = vertcat(*D_HAT)
P_K = vertcat(*P_K)
X_KF = vertcat(*X_KF)
Sl = vertcat(*Sl)
if estimating is False: 
    U = vertcat(*U)
    XS = vertcat(*XS)
    YS = vertcat(*YS)
    US = vertcat(*US)
    TIME_SS = vertcat(*TIME_SS)
    TIME_DYN = vertcat(*TIME_DYN)
    COR = vertcat(*COR)
    LAMBDA = vertcat(*LAMBDA)
    Upopt = vertcat(*Upopt) 
    Ypopt = vertcat(*Ypopt) 

## Defining time for plotting
tsim = plt.linspace(0, (Nsim-1)*h, Nsim )

plt.close("all")

 
# Defining path for saving figures 
pf = pathfigure if 'pathfigure' in locals() else './' 
    
if not os.path.exists(pf):
    os.makedirs(pf) 

if estimating is True:
    [X_HAT, _, _] = makeplot(tsim,X_HAT,'State ',pf, Xp, lableg = 'True Value') 
    [Y_HAT, Yp, _] = makeplot(tsim,Y_HAT,'Output ',pf, Yp, lableg = 'True Value')
    [X_KF, Xp, _] = makeplot(tsim,X_KF,'KF State ',pf, Xp, lableg = 'True Value')
    
else:
    if Adaptation is True:
        [Upopt2, US2, _] = makeplot(tsim,Upopt,'Optimal Input VS Target ',pf, US, pltopt = 'steps')
        [U2, Upopt2, _] = makeplot(tsim,U,'Optimal Input VS Actual ',pf, Upopt, pltopt = 'steps', lableg = 'Optimal Value')
        [Yp2, Ypopt, _] = makeplot(tsim,Yp,'Optimal Output VS Actual ',pf, Ypopt, lableg = 'Optimal Value')
        [Upopt,_, _] = makeplot(tsim,Upopt,'Optimal flow ',pf,  pltopt = 'steps')
        [COR, _, _] = makeplot(tsim,COR,'Correction on Output ',pf)  
    [X_HAT, XS, _] = makeplot(tsim,X_HAT,'State ',pf, XS) 
    # [Xp, _, _] = makeplot(tsim,Xp,'Process State ',pf) 
    [U, US, _] = makeplot(tsim,U,'Input ',pf, US, pltopt = 'steps')
    if 'defSP' in locals():
        Ysp = vertcat(*Ysp)
        [Yp, YS, Ysp] = makeplot(tsim,Yp,'Output ',pf,YS,Ysp)
    else:
        [Yp, YS, _] = makeplot(tsim,Yp,'Output ',pf,YS)
    
[D_HAT, _, _ ] = makeplot(tsim,D_HAT,'Disturbance Estimate ',pf)
