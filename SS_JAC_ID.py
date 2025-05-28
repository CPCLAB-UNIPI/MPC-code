#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 11:24:07 2017

@author: marcovaccari
"""

from casadi import *
from casadi.tools import *
import math
from Utilities import*
    
def ss_p_jac_id(ex_name, nx, nu, ny, nd, npx, npy, k, t, px, py, LinPar):
    
    import ex_name as en
    import Default_Values as DV

    # setting the offree value momentaneusly to no when disturbance model is linear
    if hasattr(en, 'offree'):
        NOLINoffree = 'no' if en.offree == 'lin' else en.offree 
    else:
        NOLINoffree = 'no'  #Takes the offree from Default_Values
    
    #### Model calculation  #####################################################
    if hasattr(en, 'User_fxm_Cont'):
        if hasattr(en, 'StateFeedback'):
            if en.StateFeedback is True:
                [Fx_model,Fy_model] = defF_model(en.x,en.u,en.y,en.d,k,t,px,py,NOLINoffree,LinPar, fx = en.User_fxm_Cont, Mx = en.Mx, SF = en.StateFeedback)
        else:
            if hasattr(en, 'User_fym'):
                [Fx_model,Fy_model] = defF_model(en.x,en.u,en.y,en.d,k,t,px,py,NOLINoffree,LinPar, fx = en.User_fxm_Cont, Mx = en.Mx, fy = en.User_fym)
            else:
                [Fx_model,Fy_model] = defF_model(en.x,en.u,en.y,en.d,k,t,px,py,NOLINoffree,LinPar, fx = en.User_fxm_Cont, Mx = en.Mx, C = en.C)
    elif hasattr(en, 'User_fxm_Dis'):
        if hasattr(en, 'StateFeedback'):
            if en.StateFeedback is True:
                [Fx_model,Fy_model] = defF_model(en.x,en.u,en.y,en.d,k,t,px,py,NOLINoffree,LinPar, Fx = en.User_fxm_Dis, SF = en.StateFeedback)
        else:
            if hasattr(en, 'User_fym'):
                [Fx_model,Fy_model] = defF_model(en.x,en.u,en.y,en.d,k,t,px,py,NOLINoffree,LinPar, Fx = en.User_fxm_Dis, fy = en.User_fym)
            else:
                [Fx_model,Fy_model] = defF_model(en.x,en.u,en.y,en.d,k,t,px,py,NOLINoffree,LinPar, Fx = en.User_fxm_Dis, C = en.C)
    elif hasattr(en, 'A') and hasattr(en, 'User_fym'):
            [Fx_model,Fy_model] = defF_model(en.x,en.u,en.y,en.d,k,t,px,py,NOLINoffree,LinPar, A = en.A, B = en.B, fy = en.User_fym)                    
    
    
    def get_attr(attr_name, mod1, mod2):
        if hasattr(mod1, attr_name):
            return getattr(mod1, attr_name)
        elif hasattr(mod2, attr_name):
            return getattr(mod2, attr_name)
        else:
            raise AttributeError(f"Attribute '{attr_name}' is neither in '{mod1.__name__}' nor in '{mod2.__name__}'.")
    
    attr_list = ["umax", "umin", "ymax", "ymin", "xmax", "xmin"] 

    ResultConstr = {attr: get_attr(attr, en, DV) for attr in attr_list}
        
    (solver_ss, wss_lb, wss_ub, gss_lb, gss_ub) = opt_ss_id(nx, nu, ny, nd, npx, npy, Fx_model,Fy_model, \
                umin = ResultConstr["umin"], umax = ResultConstr["umax"], w_s = None, z_s = None, ymin = ResultConstr["ymin"], ymax = ResultConstr["ymax"], \
                    xmin = ResultConstr["xmin"], xmax = ResultConstr["xmax"], h = en.h)
    
    # Set default values in ss point    
    d_0 = DM.zeros(nd,1)
    t_0 = 0.0 
    p_xk0 = DM.zeros((npx,1)) #Dummy variable when parameter is not present
    p_yk0 = DM.zeros((npy,1)) #Dummy variable when parameter is not present
    
    ## Paramenter for Target optimization
    par_ss = vertcat(d_0,t_0, p_xk0, p_yk0)    
    
    # Define useful dimension
    nxu = nx + nu # state+control                     
    nxuy = nx + nu + ny # state+control  
    
    ## Linearization point calculation
    wss_guess = DM.zeros(nxuy)
    wss_guess[0:nx] = en.x0_m
    wss_guess[nx:nxu] = en.u0
    y0 = Fy_model(en.x0_m,en.u0,d_0,t_0,p_yk0)
    wss_guess[nxu:nxuy] = y0
    
#    start_time = time.time()    
    sol_ss = solver_ss(lbx = wss_lb,
                    ubx = wss_ub,
                    x0  = wss_guess,
                    p = par_ss,
                    lbg = gss_lb,
                    ubg = gss_ub)
                    
#    etimess = time.time()-start_time
    wss_opt = sol_ss["x"]        
    xlin = wss_opt[0:nx]
    ulin = wss_opt[nx:nxu]
    ylin = wss_opt[nxu:nxuy]
    
    
    # get linearization of states
    Fun_in = SX.get_input(Fx_model)
    A_dm = jacobian(Fx_model.call(Fun_in)[0],Fun_in[0])
    A = Function('A', [Fun_in[0],Fun_in[1],Fun_in[2],Fun_in[3],Fun_in[4],Fun_in[5]], [A_dm])
    
    B_dm = jacobian(Fx_model.call(Fun_in)[0],Fun_in[1])
    B = Function('B', [Fun_in[0],Fun_in[1],Fun_in[2],Fun_in[3],Fun_in[4],Fun_in[5]], [B_dm])
    
    # get linearization of measurements
    Fun_in = SX.get_input(Fy_model)
    C_dm = jacobian(Fy_model.call(Fun_in)[0], Fun_in[0])
    C = Function('C', [Fun_in[0],Fun_in[1],Fun_in[2],Fun_in[3],Fun_in[4]], [C_dm])
    
    D_dm = jacobian(Fy_model.call(Fun_in)[0], Fun_in[1])
    D = Function('D', [Fun_in[0],Fun_in[1],Fun_in[2],Fun_in[3],Fun_in[4]], [D_dm])
    
    A_k = A(xlin,ulin,en.h,d_0,t_0,p_xk0).full()
    B_k = B(xlin,ulin,en.h,d_0,t_0,p_xk0).full()
      
    C_k = C(xlin,ulin,d_0,t_0,p_yk0).full()
    D_k = D(xlin,ulin,d_0,t_0,p_yk0).full()
    
    return [A_k, B_k, C_k, D_k, xlin, ulin, ylin]


def opt_ss_id(n, m, p, nd, npx, npy, Fx_model,Fy_model, umin = None, umax = None, w_s = None, z_s = None, ymin = None, ymax = None, xmin = None, xmax = None, h = None):
    """
    SUMMARY:
    It builds the steady-state hunt problem
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
    par_ss = MX.sym("par_ss", nd+1+npx+npy)
    d = par_ss[:+nd]
    t = par_ss[nd:nd+1]
    px = par_ss[nd+1:nd+1+npx]
    py = par_ss[nd+1+npx:nd+1+npx+npy]

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
        
    Xs_next = Fx_model( Xs, Us, h, d, t, px)
     
    gss.append(Xs_next - Xs)
    gss = vertcat(*gss)
    
    Ys_next = Fy_model( Xs, Us, d, t, py)
    gss = vertcat(gss , Ys_next- Ys)
    
    # Defining obj_fun
    fss_obj = mtimes((Xs_next - Xs).T,(Xs_next - Xs)) + mtimes((Ys_next - Ys).T,(Ys_next - Ys))

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
    gss_lb = DM.zeros(ng,1)   # Equalities identification 
    gss_ub = DM.zeros(ng,1)

    
    nlp_ss = {'x':wss, 'p':par_ss, 'f':fss_obj, 'g':gss}
    
    solver_ss = nlpsol('solver','ipopt', nlp_ss)

    return [solver_ss, wss_lb, wss_ub, gss_lb, gss_ub]


