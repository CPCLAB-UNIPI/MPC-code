# -*- coding: utf-8 -*-
"""
Created on December 3, 2015

@author: Marco, Mirco, Gabriele

Utilities for general purposes
"""
from __future__ import division

from builtins import str
from builtins import range
from past.utils import old_div
from casadi import *
from casadi.tools import *
from matplotlib import pylab as plt
import math
import scipy.linalg as scla
import numpy as np

def defF_p(x, u, y, k, t, pxp, pyp, pxmp, pymp,LinPar, **plant):
    """
    SUMMARY:
    Starting from system matrix or equation builds the system model
    
    SYNTAX:
    assignment = defFp(x, u, y, k, t, **plant)
  
    ARGUMENTS:
    + x, u, y           - State, Input and Output symbolic variable 
    + k                 - Integration step symbolic variable
    + t                 - Current time
    + pxp, pyp          - Parameters on State and Output maps
    + pxmp, pymp        - Measurable Parameters on State and Output maps
    + LinPar            - Tag for setting additive Parameters on State and Output maps
    + plant             - Plant equations/system matrices 
   
    OUTPUTS:
    + Fx_p          - State correlation function  
    + Fy_p          - Output correlation function    
    """
    nx = x.size1()
    
    for key in plant:
        if key == 'Ap':    # Linear model 
            Ap = plant['Ap']
            Bp = plant['Bp']
            fx_p = (mtimes(Ap,x) + mtimes(Bp,u)) + pxp + pxmp
            Fx_p = Function('Fx_p', [x,u,pxp,t,k,pxmp], [fx_p])
        
        elif key == 'Fx': # NON-Linear Discrete-Time
            Fx_p = plant['Fx']
            dummyF = Fx_p(x,t,u,pxp,pxmp)
            if LinPar is True:
                dummyF = dummyF + pxp + pxmp
            Fx_p = Function('Fx_p', [x,u,pxp,t,k,pxmp], [dummyF])
        
        elif key == 'fx': # NON-Linear Continuous-Time
                deffxp = plant['fx']
                Mx = plant['Mx']
                dummyF = deffxp(x,t,u,pxp,pxmp)
                xnew = vertcat(x,t)
                unew = vertcat(u,pxp)
                unew = vertcat(unew,pxmp)
                
                # Constructing the augmented system for the integrator
                dummyF2 = vertcat(dummyF, SX(1.))
                dummyF3 = Function('dummyF3', [xnew,unew], [dummyF2])
                
                Int_Fx_p = simpleRK(dummyF3, Mx)
                dummyF_f = vertcat(Int_Fx_p(xnew,unew,k))
                Fx_p2 = Function('Fx_p2', [x,u,t,k,pxp,pxmp], [dummyF_f])
                
                # Caring only about the [:nx] of the output:
                Fx_p = Function('Fx_p',[x,u,t,k,pxp,pxmp],[Fx_p2(x,u,t,k,pxp,pxmp)[:nx,:]])
                
                # Adding disturbance dx linearly
                if LinPar is True:
                    Pxp = Function('Pxp', [pxp], [pxp])
                    Pxmp = Function('Pxmp', [pxmp], [pxmp])
                    dummyF = vertcat(Fx_p(x,u,t,k,pxp,pxmp)) + vertcat(Pxp(pxp))+ vertcat(Pxmp(pxmp))
                    Fx_p = Function('Fx_p', [x,u,pxp,t,k,pxmp], [dummyF])
        
        if key == 'SF':
            fy_p = x
            Fy_p = Function('Fy_p', [x,u,pyp,t,pymp], [fy_p])
        else:
            if key == 'Cp':    # Linear model
                Cp = plant['Cp']
                fy_p = mtimes(Cp,x) + pyp + pymp
                Fy_p = Function('Fy_p', [x,u,pyp,t,pymp], [fy_p])
                
            elif key == 'fy':   # NON-Linear model
                deffyp = plant['fy']
                dummyF = deffyp(x,u,t,pyp,pymp) 
                if LinPar is True:
                    dummyF = dummyF + pyp + pymp
                Fy_p = Function('Fy_p', [x,u,pyp,t,pymp], [dummyF])
    
    return [Fx_p,Fy_p]
    
def defF_model(x, u, y, d, k, t, px, py, offree, LinPar, **model): 
    """
    SUMMARY:
    Starting from system matrix or equation builds the system model
    
    SYNTAX:
    assignment = defF_model(x, u, y, d, k, t, offree, **model)
  
    ARGUMENTS:
    + x, u, y, d        - State, Input, Output and Disturbance symbolic variable 
    + k                 - Integration step symbolic variable
    + t                 - Current time
    + px, py            - Parameters on State and Output maps
    + offree            - Offset free tag 
    + LinPar            - Tag for setting additive Parameters on State and Output maps
    + model             - Model equations/system matrices 
   
    OUTPUTS:
    + Fx_model          - State correlation function  
    + Fy_model          - Output correlation function    
    """
    if offree == "lin":
        Bd = model['Bd']
        Cd = model['Cd']
        
    if offree == 'nl':
        unew = vertcat(u,d)
    else:
        unew = u
    
    nx = x.size1()
    
    for key in model:
        if key == 'A':    # Linear model 
            A = model['A']
            B = model['B']
            for bkey in model:
                if bkey == 'xlin':
                    xlin = model['xlin']
                    ulin = model['ulin']
                    fx_model = mtimes(A,x - xlin) + mtimes(B,u - ulin) + xlin # The state model is linearised in xlin, ulin
                    break
            try:
                fx_model
            except NameError:
                fx_model = mtimes(A,x) + mtimes(B,u)
            
            if offree == "lin":
                fx_model = fx_model + mtimes(Bd,d)
            
            # Adding measurable parameters linearly
            fx_model = fx_model + px
            
            Fx_model = Function('Fx_model',[x,u,k,d,t,px], [fx_model])
                
        elif key == 'fx':   # NON-Linear continuous model 
            fx = model['fx']
            Mx = model['Mx']
                          
            fx_model = fx(x,u,d,t,px)
            
            dummyF = vertcat(fx_model, SX(1.))
            xnew = vertcat(x,t)
            unew = vertcat(unew,px)
            
            dummyF2 = Function('dummyF2', [xnew,unew], [dummyF])
            Int_Fx_m = simpleRK(dummyF2, Mx)
            dummyF_f = vertcat(Int_Fx_m(xnew,unew,k))
            Fx_model2 = Function('Fx_model2', [x,u,k,d,t,px], [dummyF_f])
            
            Fx_model = Function('Fx_model',[x,u,k,d,t,px],[Fx_model2(x,u,k,d,t,px)[:nx,:]])
                
            if offree == "lin":
                Dx = Function('Dx', [d], [mtimes(Bd,d)])
                dummyF = vertcat(Fx_model(x,u,k,d,t,px)) + vertcat(Dx(d))
                Fx_model = Function('Fx_model', [x,u,k,d,t,px], [dummyF])
                
            # Adding measurable parameter linearly
            if LinPar is True:
                Px = Function('Px', [px], [px])
                dummyF = vertcat(Fx_model(x,u,k,d,t,px)) + vertcat(Px(px))
                Fx_model = Function('Fx_model', [x,u,k,d,t,px], [dummyF])
        
       
        elif key == 'Fx':   # NON-linear discrete model
            
            Fx_model = model['Fx']
            
            dummyF = Fx_model(x,u,d,t,px)
            
            if offree == "lin":
                dummyF = dummyF + mtimes(Bd,d)
            
            if LinPar is True:
                dummyF = dummyF + px        
                
            Fx_model = Function('Fx_model', [x,u,k,d,t,px], [dummyF])
            
                
        if key == 'SF':
            fy_model = x    
            
            if offree == "lin":
                fy_model = fy_model + mtimes(Cd,d) 
                    
        else:
            if key == 'C':    # Linear model
                C = model['C']
                for bkey in model:
                    if bkey == 'ylin':
                        ylin = model['ylin']
                        for tkey in model:
                            if tkey == 'xlin':
                                xlin = model['xlin']
                                fy_model = mtimes(C,x - xlin) + ylin # Both the state and the output model are linearised
                                break
                        try:
                            fy_model
                            break
                        except NameError:
                            fy_model = mtimes(C,x) + ylin # Only the output model is linearised
                            break
                try:
                    fy_model
                except NameError:
                    fy_model = mtimes(C,x) # The system is linear and there was no need to linearised it
            
                if offree == "lin":
                    fy_model = fy_model + mtimes(Cd,d) 
                
            elif key == 'fy':                   # NON-Linear model 
                fy = model['fy']
              
                fy_model = fy(x,u,d,t,py)
                    
                if offree == "lin":
                    fy_model = fy_model + mtimes(Cd,d) 
    
    if LinPar is True:
        # Adding linearly parameters 
        fy_model = fy_model + py   
        
    Fy_model = Function('Fy_model', [x,u,d,t,py], [fy_model])
    return [Fx_model,Fy_model]

def xQx(x,Q):
    """
    SUMMARY:
    Starting from a vector x and a square matrix Q, the function execute the
    operation x'Qx 
    
    SYNTAX:
    result = sysaug(x, Q)
  
    ARGUMENTS:
    + x - Column vector 
    + Q - Square matrix
        
    OUTPUTS:
    + Qx2 - Result of the operation x'Qx
    """
    Qx = mtimes(Q,x)
    Qx2 = mtimes(x.T,Qx)
    return Qx2
    
def defFss_obj(x, u, y, xsp, usp, ysp, **kwargs):
    """
    SUMMARY:
    It construct the steady-state optimisation objective function
    
    SYNTAX:
    assignment = defFss_obj(x, u, y, xsp, usp, ysp, **kwargs)
  
    ARGUMENTS:
    + x, u, y       - State, Input and Output symbolic variables
    + xsp, usp, ysp - State, Input and Output setpoints 
    + kwargs        - Objective function/Matrix for QP/Vector for LP problem
    
    OUTPUTS:
    + Fss_obj       - Steady-state optimisation objective function       
    """
    for key in kwargs: 
        if key == 'r_y':          # LP problem
            rss_y = kwargs['r_y']
            x_abs = fabs(x)
            u_abs = fabs(u)
            for bkey in kwargs:
                if bkey is 'r_u':
                    rss_u = kwargs['r_u']  
                    fss_obj = mtimes(rss_y,y) + mtimes(rss_u,u_abs)
                    break
                elif bkey is 'r_Du':
                    rss_Du = kwargs['r_Du']
                    fss_obj = mtimes(rss_y,y) + mtimes(rss_Du,u_abs)
                    break      
                
            break
        elif key == 'Q':          # QP problem
            Qss = kwargs['Q']
            yQss2 = xQx(y,Qss)
            
            for bkey in kwargs: 
                if bkey is 'R':
                    Rss = kwargs['R']   
                    uRss2 = xQx(u, Rss)
                    fss_obj = 0.5*(yQss2 + uRss2)
                    break
                elif bkey is 'S':
                    Sss = kwargs['S']
                    uSss2 = xQx(u,Sss)
                    fss_obj = 0.5*(yQss2 + uSss2)
                    break                    
            break
        elif key == 'f_obj':       # NON-linear function            
            f_obj1 = kwargs['f_obj']
            fss_obj = f_obj1(x,u,y,xsp,usp,ysp)
            break
        
    Fss_obj = Function('Fss_obj', [x,u,y,xsp,usp,ysp], [fss_obj])
    return Fss_obj
    
def defF_obj(x, u, y, xs, us, ys, **kwargs): 
    """
    SUMMARY:
    It constructs the dynamic optimisation objective function
    
    SYNTAX:
    assignment = defF_obj(x, u, y, xs, us, ys, **kwargs)
  
    ARGUMENTS:
    + x, u, y       - State, Input and Output symblic variables 
    + xs, us, ys    - State, Input and Output target symblic variables 
    + kwargs        - Objective function/Matrix for QP/Vector for LP problem

    OUTPUTS:
    + F_obj         - Dynamic optimisation objective function       
    """    

    for key in kwargs:     
        if key is 'r_x':                # LP problem
            r_x = kwargs['r_x']
            x_abs = fabs(x)
            u_abs = fabs(u)
            for bkey in kwargs: 
                if bkey is 'r_u':
                    r_u = kwargs['r_u']
                    f_obj = mtimes(r_x,x_abs) + mtimes(r_u,u_abs)
                elif bkey is 'r_Du':
                    r_Du = kwargs['r_Du']
                    f_obj = mtimes(r_x,x_abs) + mtimes(r_Du,u_abs)
            F_obj = Function('F_obj', [x,u,y,xs,us,ys], [f_obj])      
        elif key is 'Q':                # QP problem
            Q = kwargs['Q']
            xQ2 = xQx(x,Q)
            for bkey in kwargs: 
                if bkey is 'R':
                    R = kwargs['R']
                    uR2 = xQx(u,R)
                    f_obj = 0.5*(xQ2 + uR2)
                    break
                elif bkey is 'S':
                    S = kwargs['S']
                    uS2 = xQx(u,S)
                    f_obj = 0.5*(xQ2 + uS2)
                    break                    
            F_obj = Function('F_obj', [x,u,y,xs,us,ys], [f_obj])       
        elif key is 'f_Cont':       # NON-linear continuous function
            f_obj = kwargs['f_Cont']
            F_obj = f_obj
        elif key is 'f_Dis':
            f_obj = kwargs['f_Dis']
            F_obj1 = f_obj(x,u,y,xs,us,ys)
            F_obj = Function('F_obj', [x,u,y,xs,us,ys], [F_obj1])
        elif key is 'f_Coll':
            s_Coll = kwargs['s_Coll']
            f_obj = kwargs['f_Coll']
            F_obj1 = f_obj(x,u,y,xs,us,ys,s_Coll)
            F_obj = Function('F_obj', [x,u,y,xs,us,ys,s_Coll], [F_obj1])
            
    return F_obj 
    
def defVfin(x,xs, **Tcost): 
    """
    SUMMARY:
    It constructs the terminal cost for the dynamic optimisation objective function
    
    SYNTAX:
    assignment = defVfin(x, **Tcost):
  
    ARGUMENTS:
    + x             - State symbolic variable
    + Tcost         - Terminal weigth specified by the user/ Matrices to calculate Riccati equation
    
    OUTPUTS:
    + Vfin          - terminal cost for the dynamic optimisation objective function       
    """ 
    
    if Tcost == {}:
        vfin = 0.0
    else:  
        for key in Tcost:
            if key == 'A':                  # Linear system & QP problem
                A = Tcost['A']
                B = Tcost['B'] 
                Q = Tcost['Q']
                R = Tcost['R'] 
            ## Solution to Riccati Equation
                P = DM(scla.solve_discrete_are(np.array(A), np.array(B), \
                      np.array(Q), np.array(R)))
    
                vfin = 0.5*(xQx(x,P)) 
                break
            elif key == 'vfin_F':
                vfin_F = Tcost['vfin_F']
                vfin = vfin_F(x,xs)
                break
    Vfin = Function('Vfin', [x,xs], [vfin])
    
    return Vfin

def makeplot(tsim,X1,label,pf,*var,**kwargs): 
    """
    SUMMARY:
    It constructs the plot where tsim is on the x-axis, 
    X1,X2,X3 on the y-axis, and label is the label of the y-axis 
    
    SYNTAX:
    makeplot(tsim,X1,label,*var):
  
    ARGUMENTS:
    + tsim          - x-axis vector (time of the simulation (min))
    + X1,X2,X3      - y-axis vectors.
    X1 represent the actual value
    X2 the target (eventual)
    X3 the setpoint (eventual)
    + label         - label for the y-axis
    + pf            - path where the plots are saved
    + var           - positional variables to include another vector/s X2 and X3 to plot together with X1
    + kwargs        - plot options including linestyle and changing the default legend values
    """ 

    linetype = '-' #defaul value for linetype
    lableg = 'Target' #defaul value for legend label
    for kwkey in kwargs:    
        if kwkey == 'pltopt':
            linetype = kwargs['pltopt']
        if kwkey == 'lableg':
            lableg = kwargs['lableg']
            
    nt = int(tsim.size)
    
    X1 = np.array(X1)
    
    sz = old_div(X1.size,nt)
    Xout1 = np.zeros((nt,sz))
    Xout2 = np.zeros((nt,sz))
    Xout3 = np.zeros((nt,sz))
    
    for k in range(sz):
        x1 = X1[k::sz]
        
        plt.figure()
        try:
            plt.plot(tsim, x1, ls = linetype)
        except ValueError:
            plt.step(tsim,x1)
        plt.xlabel('Time ')
        plt.ylabel(label + str(k+1))
        plt.gca().set_xlim(left=0,right=tsim[-1])
        
        Xout1[:,k] = np.reshape(x1,(nt,))
        
        for i_var in range(len(var)):
            # extract dimension of var
            var_i = var[i_var]
            Xi = np.array(var_i)
            xi = Xi[k::sz]
            try:
                plt.plot(tsim, xi, ls = linetype)
            except ValueError:
                plt.step(tsim,xi)
            if i_var == 0:
                plt.legend(('Actual', lableg))
                plt.gca().set_xlim(left=0,right=tsim[-1])
                Xout2[:,k] = np.reshape(xi,(nt,))
            elif i_var == 1:
                plt.legend(('Actual', 'Target', 'Set-Point'))
                plt.gca().set_xlim(left=0,right=tsim[-1])
                Xout3[:,k] = np.reshape(xi,(nt,))
                     
        plt.grid(True)
       
        plt.savefig(pf + label + str(k+1) + '.pdf', format = 'pdf', transparent = True, bbox_inches = 'tight' )
        
    return [Xout1, Xout2, Xout3]

def defLambdaT(xp,x,u,y,d,k,t,pxp,pyp,px,py, fx_model, fxp, Fy_model, Fy_p, alphalss): 
    """
    SUMMARY:
    It constructs the function to evaluate the modifiers adaptation correction term
    
    SYNTAX:
    assignment = defLambdaT(xp,x,u,y,d,k, fx_model, fxp, Fy_model, Fy_p): 
    
    ARGUMENTS:
    + xp,x,u,y,d,k  - State, Input, Output, Disturbance symbolic variables
    + fx_model      - Model state function
    + fxp           - Process state function
    + Fy_model      - Model output function
    + Fy_p          - Process output function
    + alphalss      - Filtering factor (default value = 0.2)
    
    OUTPUTS:
    + LambdaT       - Function to evaluate the modifiers correction term       
    """ 
    lambdaTprev = SX.sym('lambdaTprev',(y.size1(),u.size1()))
    
    Fun_in = SX.get_input(fx_model)
    Nablaxfx = jacobian(fx_model.call(Fun_in)[0], Fun_in[0])
    Nablaufx = jacobian(fx_model.call(Fun_in)[0], Fun_in[1]) 
    inv_Nablaxfx = solve((DM.eye(Nablaxfx.size1())- Nablaxfx), Nablaufx) 
    Fun_in = SX.get_input(Fy_model)    
    Nablaxfy = jacobian(Fy_model.call(Fun_in)[0], Fun_in[0]) 
    gradyModel = mtimes(Nablaxfy,inv_Nablaxfx)    
    
    Fun_in = SX.get_input(fxp)
    Nablaxfxp = jacobian(fxp.call(Fun_in)[0], Fun_in[0])
    Nablaufxp = jacobian(fxp.call(Fun_in)[0], Fun_in[1]) 
    inv_Nablaxfxp = solve((DM.eye(Nablaxfxp.size1())- Nablaxfxp), Nablaufxp) 
    Fun_in = SX.get_input(Fy_p)    
    Nablaxfyp = jacobian(Fy_p.call(Fun_in)[0], Fun_in[0])
    gradyPlant = mtimes(Nablaxfyp,inv_Nablaxfxp)     
    
    gradydiff = gradyPlant - gradyModel
    
    lambdaT = (1-alphalss)*lambdaTprev + alphalss*gradydiff
    
    LambdaT = Function('LambdaT', [xp,x,u,d,y,k,t,pxp,pyp,px,py,lambdaTprev], [lambdaT])
    
    return LambdaT
    
def opt_ssp(n, m, p, nd, npx, npy, npxp, Fx ,Fy ,sol_opts, xmin = None, xmax = None, h = None):
    """
    SUMMARY:
    It builds the process steady-state point optimization problem 
    """
    # Define symbolic optimization variables
    Xs = MX.sym("wss",n) 
    
    # Define parameters
    par_ss = MX.sym("par_ss", 1+m+npx+npxp+npyp)
    t = par_ss[0]
    us_k = par_ss[1:m+1]
    pxp = par_ss[m+1:m+npxp+1]
    px = par_ss[m+npxp+1:m+npxp+1+npx]
    py = par_ss[m+npxp+1+npx:m+npxp+1+npx+npy]
    
    if xmin is None:
        xmin = -DM.inf(n)
    if xmax is None:
        xmax = DM.inf(n)
    
    if h is None:
        h = .1 #Defining integrating step if not provided from the user
    gss = []

    Xs_next = Fx( Xs, us_k, pxp, t, h, px) 
        
    gss.append(Xs_next - Xs)
    gss = vertcat(*gss)
    
    fss_obj = mtimes((Xs_next - Xs).T,(Xs_next - Xs))

    ng = gss.size1()
    gss_lb = DM.zeros(ng,1)   # Equalities identification 
    gss_ub = DM.zeros(ng,1)
      
    nlp_ss = {'x':Xs, 'p':par_ss, 'f':fss_obj, 'g':gss}
    
    solver_ss = nlpsol('solver','ipopt', nlp_ss, sol_opts)
    
    return [solver_ss, xmin, xmax, gss_lb, gss_ub]
    
def opt_ssp2(n, m, p, nd, npx, npy, npxp, npyp, Fx ,Fy ,Fss_obj,QForm_ss,sol_opts, umin = None, umax = None, w_s = None, z_s = None, ymin = None, ymax = None, xmin = None, xmax = None, h = None):
    """
    SUMMARY:
    It searches the true optimal plant value
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
    par_ss = MX.sym("par_ss", n+m+p+npyp+1+npxp+npx+npy)
    usp = par_ss[0:m]   
    ysp = par_ss[m:m+p]
    xsp = par_ss[m+p:m+p+n]
    pyp = par_ss[m+p+n:m+p+npyp+n]
    t = par_ss[m+p+npyp+n:m+p+npyp+n+1]
    pxp = par_ss[m+p+npyp+n+1:m+p+npyp+n+1+npxp]
    px = par_ss[m+p+npyp+n+1+npxp:m+p+npyp+n+1+npxp+npx]
    py = par_ss[m+p+npyp+n+1+npxp+npx:m+p+npyp+n+1+npxp+npx+npy]
    
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

    Xs_next = Fx( Xs, Us, pxp, t, h, px)
        
    gss.append(Xs_next - Xs)
    gss = vertcat(*gss)
    
    Ys_next = Fy( Xs,Us, pyp, t, py)
    gss = vertcat(gss , Ys_next- Ys)
    
    # Defining obj_fun
    dy = Ys
    du = Us
    dx = Xs
    
    if QForm_ss is True:   #Checking if the OF is quadratic
        dx = dx - Xs
        dy = dy - ysp
        du = du - usp
            
    fss_obj = Fss_obj( dx, du, dy, xsp, usp, ysp)

    wss_lb = -DM.inf(nxuy)
    wss_ub = DM.inf(nxuy)
    wss_lb[0:n] = xmin
    wss_ub[0:n] = xmax
    wss_lb[n: nxu] = umin
    wss_ub[n: nxu] = umax
    wss_lb[nxu: nxuy] = ymin
    wss_ub[nxu: nxuy] = ymax
    
    ng = gss.size1()
    gss_lb = DM.zeros(ng,1)   # Equalities identification 
    gss_ub = DM.zeros(ng,1)
     
    
    nlp_ss = {'x':wss, 'p':par_ss, 'f':fss_obj, 'g':gss}
    
    solver_ss = nlpsol('solver','ipopt', nlp_ss, sol_opts)
    
    return [solver_ss, wss_lb, wss_ub, gss_lb, gss_ub]
       
          
def defF_obj_mhe(w, v, t, **kwargs): 
    """
    SUMMARY:
    It constructs the objective function for the mhe optimization problem
    
    SYNTAX:
    assignment = defF_obj(w, v, y, **kwargs)
  
    ARGUMENTS:
    + w, v, t       - State, Input and Time symblic variables 
    + kwargs        - Objective function/Matrix for QP/Vector for LP problem

    OUTPUTS:
    + F_obj         - Dynamic optimisation objective function       
    """    

    for key in kwargs:     
        if key is 'r_w':                # LP problem
            r_w = kwargs['r_w']
            r_v = kwargs['r_v']
            f_obj = mtimes(r_w,w) + mtimes(r_v,v)
            F_obj = Function('F_obj', [w,v,t], [f_obj])      
        elif key is 'Q':                # QP problem
            Q = kwargs['Q']
            wQ2 = xQx(w,Q)
            R = kwargs['R']
            vR2 = xQx(v,R)
            f_obj = 0.5*(wQ2 + vR2)
            F_obj = Function('F_obj', [w,v,t], [f_obj])       
        elif key is 'f_obj':
            f_obj = kwargs['f_obj']
            F_obj1 = f_obj(w,v,t)
            F_obj = Function('F_obj', [w,v,t], [F_obj1])

    return F_obj 
    


def defFx_mhe(x, u, w, d, k, t, px, offree, LinPar, **model): 
    """
    SUMMARY:
    Starting from equation builds the system model
    
    SYNTAX:
    assignment = defFx_mhe(x, u, w, d, k, t, offree, **model)
  
    ARGUMENTS:
    + x, u, w, d        - State, Input, State noise and Disturbance symbolic variable 
    + k                 - Integration step symbolic variable
    + t                 - Current time
    + px                - Parameters on State map
    + offree            - Offset free tag 
    + model             - Model equations/system matrices 
   
    OUTPUTS:
    + Fx_mhe            - State correlation function  
    """
    if offree == 'nl':
        unew = vertcat(u,d)
    else:
        unew = u
    
    nx = x.size1()
    nd = d.size1()
    
    csi = SX.sym("csi", nx+nd)
    x1 = csi[0:nx]
    d1 = csi[nx:nx+nd]     
    
    
    for key in model:
        if key == 'fx':   # NON-Linear continuous model 
            fx = model['fx']
            Mx = model['Mx']
            fx_mhe = fx(x,u,d,t,px,w)  # to trasform in SX to be processed in casadi function          
            
            dummyF = vertcat(fx_mhe, SX(1.))
            xnew = vertcat(x,t)
            unew = vertcat(unew,w)
            unew = vertcat(unew,px)
            
            dummyF2 = Function('dummyF2', [xnew,unew], [dummyF])
            Int_Fx_m = simpleRK(dummyF2, Mx)
            dummyF_f = vertcat(Int_Fx_m(xnew,unew,k))
            Fx_2 = Function('Fx_2', [x,u,k,t,w,px], [dummyF_f]) #In this way the output is always a SX

            # Caring only about the [:nx] of the output:
            Fx_mhe = Function('Fx_mhe',[x,u,k,t,w,px],[Fx_2(x,u,k,t,w,px)[:nx,:]])
            
            if offree == "no":
                G = model['G']
                Gw = Function('Gw', [w], [mtimes(G,w)])
                if LinPar == true:
                    Px = Function('Px', [px], [px])
                    dummyF = vertcat(Fx_mhe(x,u,k,t,w,px)) + vertcat(Gw(w)) + vertcat(Px(px))
                    Fx_mhe = Function('Fx_mhe', [x,u,k,t,w,px], [dummyF])
                else:
                    dummyF = vertcat(Fx_mhe(x,u,k,t,w,px)) + vertcat(Gw(w))
                    Fx_mhe = Function('Fx_mhe', [x,u,k,t,w,px], [dummyF])
            break
        
        elif key == 'Fx':   # NON-linear discrete model
            Fx = model['Fx']
            if offree == 'nl':
                dummyF = Fx(x1,u,d1,t,px,w)
                Fx_mhe = Function('Fx_mhe', [csi,u,k,t,w,px], [dummyF])
                
                Dx = Function('Dx',[d],[d])
                dummyFx = vertcat(Fx_mhe(csi,u,k,t,w,px), Dx(d1))
                Fx_mhe2 = Function('Fx_mhe2', [csi,u,k,t,w,px], [dummyFx])
                
                G = model['G']
                Gw = Function('Gw', [w], [mtimes(G,w)])
                if LinPar == True: 
                    Px = Function('Px', [px], [vertcat(px,DM.zeros(nd))])
                    dummyF = vertcat(Fx_mhe2(csi,u,k,t,w,px)) + vertcat(Gw(w)) + vertcat(Px(px))
                    Fx_mhe = Function('Fx_mhe', [csi,u,k,t,w,px], [dummyF])  
                else:
                    dummyF = vertcat(Fx_mhe2(csi,u,k,t,w,px)) + vertcat(Gw(w))
                    Fx_mhe = Function('Fx_mhe', [csi,u,k,t,w,px], [dummyF])  
            else:
                dummyF = Fx_model(x,u,d,t,px)
                
                if LinPar is True:
                    dummyF = dummyF + px
                
                Fx_mhe = Function('Fx_mhe', [x,u,k,t,w,px], [dummyF])
            break
                
    if offree == "lin":
            Bd = model['Bd']
            Dx = Function('Dx', [d], [mtimes(Bd,d)])
            dummyF = vertcat(Fx_mhe(x,u,k,t,w,px)) + vertcat(Dx(d))
            Fx_mhe1 = Function('Fx_mhe1', [x,u,k,t,w,d,px], [dummyF])
            Dx = Function('Dx',[d],[d])
            dummyFx = vertcat(Fx_mhe1(x1,u,k,t,w,d1,px), Dx(d1))
            Fx_mhe2 = Function('Fx_mhe2', [csi,u,k,t,w,px], [dummyFx])
            
            G = model['G']
            Gw = Function('Gw', [w], [mtimes(G,w)])
            if LinPar == True:
                Px = Function('Px', [px], [vertcat(px,DM.zeros(nd))])
                dummyF = vertcat(Fx_mhe2(csi,u,k,t,w,px)) + vertcat(Gw(w)) + vertcat(Px(px))
                Fx_mhe = Function('Fx_mhe', [csi,u,k,t,w,px], [dummyF])
            else:
                dummyF = vertcat(Fx_mhe2(csi,u,k,t,w,px)) + vertcat(Gw(w))
                Fx_mhe = Function('Fx_mhe', [csi,u,k,t,w,px], [dummyF])
            
    return Fx_mhe

def mhe_opt(n, nd, m, p,npx,npy, n_w, F_obj_mhe, Fx_model, Fy_model, N, N_mhe, ksim, h, mhe_up, sol_opts, G_ineq, H_eq, wmin = None, wmax = None, vmin = None, vmax = None, ymin = None, ymax = None, xmin = None, xmax = None):
    """
    SUMMARY:
    It builds the MHE optimization problem
    """
    # Extract dimensions
    nxv = n+p 
    nxvw = nxv + n_w
    n_opt = N*nxvw + n # total # of variables  
    nmd = n-nd
    
    # Define symbolic optimization variables
    w_opt = MX.sym("w",n_opt)  # w_opt = [x[T],w[T], ... ,x[T+N-1],w[T+N-1]]
        
    # Get states
    X = [w_opt[nxvw*k : nxvw*k+n] for k in range(N+1)]
    
    # Get output rumor
    V = [w_opt[nxvw*k+n : nxvw*k + nxv] for k in range(N)]
     
    # Get state rumor
    W = [w_opt[nxvw*k+nxv : nxvw*k + nxvw] for k in range(N)]
    
    idx = N_mhe if N_mhe == 1 else N_mhe-1
    
    # Define parameters U,Y,x_bar,P_k_r
    par = MX.sym("par", N*m+N*p+n+n*n+N+(p*(idx))**2+p*(idx)+p*(idx)*n+(p+nmd)*N)
    U = [par[m*k : m*k+m] for k in range(N)]
    Y = [par[N*m + p*k : N*m + p*k+p] for k in range(N)]
    x_bar = par[N*(m+p):N*(m+p)+n]
    P_k_inv_r = par[N*(m+p)+n:N*(m+p)+n+n*n]
    t = par[N*(m+p)+n+n*n:N*(m+p)+n+n*n+N] 
    Pycondx_inv_r = par[N*(m+p)+n+n*n+N:N*(m+p)+n+n*n+N+(p*(idx))**2]
    Hbig = par[N*(m+p)+n+n*n+N+(p*(idx))**2:N*(m+p)+n+n*n+N+(p*(idx))**2+p*(idx)]
    Obig_r = par[N*(m+p)+n+n*n+N+(p*(idx))**2+p*(idx):N*(m+p)+n+n*n+N+(p*(idx))**2+p*(idx)+p*(idx)*n]
    PX = [par[N*(m+p)+n+n*n+N+(p*(idx))**2+p*(idx)+p*(idx)*n + npx*k : N*(m+p)+n+n*n+N+(p*(idx))**2+p*(idx)+p*(idx)*n+ npx*k+npx] for k in range(N)]
    PY = [par[N*(m+p)+n+n*n+N+(p*(idx))**2+p*(idx)+p*(idx)*n +npx*N + npy*k : N*(m+p)+n+n*n+N+(p*(idx))**2+p*(idx)+p*(idx)*n+ N*npx + npy*k+npy] for k in range(N)]
    
    Pycondx_inv = Pycondx_inv_r.reshape((p*(idx),p*(idx))) 
    Obig = Obig_r.reshape((p*(idx),n)) 

    P_k_inv = P_k_inv_r.reshape((n,n)) #shaping P_k vector in order to reconstruct the matrix

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
    if wmin is None:
        wmin = -DM.inf(n_w)
    if wmax is None:
        wmax = DM.inf(n_w) 
    if vmin is None:
        vmin = -DM.inf(p)
    if vmax is None:
        vmax = DM.inf(p) 
       
    if h is None:
        h = .1 #Defining integrating step if not provided from the user
   

    if G_ineq != None:
        g_ineq = G_ineq(xSX,uSX,dSX,tSX,pxSX)
        G_ineqSX = Function('G_ineqSX', [xSX,uSX,dSX,tSX,pxSX], [g_ineq])
        
    if H_eq != None:
        h_eq = H_eq(xSX,uSX,dSX,tSX,pxSX)
        H_eqSX = Function('H_eqSX', [xSX,uSX,dSX,tSX,pxSX], [h_eq])
        
    # Initializing constraints vectors and obj fun
    g = []
    g1 = [] # Costraint vector for y bounds
    g4 = [] # User defined inequality constraints
    g5 = [] # User defined equality constraints

    f_obj = 0.0;

    for k in range(N):
        Y_k = Fy_model( X[k],U[k], t[k], PY[k]) + V[k]
         
        if G_ineq != None:
            G_k = G_ineqSX(X[k], U[k], d, t, p_xk[:,k])
        else:
            G_k = []
        if H_eq != None:
            H_k = H_eqSX(X[k], U[k], d, t, p_xk[:,k])
        else:
            H_k = []
            
        g4.append(G_k)
        g5.append(H_k)
        
        if yFree is False:
            g1.append(Y_k) #bound constraint on Y_k

        g.append(Y_k - Y[k])
        
        X_next = Fx_model( X[k], U[k], h, t[k], W[k], PX[k]) 

        g.append(X_next - X[k+1])
    
        f_obj_new = F_obj_mhe( W[k], V[k], t[k] ) 
        
        f_obj += f_obj_new
        
    
    g = vertcat(*g)
    g1 = vertcat(*g1) #bound constraint on Y_k 
    g4 = vertcat(*g4)
    g5 = vertcat(*g5)
       
    v_in = 0.5*xQx((X[0]-x_bar) ,P_k_inv)
    f_obj += v_in #adding the prior weight
    

    ## Subtracting repeated terms from the prior weight when using smoothing update
    if mhe_up == 'smooth' and ksim >= N_mhe: 
        yes = vertcat(*Y[0:N_mhe-1])-mtimes(Obig,X[0])-Hbig
        v_s = 0.5*xQx(yes ,Pycondx_inv)
        f_obj += -v_s
    
    #Defining bound constraint
    w_lb = -DM.inf(n_opt)
    w_ub = DM.inf(n_opt)
    w_lb[0:n] = xmin
    w_ub[0:n] = xmax
    
    ng = g.size1()
    ng1 = g1.size1()
    ng = g.size1()
    ng4 = g4.size1()
    ng5 = g5.size1()
    g_lb = DM.zeros(ng+ng1+ng4+ng5,1)
    g_ub = DM.zeros(ng+ng1+ng4+ng5,1)
    
    if ng4 != 0:
       g_lb[ng+ng1+ng2:ng+ng1+ng2+ng4] = -DM.inf(ng4)
    
    for k in range(1,N+1,1):
        w_lb[nxvw*k : nxvw*k+n] = xmin
        w_ub[nxvw*k : nxvw*k+n] = xmax
        w_lb[nxvw*k-p-n_w : nxvw*k-n_w] = vmin 
        w_ub[nxvw*k-p-n_w : nxvw*k-n_w] = vmax 
        w_lb[nxvw*k-n_w : nxvw*k] = wmin
        w_ub[nxvw*k-n_w : nxvw*k] = wmax
        
        
        if yFree is False:
            g_lb[ng+(k-1)*p: ng+k*p] = ymin + 0.5*ymin
            g_ub[ng+(k-1)*p: ng+k*p] = ymax + 0.5*ymax

    g = vertcat(g, g1, g4, g5)
    
    nlp = {'x':w_opt, 'p':par, 'f':f_obj, 'g':g}

    solver = nlpsol('solver', 'ipopt', nlp, sol_opts)
    
    return [solver, w_lb, w_ub, g_lb, g_ub]
