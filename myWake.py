# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:34:24 2020

@author: Claudia
"""
import numpy as np
import scipy as sc
import funcs

def lnSoln(s_lims, t_lims, A, B, C):
    [s_max, s_min] = s_lims
    [t_max, t_min] = t_lims
    
    F1 = lambda s, t: t/np.sqrt(A)*np.log(2*A*s + B*t + 2*np.sqrt(A*(A*s**2 + B*s*t + C*t**2)))
    F2 = lambda s, t: s/np.sqrt(C)*np.log(2*C*t + B*s + 2*np.sqrt(C*(A*s**2 + B*s*t + C*t**2)))
    evlt = lambda t, func: func(s_max, t) - func(s_min, t)
    I = evlt(t_max,F2) - evlt(t_min,F2)
    if t_max != 0.:
        I += evlt(t_max, F1)
    if t_min !=0.:
        I -= evlt(t_min,F1)
    
    return I

def wake(u,x,y,dx,N,M,n,F,epsi):
    [phi1, phix, zet1, zetx] = funcs.reshapingUnknowns(u,M,N)
    phi = funcs.allVals(phi1,phix,dx,M,N)
    zet = funcs.allVals(zet1,zetx,dx,M,N)
    phiy = funcs.yDerivs(phi,y,M,N)
    zety = funcs.yDerivs(zet,y,M,N)
    phixx1 = funcs.xDerivs(phix,dx)
    zetxx1 = funcs.xDerivs(zetx,dx)

    # computing half-mesh points
    phiH = funcs.halfMesh(phi) 
    phixH = funcs.halfMesh(phix)
    phiyH = funcs.halfMesh(phiy)
    zetH = funcs.halfMesh(zet) 
    zetxH = funcs.halfMesh(zetx) 
    zetyH = funcs.halfMesh(zety) 
    xH = (x[1:]+x[:-1])/2.
    
    # Try with pressure
    '''
    P = np.zeros((M,N-1))

    xInd = np.array((np.abs(xH)<1),'double')
    yInd = np.array((np.abs(y)<1),'double')
    Pind = np.outer(xInd,yInd)
    P = np.exp(1./(xH**2.-1.)+1./(y**2.-1.))*Pind.T
    '''
    # Bernoulli equation
    eqnSurf = .5*(((1+zetxH**2)*phiyH**2+(1+zetyH**2)*phixH**2-2*zetxH*zetyH*phixH*phiyH)/(1+phixH**2+phiyH**2)-1)+ zetH/F**2 #+ epsi*P
    #print(eqnSurf)
    eqns = np.zeros(np.shape(u))
    eqnsI = np.zeros((M,N-1))
    
    A = np.square(zetxH) + 1.
    B = 2.*np.multiply(zetxH,zetyH)
    C = np.square(zetyH) + 1.
    #print(f"{A}\n{B}\n{C}\n")
    
    for jj in range(M):
        for ii in range(N-1):
            # evaluate Kernal functions and S2 
            s2denom = lambda c: np.sqrt(A[jj,ii]*(x-xH[ii])**2. + c*B[jj,ii]*(x-xH[ii])*(y-c*y[jj])+C[jj,ii]*(y-c*y[jj])**2.)
            k2denom = lambda c: np.sqrt((x-xH[ii])**2 + (y-c*y[jj])**2 + (zet-zetH[jj,ii])**2)
            k1numer = lambda c: zet-zetH[jj,ii] - (x-xH[ii])*zetx - (y-c*y[jj])*zety
        
            K1 = k1numer(1.)/k2denom(1.)**3 + k1numer(-1.)/k2denom(-1.)**3
            K2 = 1./k2denom(1.) + 1/k2denom(-1.)
            S2 = 1./s2denom(1.) + 1/s2denom(-1.)
            #print(K2)
            #print(zet-zetH[jj,ii])
            I1intgrnd = (phi-phiH[jj,ii]-x-xH[ii])*K1
            I1 = np.trapz(np.trapz(I1intgrnd,x).T,y.T)
        
            I2pintgrnd = zetx*K2 - zetx[jj,ii]*S2
            I2p = np.trapz(np.trapz(I2pintgrnd,x).T,y.T)
            # evaluate analytical solution for I2"
            slims = [x[-1]-xH[ii], x[0]-xH[ii]]
            tlims = lambda c: [y[-1,0]-c*y[jj,0], y[0,0]-c*y[jj,0]]
            I2pp = lnSoln(slims,tlims(1.),A[jj,ii],B[jj,ii],C[jj,ii]) + lnSoln(slims,tlims(-1.),A[jj,ii],-B[jj,ii],C[jj,ii])
            #print(f"(m={jj},n={ii})\n{I2pp}\n")
            I2 = I2p + zetxH[jj,ii]*I2pp
        
            src = epsi/np.sqrt(xH[ii]**2.+ y[jj]**2.+ (zetH[jj,ii]+ 1)**2.) 
        
            eqnsI[jj,ii] = I1 + I2 -2*np.pi*(phiH[jj,ii]-xH[ii]) - src

    # enforcing the boundary condition
    bc1 = x[0]*(phix[:,0]-1.)+n*(phi[:,0]-x[0])       
    bc2 = x[0]*(phixx1)+n*(phix[:,0]-1.)
    bc3 = x[0]*(zetx[:,0])+n*(zet[:,0])
    bc4 = x[0]*(zetxx1)+n*(zetx[:,0])
    eqnsBC = np.vstack((bc1, bc2, bc3, bc4)).reshape(4*M,1)
    eqns = np.vstack((eqnsI.reshape(M*(N-1),1),eqnSurf.reshape(M*(N-1),1),eqnsBC))
    eqns = eqns[:,0]
    #print(eqnsI)
    return eqns