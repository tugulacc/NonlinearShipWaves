# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 17:15:03 2020

@author: Claudia
"""

import numpy as np
from numpy import sin, cos, exp, pi, sqrt, inf
import scipy.integrate as integrate
#from scipy.integrate.quadpack import quad

def zeta_peters(x,y,F,eps):
    g = lambda k,theta: F**2*k*sin(k*cos(theta))+cos(theta)*cos(k*cos(theta))
    xi = lambda Lambda: sqrt(Lambda**2+1)/F**2
    ## integral coefficients
    c0 = eps*np.heaviside(x,0)/pi
    c1 = -eps*F**2*np.sign(x)/pi**2
    ## integrand functions
    i0 = lambda Lambda: c0*xi(Lambda)*exp(-F**2*xi(Lambda)**2)*cos(x*xi(Lambda))*cos(y*Lambda*xi(Lambda))
    i1 = lambda k, theta: c1*cos(theta)*k*exp(-k*abs(x))*cos(k*y*sin(theta))*g(k,theta)/(F**4*k**2+cos(theta)**2)
    ## integral over Lambda
    I0 = integrate.quad(i0, -inf, inf)[0]
    ## double integral over theta and k
    I1 = integrate.dblquad(i1, 0., pi/2., lambda k: 0., lambda k: inf)[0]
    return I0+I1

def zeta_havelock(x,y,F,eps):
    # accurate for |x|<4
    k0 = lambda theta: 1/(F**2*cos(theta)**2)
    c  = -eps*F**2*x/(2*pi*sqrt(x**2+y**2+1)**3)
    i0 = lambda theta: (eps/pi*F**2)*np.reciprocal(cos(theta))**3*exp(-k0(theta))*cos(k0(theta)*x*cos(theta))*cos(k0(theta)*y*sin(theta))
    I0 = integrate.quad(i0, 0., pi/2.)[0]
    ## double integral over theta and k
    i1 = lambda k, theta: eps*F**2/(pi**2)*cos(theta)*k**2*exp(-k)*sin(k*x*cos(theta))*cos(k*y*sin(theta))/(k-k0(theta))
    I1 = integrate.dblquad(i1, 0., pi/2., lambda k: 0., lambda k: inf)[0]
    return c+I0+I1

def linZeta(x,y,M,N,F,eps):
    zeta = np.zeros((M,N))
    for jj in range(M):
        for ii in range(N):
            zeta[jj,ii] = zeta_peters(x[ii],y[jj],F,eps)
        
    zeta1 = zeta[:,0].reshape(M,1)
    zetax = np.gradient(zeta,axis=1)
    return zeta1, zetax