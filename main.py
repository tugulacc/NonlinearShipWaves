# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 04:06:54 2020

@author: Claudia
"""


import numpy as np
import scipy as sc
from scipy import optimize 
import matplotlib.pyplot as plt
#import jacfuncs
import pltfuncs
import funcs
import myWake
import linSoln

# Declaring Constants
N, M = 33, 13
dx, dy = 0.4, 0.4
x1, y1 = -3., dy*M/2.
F, n, epsi = 0.7, 0.05, 1.

# Defining the domain
x = dx*sc.r_[:N] + x1
y = np.reshape((sc.r_[0:M]*dy - M/2.*dy),(M,1)) + y1 #dy*sc.c_[:M] 

#J = jacfuncs.Jacobian(x,y,dx,dy,N,M,F,n)
#pltfuncs.plot_jac(J)


### INITIAL GUESS ###
phi1 = x1*np.ones((M,1))
phix = np.ones((M,N))
#[zeta1, zetax] = linSoln.linZeta(x,y,M,N,F,epsi)
zeta1 = np.zeros((M,1))
zetax = np.ones((M,N))
#print(f"\u03D5\u2081:{phi1.shape}\t\u03D5\u2093:{phix.shape}\n\u03B6\u2081:{zeta1.shape}\t\u03B6\u2093:{zetax.shape}\n")

uInit = funcs.guessUnknowns(phi1,phix,zeta1,zetax,M,N) # initial guess for vector of unknowns
#print(f"u:{uIG.shape}, 2M(N+1)={2*M*(N+1)}")

uNew = optimize.fsolve(myWake.wake,uInit,args=(x,y,dx,N,M,n,F,epsi))


allZet = uNew[M+N*M:]
indx = (N+1)*np.arange(M)
zet1 = allZet[indx]
zetx = np.delete(allZet,indx).reshape(M,N)
zeta = funcs.allVals(zet1,zetx,dx,M,N)

'''
plt.plot(x,zeta[0,:])
plt.ylabel(r'$\zeta (x,0)$')
plt.xlabel(r'$x$')
plt.show()'''

pltfuncs.plot_surf(x,y,zeta)