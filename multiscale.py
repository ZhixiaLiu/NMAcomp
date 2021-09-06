import sys
from prody import GNM,parsePDB,ANM,calcTempFactors
from pylab import *
import numpy as np
from sklearn.linear_model import LinearRegression
from functools import reduce

def multiscale_gamma(eta=3,v=3,kernel='exponential'):
    kernel = kernel.lower()
    if kernel == 'exponential':
        def gamma(dist2,i,j):
            return np.exp(-(np.sqrt(dist2)/eta)**v)
    elif kernel == 'lorentz':
        def gamma(dist2,i,j):
            return 1/(1+(np.sqrt(dist2)/eta)**v)
    elif kernel == 'ilf':
        def gamma(dist2,i,j):
            if dist2<=eta**2:
                return 1
            else:
                return 0
    else:
        raise ValueError("Don't have this kernel.")
    return gamma

class MultiscaleGNM(GNM):
    
    def buildMultiscaleKirchhoff(self,atoms,scale,eta_n,bfactor,kernel='exponential',v=3):
        if isinstance(kernel,str):
            kernel = [kernel]*scale
        # build n kirchhoffs
        diag_n=[]
        kirchhoff_n = []
        for i in range(scale):
            self.buildKirchhoff(atoms,cutoff=9999, gamma=multiscale_gamma(eta=eta_n[i],v=v,kernel=kernel[i]))
            kirchhoff_n.append(self.getKirchhoff())
            diag_n.append(np.diagonal(self.getKirchhoff()))
        diag_n=np.array(diag_n)
        kirchhoff_n=np.array(kirchhoff_n)
        # linear regression to find the coeff
        reg = LinearRegression(fit_intercept=False)
        reg.fit(diag_n.T,1/bfactor)
        kirchhoff = sum(kirchhoff_n.T*reg.coef_,axis = 2)
        self.setKirchhoff(kirchhoff)
    
    def buildMultiscaleKirchhoffFromKernels(self,kirchhoff_n,bfactor):
        diag_n=[]
        for m in kirchhoff_n:
            diag_n.append(np.diagonal(m))
        diag_n=np.array(diag_n)
        kirchhoff_n=np.array(kirchhoff_n)
        # linear regression to find the coeff
        reg = LinearRegression(fit_intercept=False)
        reg.fit(diag_n.T,1/bfactor)
        kirchhoff = sum(kirchhoff_n.T*reg.coef_,axis = 2)
        self.setKirchhoff(kirchhoff)

class MultiscaleANM(ANM):
    def buildMultiscaleHessian(self,atoms,scale,eta_n,bfactor,kernel='exponential',v=3):
        if isinstance(kernel,str):
            kernel = [kernel]*scale
        # build n kirchhoffs
        mu_n=[]
        hessian_n = []
        for i in range(scale):
            self.buildHessian(atoms,cutoff=9999, gamma=multiscale_gamma(eta=eta_n[i],v=v,kernel=kernel[i]))
            hessian_n.append(self.getHessian())
            diag = np.diagonal(self.getHessian())
            mu_n.append(sum(diag.reshape(-1,3),axis = 1))
        mu_n=np.array(mu_n)
        hessian_n=np.array(hessian_n)
        # linear regression to find the coeff
        reg = LinearRegression(fit_intercept=False)
        reg.fit(mu_n.T,1/bfactor)
        hessian = sum(hessian_n.T*reg.coef_,axis = 2)
        self.setHessian(hessian)
        
    def buildMultiscaleHessianFromKernels(self,hessian_n,bfactor):
        mu_n=[]
        for m in hessian_n:
            diag = np.diagonal(m)
            mu_n.append(sum(diag.reshape(-1,3),axis = 1))
        mu_n=np.array(mu_n)
        hessian_n=np.array(hessian_n)
        # linear regression to find the coeff
        #print(mu_n.shape,bfactor.shape)
        reg = LinearRegression(fit_intercept=False)
        reg.fit(mu_n.T,1/bfactor)
        hessian = sum(hessian_n.T*reg.coef_,axis = 2)
        self.setHessian(hessian)
