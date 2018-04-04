# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:40:33 2017

@author: ju357
"""

import numpy as np

def filter_cort(magdata,graddata,Ndim):
    
    Qg,r = np.linalg.qr(graddata.T)
    Qm,r = np.linalg.qr(magdata.T)
    C = np.dot(Qg.T,Qm)
    Y,S,Z = np.linalg.svd(C)
    u = np.matrix(np.dot(Qg,Y))
    
    Nmag = np.matrix(magdata)
    for i in range(Ndim):
        NpProj = np.dot(Nmag,u[:,i]);
        Nmag = Nmag - np.dot(NpProj,u[:,i].T);
    

        
    return np.asarray(Nmag)