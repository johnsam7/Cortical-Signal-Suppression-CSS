# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:40:33 2017

@author: ju357
"""

import numpy as np

def filter_cort(subspace_data1, subspace_data2, filterdata, r=6):
    
    Qg,rg = np.linalg.qr(subspace_data2.T)
    Qm,rm = np.linalg.qr(subspace_data1.T)
    C = np.dot(Qg.T,Qm)
    Y,S,Z = np.linalg.svd(C)
    u = np.matrix(np.dot(Qg,Y))
    
    Nmag = np.matrix(filterdata)
    for i in range(r):
        NpProj = np.dot(Nmag,u[:,i]);
        Nmag = Nmag - np.dot(NpProj,u[:,i].T);
    

        
    return np.asarray(Nmag)