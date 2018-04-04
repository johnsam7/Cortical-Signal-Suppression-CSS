# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:39:38 2017

@author: ju357
"""

from getFFT import getFFT
import numpy as np
import matplotlib.pyplot as plt

def get_all_fft(data,Fs):
    Nmag = np.asarray(data)
    c = getFFT(Nmag[0,:], Fs)
    mata = np.zeros(np.size(c[1]))
    for k in range(0,len(Nmag)):
        a = getFFT(Nmag[k,:], Fs)
        mata = mata + np.abs(a[1])
        
    
        
    return [mata,a[0]]
    
