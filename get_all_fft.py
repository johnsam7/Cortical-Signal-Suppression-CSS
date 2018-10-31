# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:39:38 2017

@author: ju357
"""

from getFFT import getFFT
import numpy as np

def get_all_fft(data,Fs):
    Nmag = np.asarray(data)
    c = getFFT(Nmag[0,:], Fs)
    y = np.zeros(np.size(c[1]))
    for k in range(0,len(Nmag)):
        a = getFFT(Nmag[k,:], Fs)
        y = y + np.abs(a[1])
    f = a[0]
            
    return [y,f]
    
