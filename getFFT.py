# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 12:18:58 2017
does a PSD
@author: ju357
"""
import numpy as np
import matplotlib.pylab as plt


    
def getFFT(y,Fs):
    Ft = np.fft.fft(y)
    Ft = np.abs(Ft[0:len(Ft)/2+1])**2;
    Y = np.sqrt(Ft/(len(y)*Fs));
    f = Fs/2*np.linspace(0,1,len(y)/2+1);

    outp = [f,Y]
    return outp
