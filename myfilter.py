import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt


def myfilter(data,kind,cutoff,fs):
    
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype=kind, analog=False)
        return b, a
    def butter_highpass_filter(data, cutoff, fs, order=5):
        b, a = butter_highpass(cutoff, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y
    filtered_data = butter_highpass_filter(data,cutoff,fs)
    return filtered_data


