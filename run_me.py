# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:52:34 2017

@author: John G. Samuelsson

Simple simulations to showcase efficacy of SECS algorithm from MNE-python "sample" data folder
"""

    

#Import packages and SECS
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.simulation import simulate_sparse_stc, simulate_evoked
import os.path as op
print(__doc__)
from myfilter import myfilter

from SECS import SECS


#%%
#Load sample subject data and labels from local labels folder

data_path = sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
raw = mne.io.read_raw_fif(data_path + '/MEG/sample/sample_audvis_raw.fif')
proj = mne.read_proj(data_path + '/MEG/sample/sample_audvis_ecg_proj.fif')
raw.info['projs'] += proj
raw.info['bads'] = ['MEG 2443', 'EEG 053']

fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-no-filter-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'
trans_fname = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
bem_fname = op.join(subjects_dir, 'sample', 'bem', 'sample-5120-bem-sol.fif')

fwd = mne.read_forward_solution(fwd_fname, force_fixed=True, surf_ori=True)
fwd = mne.pick_types_forward(fwd, meg=True, eeg=True, exclude=raw.info['bads'])
cov = mne.read_cov(cov_fname)
info = mne.io.read_info(ave_fname)

hiplab = mne.read_label('./labels/lh.parahippocampal.label') 
postcenlab = mne.read_label('./labels/lh.G_postcentral.label') 
precenlab = mne.read_label('./labels/lh.G_precentral.label') 

Fs = raw.info['sfreq']



  
#%%
#Simulate one subcortical dipole oscillating with frequency fd=223 Hz in the parahippocampal area
#and one cortical dipole with frequency fc=40 Hz in area around central sulcus

fd = 223
fc = 40

def data_fun_cortical(times):
    return 1e-7 * np.cos(times*2*np.pi*fc)

def data_fun_subcortical(times):
    return 1e-7 * np.cos(times*2*np.pi*fd)

times = np.arange(3000, dtype=np.float) / Fs

cor_lab = [postcenlab,precenlab,hiplab]

stc_all = simulate_sparse_stc(fwd['src'], n_dipoles=3, times=times,
                          random_state=42, labels=cor_lab, data_fun=data_fun_cortical)#cortical label
stc_cortical = simulate_sparse_stc(fwd['src'], n_dipoles=2, times=times,
                          random_state=42, labels=[postcenlab,precenlab], data_fun=data_fun_cortical)#cortical label
                          
#Endow vertices in the deep area with frequency fd instead of fc
for k in range(0,len(stc_all.vertices[0])):
    if not stc_all.vertices[0][k] in stc_cortical.vertices[0]:
        stc_all.data[k,:] = data_fun_subcortical(times)
        
picks = mne.pick_types(raw.info, meg=True, exclude = 'bads')
snr =120  
evoked = simulate_evoked(fwd, stc_all, info, cov, snr, iir_filter=None, random_state=7)
evoked.pick_types(meg=True, eeg=False)

mag_ev = evoked.copy()
grad_ev = evoked.copy()
grad_ev = grad_ev.pick_types(meg='grad')
mag_ev = mag_ev.pick_types(meg='mag')

mag_data_raw = mag_ev.data
grad_data = grad_ev.data

#Remove cortical signals with SECS
subcortical_signal = SECS(mag_data_raw,grad_data,1)        

#%%
#High- and lowpass raw and processed data with 5:th order Butterworth filter 
#to separate cortical from subcortical signals and see how they are affected by SECS

subcortical_signal = np.asarray(subcortical_signal)
deep_comp_org = myfilter(mag_data_raw,'high',cutoff=100,fs=Fs)
cort_comp_org = myfilter(mag_data_raw,'low',cutoff=100,fs=Fs)
deep_comp_filt = myfilter(subcortical_signal,'high',cutoff=100,fs=Fs)
cort_comp_filt = myfilter(subcortical_signal,'low',cutoff=100,fs=Fs)


plt.figure()

plt.subplot(2,1,1)
myplto = plt.plot(times[0:200],deep_comp_org[60,0:200]*10**15,label='raw')
mypltf = plt.plot(times[0:200],deep_comp_filt[60,0:200].T*10**15,'--',label='processed')
plt.ylabel('subcortical signal [fT]')

plt.legend()

plt.subplot(2,1,2)
plt.plot(times[0:800],cort_comp_org[60,0:800]*10**15)
plt.plot(times[0:800],cort_comp_filt[60,0:800].T*10**15,'--')

plt.xlabel('time [s]')
plt.ylabel('cortical signal [fT]')

