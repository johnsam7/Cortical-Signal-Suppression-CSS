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
import SECS
from SECS import filter_cort
import getFFT
from getFFT import getFFT
from get_all_fft import get_all_fft as get_ch_fft


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
subcortical_signal = filter_cort(mag_data_raw,grad_data,1)        

#%%
#High- and lowpass raw and processed data with 5:th order Butterworth filter 
#to separate cortical from subcortical signals and see how they are affected by SECS
#(Fig 3)

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




#Calculate PSD

fft_lengt = np.size(getFFT(subcortical_signal[0,:], Fs)[0])
raw_fft = np.zeros([fft_lengt])
subcortical_fft = np.zeros([fft_lengt])

for k in range(0,102):
    a = getFFT(subcortical_signal[k,:], Fs)
    b = getFFT(mag_data_raw[k,:],Fs)
    subcortical_fft = subcortical_fft + np.abs(a[1])
    raw_fft = raw_fft + np.abs(b[1])

plt.figure()
linea, = plt.plot(a[0],raw_fft*10**12, label = 'raw')
lineb, = plt.plot(a[0],subcortical_fft*10**12,'--', label = 'processed')
plt.legend()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Spectral density $[pT/\sqrt{Hz}]$')


#%%
#Moderate SNR simulations (SNR=0.5dB)
snr =0.5  
evoked = simulate_evoked(fwd, stc_all, info, cov, snr, iir_filter=None, random_state=7)
evoked.pick_types(meg=True, eeg=False)

mag_ev = evoked.copy()
grad_ev = evoked.copy()
grad_ev = grad_ev.pick_types(meg='grad')
mag_ev = mag_ev.pick_types(meg='mag')

mag_data_raw = mag_ev.data
grad_data = grad_ev.data



#%%
#Graphs showing dependency of dimensionality of common subspace removed, Ndim
#(Fig 4)

f = a[0]
y = raw_fft
yf = subcortical_fft

indd = np.argmin(np.abs(fd-f))
yd = np.max(y[indd-6:indd+6])
yfd = np.max(yf[indd-6:indd+6])

indc = np.argmin(np.abs(fc-f))
yc = np.max(y[indc-6:indc+6])
yfc = np.max(yf[indd-6:indd+6])

SNR = yd/np.sum(np.sort(y)[np.int(0.1*len(y)):np.int(0.90*len(y))])
SNRf = yfd/np.sum(np.sort(yf)[np.int(0.1*len(yf)):np.int(0.90*len(yf))])

SNR_rel = yd/yc
SNR_rel_f = yfd/yfc

SNR_v = np.zeros([102,1])
SNR_rel_v = np.zeros([102,1])
SNR_v_d = np.zeros([102,1])
SNR_v_c = np.zeros([102,1])


for m in range(102):
    subcortical_signal = filter_cort(mag_data_raw,grad_data,m)
    fft_lengt = np.size(getFFT(subcortical_signal[0,:], Fs)[0])
    raw_fft = np.zeros([fft_lengt])
    subcortical_fft = np.zeros([fft_lengt])
    for k in range(0,102):
        a = getFFT(subcortical_signal[k,:], Fs)
        b = getFFT(mag_data_raw[k,:],Fs)
        subcortical_fft = subcortical_fft + np.abs(a[1])
        raw_fft = raw_fft + np.abs(b[1])
    y = raw_fft
    yf = subcortical_fft
    indd = np.argmin(np.abs(fd-f))
    yd = np.max(y[indd-6:indd+6])
    yfd = np.max(yf[indd-6:indd+6])
    indc = np.argmin(np.abs(fc-f))
    yc = np.max(y[indc-6:indc+6])
    yfc = np.max(yf[indc-6:indc+6])
    SNR = yd/np.sum(np.sort(y)[np.int(0.1*len(y)):np.int(0.90*len(y))])
    SNRf = yfd/np.sum(np.sort(yf)[np.int(0.1*len(yf)):np.int(0.90*len(yf))])
    SNR_rel = yd/yc
    SNR_rel_f = yfd/yfc
    SNR_rel_v[m] = SNR_rel_f
    SNR_v[m] = SNRf

Qg,r = np.linalg.qr(grad_data.T)
Qm,r = np.linalg.qr(mag_data_raw.T)
C = np.dot(Qg.T,Qm)
Y,S,Z = np.linalg.svd(C)
u = np.matrix(np.dot(Qg,Y))

#    
for k in range(102):
    y = getFFT(np.asarray(u[:,k])[:,0],Fs)[1]
    yd = np.max(y[indd-6:indd+6])
    indc = np.argmin(np.abs(fc-f))
    yc = np.max(y[indc-6:indc+6])
    SNR_v_d[k] = yd/np.sum(np.sort(y)[np.int(0.1*len(y)):np.int(0.90*len(y))])
    SNR_v_c[k] = yc/np.sum(np.sort(y)[np.int(0.1*len(y)):np.int(0.90*len(y))])    
#    SNR_rel_v[k] = yd/yc
        
plt.figure()
plt.subplot(2,1,1)
SNR_sc = plt.plot(SNR_v/SNR,color='k')
ylim = (SNR_sc[0].axes.get_ylim())
plt.plot((6,6),ylim,'--',color='r')
plt.ylabel('$SNR\ change$')
plt.xlim((0,20))
plt.ylim(ylim)
ticks = [0, 3, 6, 9, 12, 15, 18]
plt.xticks(ticks)
plt.gca().get_xticklabels()[2].set_color('red')
plt.grid()


plt.subplot(2,1,2)
SNR_rel_hold = plt.plot(SNR_rel_v/SNR_rel,color='k')
ylim = (SNR_rel_hold[0].axes.get_ylim())
plt.plot((6,6),ylim,'--',color='r')
plt.ylabel('$SNR_{rel}$')
plt.xlabel('$N_{dim}$')
plt.xlim((0,20))
plt.ylim(ylim)
plt.xticks(ticks)
plt.gca().get_xticklabels()[2].set_color('red')
plt.grid()
        
#plt.plot(SNR_v/SNR)
plt.figure()
myc = plt.plot(SNR_v_c/np.max(SNR_v_c),label = 'Normalized cortical SNR')
myd = plt.plot(SNR_v_d/np.max(SNR_v_d),label = 'Normalized subcortical SNR')
mys = plt.plot(S, label = 'Correlation')
plt.legend()
plt.xlabel('Common subspace vector element $\{v_i\}_i$')
plt.grid()


#%% Sensitivity Brain plots (Fig 1)

src = fwd['src']
h1 = src[0]
h2 = src[1]
all_vertices = [h1['vertno'],h2['vertno']]
stc_data = 1
stc_d = np.tile(stc_data,(len(h1['vertno'])+len(h2['vertno']),1))
stc_all = mne.SourceEstimate(stc_d, vertices=all_vertices, tmin=0, tstep=1, subject='fsaverage')
picks_grad = mne.pick_types(fwd['info'], meg='grad', exclude = 'bads')
picks_mag = mne.pick_types(fwd['info'], meg='mag', exclude = 'bads')
picks_eeg = mne.pick_types(fwd['info'], meg=False, eeg=True, exclude = 'bads')

G_grad = fwd['sol']['data'][picks_grad,:]
G_mag = fwd['sol']['data'][picks_mag,:]
G_eeg = fwd['sol']['data'][picks_eeg,:]

stc_eeg = stc_all.copy()
stc_grad = stc_all.copy()
stc_mag = stc_all.copy()

for k in range(len(h1['vertno'])):
    stc_eeg.data[k] = np.linalg.norm(G_eeg[:,k])
    stc_grad.data[k] = 10**7*np.linalg.norm(G_grad[:,k])
    stc_mag.data[k] = 10**7*np.linalg.norm(G_mag[:,k])

#stc_eeg.data[k] = stc_eeg.data[:]/np.max(stc_eeg.data[:])

brain = stc_eeg.plot(subject='sample', subjects_dir=subjects_dir,hemi='split', views=['lat', 'med'],time_viewer=True)
#brain.scale_data_colormap(300,500,700,transparent=False)
brain.scale_data_colormap(0,np.max(stc_eeg.data[:])/2,np.max(stc_eeg.data[:]),transparent=False)
brain = stc_grad.plot(subject='sample', subjects_dir=subjects_dir,hemi='split', views=['lat', 'med'],time_viewer=True)
brain.scale_data_colormap(0,np.max(stc_grad.data[:])/2,np.max(stc_grad.data[:]),transparent=False)
brain = stc_mag.plot(subject='sample', subjects_dir=subjects_dir,hemi='split', views=['lat', 'med'],time_viewer=True)
brain.scale_data_colormap(0,np.max(stc_mag.data[:])/2,np.max(stc_mag.data[:]),transparent=False)

  


#%% SECS Brain plot (Fig 5)


fd = 223
fc = 40

def data_fun1(times):
    return 1e-8 * np.cos(times*2*np.pi*fc)#For whole brain simulation - this is 1e-8

def data_fun2(times):
    return 1e-7 * np.cos(times*2*np.pi*fd)#np.ones(len(times))

src = fwd['src']
h1 = src[0]
h2 = src[1]
all_vertices = [h1['vertno'],h2['vertno']]
tmin = 0
tmax = 1
times = np.linspace(tmin,tmax,(tmax-tmin)*Fs+1)
tstep = np.diff(times)[0]
stc_data = data_fun1(times)
stc_d = np.tile(stc_data,(len(h1['vertno'])+len(h2['vertno']),1))
stc_all = mne.SourceEstimate(stc_d, vertices=all_vertices, tmin=tmin, tstep=tstep, subject='fsaverage')
   
#stc_all.plot( subject='sample', subjects_dir=subjects_dir,hemi='split', views=['lat', 'med'],time_viewer=True)
#k=1000 and 1500 - deep vertices. Ndim=3-6 optimal for snr=0.05. Choose Ndim=6 to highlit geometric difference.

dic_res_2 = {}
for k in range(len(h1['vertno'])):

    stc_v = stc_all.copy()
    stc_v.data[k,:] = data_fun2(times)
    picks = mne.pick_types(raw.info, meg=True, exclude = 'bads')
    snr = 0.5  # dB
    evoked = simulate_evoked(fwd, stc_v, raw.info, cov, snr, iir_filter=None, random_state=18)
    evoked.pick_types(meg=True, eeg=False)
    grad_evoked = evoked.copy().pick_types(meg='grad')
    grad_data = grad_evoked.data
    mag_data = evoked.pick_types(meg='mag').data
    proc_data = filter_cort(mag_data,grad_data,Ndim = 6)
    
    [PSD_raw,frq] = get_ch_fft(mag_data,Fs)
    [PSD_proc,frq] = get_ch_fft(proc_data,Fs)
    ind_d = np.argmin(np.abs(fd-frq))
    ind_c = np.argmin(np.abs(fc-frq))
    
    ratio = (PSD_proc[ind_d]/PSD_raw[ind_d])*(PSD_raw[ind_c]/PSD_proc[ind_c])
    
    dic_res_2[h1['vertno'][k]] = ratio

    print('\n' + str(float(k)/float(len(h1['vertno']))*100.0) + '% complete' + '\n')


plt.figure()
plt.plot(frq,PSD_raw)
plt.plot(frq,PSD_proc)

stc_plot = stc_all.copy()
stc_plot.data = np.zeros(stc_plot.data.shape)
vertnos = stc_plot.vertices[0]
k=0

for vert in h1['vertno']:
    stc_plot.data[k,0] = dic_res_2[vert]
    k = k+1

brain = stc_plot.plot(subject='sample', subjects_dir=subjects_dir,hemi='split', views=['lat', 'med'],time_viewer=True)
brain.scale_data_colormap(0,40,80,transparent=False)

#Fmin=0, Fmid = 40, Fmax = 80, smoothing steps = 10, transparent=False


































