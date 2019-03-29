#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:45:19 2019

@author: Zach Sheldon
"""
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import auc

## GOAL: explore direction and coherence sensitivity of neurons from MT (visual cortical area that is 
## known to be sensitive to coherent large-field motion). Single cells can be found that are tuned to 
## a certain direction of motion. These cells are sensitive to a large visual field and respond even 
## if the motion is not completely coherent. We will assume that, in an experimental setting, the monkey 
## uses these cells to make decisions based on the motion direction. The monkey is tested on its ability 
## to identify the direction of motion of a stimulus. We will extract the neurometric and psychometric data 
## from this experiment to see if a single neuron contains enough information to let the monkey make this decision.

## load in data - cellArrayOfCells structure contains data from 5 MT neurons (recorded by members of Josh Gold's lab @ Penn)
## Recordings were made while the monkey was awake and fixating on a central spot,
## while a random-dot stimulus was shown in the neuron’s receptive field. 
## The stimulus consisted of motion at a fixed speed in one of several possible directions 
## and one of several possible coherence levels.
data_dict = io.loadmat('HW5_data.mat')
cellArrayOfCells = data_dict['cellArrayOfCells']
# rows are individual neurons and columns are either direction tuning data or coherence dependence data
dir_tuning = cellArrayOfCells[:, 0]
coh_dependence = cellArrayOfCells[:, 1] # to be used later in script

############################ DIRECTION TUNING ################################

# create new data structure to store neuron/trial information 
cell_data = [] # data for each trial
cell_spikes = [] # spike times for each trial
for i in range(0, len(dir_tuning)):
    curr_dict = dir_tuning[i]
    curr_ecodes = curr_dict['ecodes']
    curr_spikes = curr_dict['spikes'][0][0] # matlab struct is wrapped in arrays
    cell_data.append(curr_ecodes[0][0][0][0][1]) # contains stimulus onset/offset info, coherence level, motion direction, and task score
    curr_spikes_arr = []
    for j in range(0, len(curr_spikes)):
        curr_spikes_arr.append(curr_spikes[j][0])
    cell_spikes.append(curr_spikes_arr)
num_cells = len(cell_spikes)

## 1) Compute and plot the mean spike rates (in polar coordinates) as a function of motion direction

# create a list of dictionaries that map motion direction to a list containing total number of spikes 
# and total interval length in order to calculate mean firing rate 
motion_directions_dict1 = {-0.0:[0, 0], 45.0:[0, 0], 90.0:[0, 0], 135.0:[0, 0], 180.0:[0, 0], 225.0:[0, 0], 270.0:[0, 0], 315.0:[0, 0]}
motion_directions_dict2 = {-0.0:[0, 0], 45.0:[0, 0], 90.0:[0, 0], 135.0:[0, 0], 180.0:[0, 0], 225.0:[0, 0], 270.0:[0, 0], 315.0:[0, 0]}
motion_directions_dict3 = {-0.0:[0, 0], 45.0:[0, 0], 90.0:[0, 0], 135.0:[0, 0], 180.0:[0, 0], 225.0:[0, 0], 270.0:[0, 0], 315.0:[0, 0]}
motion_directions_dict4 = {-0.0:[0, 0], 45.0:[0, 0], 90.0:[0, 0], 135.0:[0, 0], 180.0:[0, 0], 225.0:[0, 0], 270.0:[0, 0], 315.0:[0, 0]}
motion_directions_dict5 = {-0.0:[0, 0], 45.0:[0, 0], 90.0:[0, 0], 135.0:[0, 0], 180.0:[0, 0], 225.0:[0, 0], 270.0:[0, 0], 315.0:[0, 0]}
motion_directions_arr = [motion_directions_dict1, motion_directions_dict2, motion_directions_dict3, motion_directions_dict4, motion_directions_dict5]

# Get number of spikes and interval lengths for each motion direction for each cell
for i in range(0, num_cells):
    # get spiking data
    curr_all_spikes = cell_spikes[i]
    # get trials
    curr_num_trials = len(curr_all_spikes)
    # get dict to add values to 
    curr_dict = motion_directions_arr[i]
    for j in range(0, curr_num_trials):
        # get motion direction
        curr_mot_dir = cell_data[i][j][2]
        # get time interval
        curr_onset = cell_data[i][j][0] # stim onset in ms
        curr_offset = cell_data[i][j][1] # stim offset in ms
        curr_interval = (curr_offset - curr_onset) / 1000 # convert to sec
        # add number of spikes that occur within the interval
        curr_num_spikes = 0
        for k in range(0, len(curr_all_spikes[j])):
            if curr_all_spikes[j][k] >= curr_onset and curr_all_spikes[j][k] <= curr_offset:
                curr_num_spikes += 1
        # find correct key in dictionary
        for mot_dir in curr_dict:
            if mot_dir == curr_mot_dir:
                # add to totals
                curr_list_of_totals = curr_dict[mot_dir]
                curr_list_of_totals[0] += curr_num_spikes
                curr_list_of_totals[1] += curr_interval

# calculate mean spike rates by dividing total spikes by interval length
mean_spike_rates = [] # Hz (spikes/sec)
motion_directions = [-0.0, 45, 90, 135, 180, 225, 270, 315] # degrees
for i in range(0, num_cells):
    # get dictionary of totals
    curr_dict = motion_directions_arr[i]
    # store mean spike rates in list
    curr_means = []
    # iterate through motion directions
    for mot_dir in curr_dict:
        # get list of totals
        curr_list_of_totals = curr_dict[mot_dir]
        curr_means.append(curr_list_of_totals[0]/curr_list_of_totals[1])
    mean_spike_rates.append(curr_means)

# plot results for each cell
plt.figure(figsize=(6,6))
plt.axes(projection='polar')
plt.polar(motion_directions, mean_spike_rates[0], 'ko', markersize=10, label='Cell 1')
plt.title('Cell 1 - Motion Direction (Degrees) vs. Mean Spike Rate (Hz)')
plt.tight_layout()
#plt.savefig('polar_plot_cell1.png')
plt.figure(figsize=(6,6))
plt.axes(projection='polar')
plt.polar(motion_directions, mean_spike_rates[1],'ko', markersize=10, label='Cell 1')
plt.title('Cell 2 - Motion Direction (Degrees) vs. Mean Spike Rate (Hz)')
plt.tight_layout()
#plt.savefig('polar_plot_cell2.png')
plt.figure(figsize=(6,6))
plt.axes(projection='polar')
plt.polar(motion_directions, mean_spike_rates[2],'ko', markersize=10, label='Cell 1')
plt.title('Cell 3 - Motion Direction (Degrees) vs. Mean Spike Rate (Hz)')
plt.tight_layout()
#plt.savefig('polar_plot_cell3.png')
plt.figure(figsize=(6,6))
plt.axes(projection='polar')
plt.polar(motion_directions, mean_spike_rates[3],'ko', markersize=10, label='Cell 1')
plt.title('Cell 4 - Motion Direction (Degrees) vs. Mean Spike Rate (Hz)')
plt.tight_layout()
#plt.savefig('polar_plot_cell4.png')
plt.figure(figsize=(6,6))
plt.axes(projection='polar')
plt.polar(motion_directions, mean_spike_rates[4],'ko', markersize=10, label='Cell 1')
plt.title('Cell 5 - Motion Direction (Degrees) vs. Mean Spike Rate (Hz)')
plt.tight_layout()
#plt.savefig('polar_plot_cell5.png')

## 2) fit the relationship between mean spike rate M and motion direction theta using a von Mises function given by:
## M(theta) = A*exp(k*{cos(theta - phi) - 1}) where A is the value of the function at the preferred orientation
## phi, and k is a width parameter

# get preferred orientations and max firing rates at those orientations
pref_orientations = []
max_fr = []
for i in range(0, len(mean_spike_rates)):
    # get current mean spike rate
    curr_spike_rates = mean_spike_rates[i]
    # calculate max and index of max
    max_fr_val = max(curr_spike_rates)
    idx_max_fr = curr_spike_rates.index(max_fr_val)
    max_fr.append(max_fr_val)
    # get preferred orientation
    pref_orientations.append(motion_directions[idx_max_fr])

# von Mises functions for each cell
def von_Mises_cell1(theta, k):
    return (max_fr[0]*np.exp(k*(np.cos(np.array(theta) - np.array(pref_orientations[0])) - 1.0)))
def von_Mises_cell2(theta, k):
    return (max_fr[1]*np.exp(k*(np.cos(np.array(theta) - np.array(pref_orientations[1])) - 1.0)))
def von_Mises_cell3(theta, k):
    return (max_fr[2]*np.exp(k*(np.cos(np.array(theta) - np.array(pref_orientations[2])) - 1.0)))
def von_Mises_cell4(theta, k):
    return (max_fr[3]*np.exp(k*(np.cos(np.array(theta) - np.array(pref_orientations[3])) - 1.0)))
def von_Mises_cell5(theta, k):
    return (max_fr[4]*np.exp(k*(np.cos(np.array(theta) - np.array(pref_orientations[4])) - 1.0)))

# fit parameters
popt1, _ = curve_fit(von_Mises_cell1, motion_directions, mean_spike_rates[0])
popt2, _ = curve_fit(von_Mises_cell2, motion_directions, mean_spike_rates[1])
popt3, _ = curve_fit(von_Mises_cell3, motion_directions, mean_spike_rates[2])
popt4, _ = curve_fit(von_Mises_cell4, motion_directions, mean_spike_rates[3])
popt5, _ = curve_fit(von_Mises_cell5, motion_directions, mean_spike_rates[4])

# plot tuning curves 
plt.figure(figsize=(8,12))
plt.subplot(511)
plt.title('Tuning Curves')
plt.plot(motion_directions, mean_spike_rates[0], 'ko', label='Cell 1')
plt.plot(motion_directions, von_Mises_cell1(motion_directions, popt1[0]), 'r-', label='von Mises Fit')
plt.legend()
plt.ylabel('Mean Firing Rate (Hz)')
plt.subplot(512)
plt.plot(motion_directions, mean_spike_rates[1], 'ko', label='Cell 2')
plt.plot(motion_directions, von_Mises_cell2(motion_directions, popt2[0]), 'r-', label='von Mises Fit')
plt.legend()
plt.ylabel('Mean Firing Rate (Hz)')
plt.subplot(513)
plt.plot(motion_directions, mean_spike_rates[2], 'ko', label='Cell 3')
plt.plot(motion_directions, von_Mises_cell3(motion_directions, popt3[0]), 'r-', label='von Mises Fit')
plt.legend()
plt.ylabel('Mean Firing Rate (Hz)')
plt.subplot(514)
plt.plot(motion_directions, mean_spike_rates[3], 'ko', label='Cell 4')
plt.plot(motion_directions, von_Mises_cell4(motion_directions, popt4[0]), 'r-', label='von Mises Fit')
plt.legend()
plt.ylabel('Mean Firing Rate (Hz)')
plt.subplot(515)
plt.plot(motion_directions, mean_spike_rates[4], 'ko', label='Cell 5')
plt.plot(motion_directions, von_Mises_cell5(motion_directions, popt5[0]), 'r-', label='von Mises Fit')
plt.legend()
plt.ylabel('Mean Firing Rate (Hz)')
plt.xlabel('Motion Direction (degrees)')
plt.tight_layout()
#plt.savefig('von_mises_fits.png')

########################## COHERENCE DEPENDENCE ##############################

# create new data structure to store spiking/trial data
cell_data_coh = []
cell_spikes_coh = []
for i in range(0, len(coh_dependence)):
    curr_dict = coh_dependence[i]
    curr_ecodes = curr_dict['ecodes']
    curr_spikes = curr_dict['spikes'][0][0] # matlab struct is wrapped in arrays
    cell_data_coh.append(curr_ecodes[0][0][0][0][1])
    curr_spikes_arr = []
    for j in range(0, len(curr_spikes)):
        curr_spikes_arr.append(curr_spikes[j][0])
    cell_spikes_coh.append(curr_spikes_arr)

## 3) Compute and plot ROC curves for each cell

# compute firing rates and accuracy scores for each cell, each direction, and each coherence level
coh_vals = [0.0, 3.2, 6.4, 12.8, 25.6, 51.2, 99.9]
cell1_coh_dict_fr = {0:[[], [], []], 3.2:[[], [], []], 6.4:[[], [], []], 12.8:[[], [], []], 25.6:[[], [], []], 51.2:[[], [], []], 99.9:[[], [], []]}
cell2_coh_dict_fr = {0:[[], [], []], 3.2:[[], [], []], 6.4:[[], [], []], 12.8:[[], [], []], 25.6:[[], [], []], 51.2:[[], [], []], 99.9:[[], [], []]}
cell3_coh_dict_fr = {0:[[], [], []], 3.2:[[], [], []], 6.4:[[], [], []], 12.8:[[], [], []], 25.6:[[], [], []], 51.2:[[], [], []], 99.9:[[], [], []]}
cell4_coh_dict_fr = {0:[[], [], []], 3.2:[[], [], []], 6.4:[[], [], []], 12.8:[[], [], []], 25.6:[[], [], []], 51.2:[[], [], []], 99.9:[[], [], []]}
cell5_coh_dict_fr = {0:[[], [], []], 3.2:[[], [], []], 6.4:[[], [], []], 12.8:[[], [], []], 25.6:[[], [], []], 51.2:[[], [], []], 99.9:[[], [], []]}
all_cells_coh_dict_fr = [cell1_coh_dict_fr, cell2_coh_dict_fr, cell3_coh_dict_fr, cell4_coh_dict_fr, cell5_coh_dict_fr]

# number of trials for each cell
all_cells_num_trials = [len(cell_spikes_coh[0]), len(cell_spikes_coh[1]), len(cell_spikes_coh[2]), len(cell_spikes_coh[3]), len(cell_spikes_coh[4])]

# preferred and opposite directions for each cell
all_cells_directions = [[180, 0], [315, 135], [180, 0], [180, 0], [200, 20]]

# iterate through each cell
for i in range(0, num_cells):
    # get current cell's dictionary
    curr_dict = all_cells_coh_dict_fr[i]
    # get current cell's preferred and opposite direction
    curr_pref_dir = all_cells_directions[i][0]
    curr_opp_dir = all_cells_directions[i][1]
    # get current number of trials
    curr_num_trials = all_cells_num_trials[i]
    # iterate through each trial for each cell
    for j in range(0, curr_num_trials):
        # get data for current trial
        curr_spikes = cell_spikes_coh[i][j] # spike times
        curr_data = cell_data_coh[i][j] # overall data
        curr_coh = curr_data[3] # coherence value
        curr_score = curr_data[5] # score for this trial
        curr_dir = curr_data[2] # direction
        curr_onset = curr_data[0] # onset of stim - msec
        curr_offset = curr_data[1] # offset of stim - msec
        curr_interval = (curr_offset - curr_onset) / 1000 # convert to sec
        # counter for number of spikes
        curr_num_spikes = 0 
        # count total number of spikes within interval of stim
        for k in range(0, len(curr_spikes)):
            if curr_spikes[k] >= curr_onset and curr_spikes[k] <= curr_offset:
                curr_num_spikes += 1
        # find correct coherence level in dict
        for coh_val in curr_dict:
            if curr_coh == coh_val:
                # add firing rate to list for preferred direction
                if curr_dir == curr_pref_dir:
                    curr_list_fr_pref = curr_dict[coh_val][0]
                    curr_list_fr_pref.append(curr_num_spikes/curr_interval)
                # add firing rate to list for opposite direction
                elif curr_dir == curr_opp_dir:
                    curr_list_fr_opp = curr_dict[coh_val][1]
                    curr_list_fr_opp.append(curr_num_spikes/curr_interval)
                # add score to list for this coherence value
                curr_dict[coh_val][2].append(curr_score)
     
# plot histograms of firing rates for each coherence level of cell 1 - use these distributions for ROC computation
plt.figure(figsize=(8,14))
plt.subplot(711)
plt.hist(cell1_coh_dict_fr[0][0], color='r', label='Pref. Direction')
plt.hist(cell1_coh_dict_fr[0][1], color='b', label='Opp. Direction')
plt.ylabel('Number of Trials')
plt.title('Coherence = 0.0 %')
plt.legend()
plt.subplot(712)
plt.hist(cell1_coh_dict_fr[3.2][0], color='r', label='Pref. Direction')
plt.hist(cell1_coh_dict_fr[3.2][1], color='b', label='Opp. Direction')
plt.ylabel('Number of Trials')
plt.title('Coherence = 3.2 %')
plt.legend()
plt.subplot(713)
plt.hist(cell1_coh_dict_fr[6.4][0], color='r', label='Pref. Direction')
plt.hist(cell1_coh_dict_fr[6.4][1], color='b', label='Opp. Direction')
plt.ylabel('Number of Trials')
plt.title('Coherence = 6.4 %')
plt.legend()
plt.subplot(714)
plt.hist(cell1_coh_dict_fr[12.8][0], color='r', label='Pref. Direction')
plt.hist(cell1_coh_dict_fr[12.8][1], color='b', label='Opp. Direction')
plt.ylabel('Number of Trials')
plt.title('Coherence = 12.8 %')
plt.legend()
plt.subplot(715)
plt.hist(cell1_coh_dict_fr[25.6][0], color='r', label='Pref. Direction')
plt.hist(cell1_coh_dict_fr[25.6][1], color='b', label='Opp. Direction')
plt.ylabel('Number of Trials')
plt.title('Coherence = 6.4 %')
plt.legend()
plt.subplot(716)
plt.hist(cell1_coh_dict_fr[51.2][0], color='r', label='Pref. Direction')
plt.hist(cell1_coh_dict_fr[51.2][1], color='b', label='Opp. Direction')
plt.ylabel('Number of Trials')
plt.title('Coherence = 51.2 %')
plt.legend()
plt.subplot(717)
plt.hist(cell1_coh_dict_fr[99.9][0], color='r', label='Pref. Direction')
plt.hist(cell1_coh_dict_fr[99.9][1], color='b', label='Opp. Direction')
plt.xlabel('Firing Rate (Hz)')
plt.title('Coherence = 99.9 %')
plt.ylabel('Number of Trials')
plt.legend()
plt.tight_layout()
#plt.savefig('histograms_cell1.png')

# calculate false positive rates and true positive rates for roc curve 
# written by Dr. Balasubramanian for this project and converted from matlab to python code
# takes in x and y distributions (in this case firing rates at each coherence level)
def roc(x,y):
    N = 100 # use 100 data points for plot
    # reshape arrays
    m = np.shape(x)[0]
    x = np.reshape(x,[1,m])
    m = np.shape(y)[0]
    y = np.reshape(y,[1,m])
    # get minimum and maximum
    min_x, min_y = min(x[0]), min(y[0])
    max_x, max_y = max(x[0]), max(y[0])
    # calculate threshold values to evaluate false positive and true positive rates
    zlo = min(min_x, min_y)
    zhi = max(max_x, max_y)
    z = np.linspace(zlo,zhi, N)
    fpr, tpr = np.zeros(N), np.zeros(N)
    for i in range(0, N):
        fpr[N-i-1] = sum((y[:] > z[i])[0])
        tpr[N-i-1] = sum((x[:] > z[i])[0])
    [_,ny] = np.shape(y)
    [_,nx] = np.shape(x)
    # normalize
    fpr, tpr = fpr/ny, tpr/nx
    fpr[0], tpr[0] = 0, 0
    fpr[N-1], tpr[N-1] = 1, 1
    return fpr,tpr

# calculate fpr and tpr for each cell
all_roc = []
for i in range(0, num_cells):
    curr_dict = all_cells_coh_dict_fr[i]
    # temporary list to ultimately add to all_roc
    temp_roc_list = []
    for coh_val in curr_dict:
        # get x and y distributions of firing rates
        curr_x = curr_dict[coh_val][0]
        curr_y = curr_dict[coh_val][1]
        # ROC computation
        fpr,tpr = roc(curr_x, curr_y)
        temp_roc_list.append([fpr,tpr])
    all_roc.append(temp_roc_list)

# plot results for all cells
plt.figure(figsize=(8,22))
plt.subplot(511)
plt.plot(all_roc[0][0][0], all_roc[0][0][1], 'o-', label='Coherence Level = 0.0')
plt.plot(all_roc[0][1][0], all_roc[0][1][1], 'o-', label='Coherence Level = 3.2')
plt.plot(all_roc[0][2][0], all_roc[0][2][1], 'o-', label='Coherence Level = 6.4')
plt.plot(all_roc[0][3][0], all_roc[0][3][1], 'o-', label='Coherence Level = 12.8')
plt.plot(all_roc[0][4][0], all_roc[0][4][1], 'o-', label='Coherence Level = 25.6')
plt.plot(all_roc[0][5][0], all_roc[0][5][1], 'o-', label='Coherence Level = 51.2')
plt.plot(all_roc[0][6][0], all_roc[0][6][1], 'o-', label='Coherence Level = 99.9')
plt.plot([0.0, 1.0], [0.0, 1.0], 'k--')
plt.legend()
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Cell 1')
plt.subplot(512)
plt.plot(all_roc[1][0][0], all_roc[1][0][1], 'o-', label='Coherence Level = 0.0')
plt.plot(all_roc[1][1][0], all_roc[1][1][1], 'o-', label='Coherence Level = 3.2')
plt.plot(all_roc[1][2][0], all_roc[1][2][1], 'o-', label='Coherence Level = 6.4')
plt.plot(all_roc[1][3][0], all_roc[1][3][1], 'o-', label='Coherence Level = 12.8')
plt.plot(all_roc[1][4][0], all_roc[1][4][1], 'o-', label='Coherence Level = 25.6')
plt.plot(all_roc[1][5][0], all_roc[1][5][1], 'o-', label='Coherence Level = 51.2')
plt.plot(all_roc[1][6][0], all_roc[1][6][1], 'o-', label='Coherence Level = 99.9')
plt.plot([0.0, 1.0], [0.0, 1.0], 'k--')
plt.legend()
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Cell 2')
plt.subplot(513)
plt.plot(all_roc[2][0][0], all_roc[2][0][1], 'o-', label='Coherence Level = 0.0')
plt.plot(all_roc[2][1][0], all_roc[2][1][1], 'o-', label='Coherence Level = 3.2')
plt.plot(all_roc[2][2][0], all_roc[2][2][1], 'o-', label='Coherence Level = 6.4')
plt.plot(all_roc[2][3][0], all_roc[2][3][1], 'o-', label='Coherence Level = 12.8')
plt.plot(all_roc[2][4][0], all_roc[2][4][1], 'o-', label='Coherence Level = 25.6')
plt.plot(all_roc[2][5][0], all_roc[2][5][1], 'o-', label='Coherence Level = 51.2')
plt.plot(all_roc[2][6][0], all_roc[2][6][1], 'o-', label='Coherence Level = 99.9')
plt.plot([0.0, 1.0], [0.0, 1.0], 'k--')
plt.legend()
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Cell 3')
plt.subplot(514)
plt.plot(all_roc[3][0][0], all_roc[3][0][1], 'o-', label='Coherence Level = 0.0')
plt.plot(all_roc[3][1][0], all_roc[3][1][1], 'o-', label='Coherence Level = 3.2')
plt.plot(all_roc[3][2][0], all_roc[3][2][1], 'o-', label='Coherence Level = 6.4')
plt.plot(all_roc[3][3][0], all_roc[3][3][1], 'o-', label='Coherence Level = 12.8')
plt.plot(all_roc[3][4][0], all_roc[3][4][1], 'o-', label='Coherence Level = 25.6')
plt.plot(all_roc[3][5][0], all_roc[3][5][1], 'o-', label='Coherence Level = 51.2')
plt.plot(all_roc[3][6][0], all_roc[3][6][1], 'o-', label='Coherence Level = 99.9')
plt.plot([0.0, 1.0], [0.0, 1.0], 'k--')
plt.legend()
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Cell 4')
plt.subplot(515)
plt.plot(all_roc[4][0][0], all_roc[4][0][1], 'o-', label='Coherence Level = 0.0')
plt.plot(all_roc[4][1][0], all_roc[4][1][1], 'o-', label='Coherence Level = 3.2')
plt.plot(all_roc[4][2][0], all_roc[4][2][1], 'o-', label='Coherence Level = 6.4')
plt.plot(all_roc[4][3][0], all_roc[4][3][1], 'o-', label='Coherence Level = 12.8')
plt.plot(all_roc[4][4][0], all_roc[4][4][1], 'o-', label='Coherence Level = 25.6')
plt.plot(all_roc[4][5][0], all_roc[4][5][1], 'o-', label='Coherence Level = 51.2')
plt.plot(all_roc[4][6][0], all_roc[4][6][1], 'o-', label='Coherence Level = 99.9')
plt.plot([0.0, 1.0], [0.0, 1.0], 'k--')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Cell 5')
plt.tight_layout()
#plt.savefig('roc_all_cells.png')

# calculate area under curve for each cell
all_auc = []
for i in range(0, len(all_roc)):
    curr_cell = all_roc[i]
    # temporary list to store values of auc for each coherence level
    temp_list = []
    # iterate through coh levels
    for j in range(0, len(curr_cell)):
        # get fpr and tpr vals
        curr_fa_vals = curr_cell[j][0]
        curr_hit_vals = curr_cell[j][1]
        # calculate auc
        curr_auc = auc(curr_fa_vals, curr_hit_vals)
        temp_list.append(curr_auc)
    all_auc.append(temp_list)

## 4) fit the neurometric function (probability correct p computed as AUC versus coherence level c)
## with a cumulative Weibull function given by:
## p(c) = 1 - 0.5*exp(-(c/alpha)^beta) with free parameters alpha and beta

# divide coherence values by 100 to allow for easier convergence of fitting
coh_vals_norm = []
for i in range(0, len(coh_vals)):
    coh_vals_norm.append(coh_vals[i]/100)

def weibull(c, alpha, beta):
    return (1 - 0.5*np.exp(-(c/alpha)**beta))

# fit the data
popt1_coh, _ = curve_fit(weibull, coh_vals_norm, all_auc[0])
popt2_coh, _ = curve_fit(weibull, coh_vals_norm, all_auc[1])
popt3_coh, _ = curve_fit(weibull, coh_vals_norm, all_auc[2])
popt4_coh, _ = curve_fit(weibull, coh_vals_norm, all_auc[3])
popt5_coh, _ = curve_fit(weibull, coh_vals_norm, all_auc[4])

# plot neurometric fit results
plt.figure(figsize=(8,8))
ax = plt.gca()
ax.set_xscale('log')
plt.plot(coh_vals_norm, all_auc[0], 'ko', label='Cell 1')
plt.plot(coh_vals_norm, weibull(coh_vals_norm, popt1_coh[0], popt1_coh[1]), 'k')
plt.plot(coh_vals_norm, all_auc[1], 'ro', label='Cell 2')
plt.plot(coh_vals_norm, weibull(coh_vals_norm, popt2_coh[0], popt2_coh[1]), 'r')
plt.plot(coh_vals_norm, all_auc[2], 'bo', label='Cell 3')
plt.plot(coh_vals_norm, weibull(coh_vals_norm, popt3_coh[0], popt3_coh[1]), 'b')
plt.plot(coh_vals_norm, all_auc[3], 'go', label='Cell 4')
plt.plot(coh_vals_norm, weibull(coh_vals_norm, popt4_coh[0], popt4_coh[1]), 'g')
plt.plot(coh_vals_norm, all_auc[4], 'yo', label='Cell 5')
plt.plot(coh_vals_norm, weibull(coh_vals_norm, popt5_coh[0], popt5_coh[1]), 'y')
plt.xlabel('Coherence (Normalized to 1)')
plt.ylabel('Fraction Correct')
plt.legend()
plt.title('Neurometric Functions')
#plt.savefig('neurometric_fit.png')

## 5) fit the behavioral data (percent correct versus coherence) with Weibull functions

# calculate percent correct
all_behavioral_scores = []
for i in range(0, num_cells):
    curr_dict = all_cells_coh_dict_fr[i]
    # temporary list to store each cell's score in
    temp_list = []
    for coh_val in curr_dict:
            curr_score_list = curr_dict[coh_val][2]
            temp_list.append(np.sum(curr_score_list)/len(curr_score_list))
    all_behavioral_scores.append(temp_list)

# fit the data
popt1_coh_beh, _ = curve_fit(weibull, coh_vals_norm, all_behavioral_scores[0])
popt2_coh_beh, _ = curve_fit(weibull, coh_vals_norm, all_behavioral_scores[1])
popt3_coh_beh, _ = curve_fit(weibull, coh_vals_norm, all_behavioral_scores[2])
popt4_coh_beh, _ = curve_fit(weibull, coh_vals_norm, all_behavioral_scores[3])
popt5_coh_beh, _ = curve_fit(weibull, coh_vals_norm, all_behavioral_scores[4])

# plot behavioral fit results
plt.figure(figsize=(8,8))
ax = plt.gca()
ax.set_xscale('log')
plt.plot(coh_vals_norm, all_behavioral_scores[0], 'ko', label='Cell 1')
plt.plot(coh_vals_norm, weibull(coh_vals_norm, popt1_coh_beh[0], popt1_coh_beh[1]), 'k')
plt.plot(coh_vals_norm, all_behavioral_scores[1], 'ro', label='Cell 2')
plt.plot(coh_vals_norm, weibull(coh_vals_norm, popt2_coh_beh[0], popt2_coh_beh[1]), 'r')
plt.plot(coh_vals_norm, all_behavioral_scores[2], 'bo', label='Cell 3')
plt.plot(coh_vals_norm, weibull(coh_vals_norm, popt3_coh_beh[0], popt3_coh_beh[1]), 'b')
plt.plot(coh_vals_norm, all_behavioral_scores[3], 'go', label='Cell 4')
plt.plot(coh_vals_norm, weibull(coh_vals_norm, popt4_coh_beh[0], popt4_coh_beh[1]), 'g')
plt.plot(coh_vals_norm, all_behavioral_scores[4], 'yo', label='Cell 5')
plt.plot(coh_vals_norm, weibull(coh_vals_norm, popt5_coh_beh[0], popt5_coh_beh[1]), 'y')
plt.xlabel('Coherence (Normalized to 1)')
plt.ylabel('Fraction Correct')
plt.legend()
plt.title('Behavioral Functions')
#plt.savefig('behavioral_fit.png')