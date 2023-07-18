import sys

import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D

import itertools

import pickle
import emcee

import scipy.io
import scipy as sp
from scipy.optimize import curve_fit, leastsq, brute, basinhopping, minimize
from scipy.stats import norm
from scipy.signal import medfilt2d

from iminuit import Minuit
from iminuit import minimize as minuitimize

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

import math
import seaborn as sns

import target_io

import analysis_quicklook as aq
import make_clean_showers as mcs
import find_crab as fc
from apply_gains import apply_gains
import psct_reader
import utils

saving_dict  = True
saving_param_df  = True

ana_type = "burn" # "c2c" "burn" "blind"
special_ID = "_minuit_redchi2"#"_chi2_min"

use_chi2_min = True
use_minuit   = True

# "island" fits only the non-nan pixels contained in the island produced by cleaning
# "full_unclean" fits all non-nan pixels in the uncleaned imaged
##### images are pretty noisy, not sure if this will have much success
# "full_clean" fits all non-nan pixels (including zero-ed out ones) in cleaned image
##### not scientifically motivated, but doesn't seem to cause too many issues
fit_coverage = "island"  # "island" "full_unclean" "full_clean"

ana_dir			 = '/data/user/cadams/psct/2d_gaussian'
fit_data_dir	 = f'{ana_dir}/saved_data'
image_dir		 = f'{ana_dir}/images'
DATADIR='/mnt/lfs7/wipac/CTA/target5and7data/runs_320000_through_329999'
TRACKDIR = f"{DATADIR}/positioner_logs"
BRENTDIR='/home/bmode/analysis'
HILLAS_DATADIR = f"/data/user/cadams/hillas_data/{ana_type}"	  #"/data/user/bmode/crab_data/v0.2"

fit_data_dict_fn = f'{fit_data_dir}/{ana_type}{special_ID}_{fit_coverage}_fit_data_dict.p'
fit_params_df_fn = f'{fit_data_dir}/{ana_type}{special_ID}_{fit_coverage}_fit_params_df.csv' #f'{fit_data_dir}/{ana_type}_fit_params_df.csv' #f'{fit_data_dir}/linear_{ana_type}_fit_params_df.csv'

pedvar_fn = f'{fit_data_dir}/burn_pedvars.p' #{ana_type}_pedvars.p'

CRAB = "05 34 31.94 +22 00 52.2"
source = SkyCoord(CRAB, unit=(u.hourangle, u.deg), frame="icrs")

mod_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 103, 106, 107,
			108, 111, 112, 114, 115, 119, 121, 123, 124, 125, 126]

mod_nums = [100, 111, 114, 107, 6, 115, 123, 124, 112, 7, 119,
			108, 110, 121, 8, 103, 125, 126, 106, 9, 4, 5, 1, 3, 2]
fpm_nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
			17, 18, 19, 20, 21, 22, 23, 24]

fpm_pos = np.mgrid[0:5, 0:5]
fpm_pos = zip(fpm_pos[0].flatten(), fpm_pos[1].flatten())

mod_to_fpm = dict(zip(mod_nums, fpm_nums))
fpm_to_pos = dict(zip(fpm_nums, fpm_pos))

ch_nums = np.array([[21, 20, 17, 16, 5, 4, 1, 0],
					[23, 22, 19, 18, 7, 6, 3, 2],
					[29, 28, 25, 24, 13, 12, 9, 8],
					[31, 30, 27, 26, 15, 14, 11, 10],
					[53, 52, 49, 48, 37, 36, 33, 32],
					[55, 54, 51, 50, 39, 38, 35, 34],
					[61, 60, 57, 56, 45, 44, 41, 40],
					[63, 62, 59, 58, 47, 46, 43, 42]])
rot_ch_nums = np.rot90(ch_nums, k=2)
ch_to_pos = dict(zip(ch_nums.reshape(-1), np.arange(64)))
rot_ch_to_pos = dict(zip(rot_ch_nums.reshape(-1), np.arange(64)))

num_columns = 5
total_cells = num_columns * num_columns * 64
indices = np.arange(total_cells).reshape(-1, int(np.sqrt(total_cells)))
grid_ind = List()
for index, mod in enumerate(mod_list):
	i, j = fpm_to_pos[mod_to_fpm[mod]]
	ch_map = dict()
	if j % 2 == 0:
		ch_map = rot_ch_to_pos
	else:
		ch_map = ch_to_pos
	# print(f"Channel Map: {ch_map}")
	j = num_columns - 1 - j
	pix_ind = np.array(indices[
		(8*i):8*(i+1), (8*j):8*(j+1)]).reshape(-1)
	# print(f"Pixel Index: {pix_ind}")
	for asic in range(4):
		for ch in range(16):
			grid_ind.append(int(pix_ind[ch_map[asic * 16 + ch]]))

# start by loading in .csv data with hillas parameters
# set param true if first column of csv is the run number, false if not (most likely going to be false)
run_num_included = False

if ana_type == "c2c":
	runs = [328629, 328630, 328631]
elif ana_type == "burn":
	runs = [328555, 328564, 328565, 328585, 328608, 328617, 328642,
				328719, 328748, 328761, 328854]
elif ana_type == "blind":
	runs = []

all_fns = [f'{HILLAS_DATADIR}/run{runID}_elliptical_fit_study.csv' for runID in runs]

df_list = [pd.read_csv(f, skipinitialspace=True) for f in all_fns]

if not run_num_included:
	for i, run_df in enumerate(df_list):
		run_df.insert(0,"run",runs[i])

data = pd.concat(df_list, ignore_index=True)
data = data[np.logical_not(np.logical_and(data["run"]==328631, data["ev"]>=122804))]

def twoD_Gaussian_VEGAS(xy, amplitude, xo, yo, sigma_l, sigma_w, theta):
	x, y = xy
	xo = float(xo)
	yo = float(yo)
	g = amplitude*np.exp( (-1/2)*( (((x-xo)*np.cos(-theta)-(y-yo)*np.sin(-theta)) / sigma_l)**2 
								  + (((x-xo)*np.sin(-theta)+(y-yo)*np.cos(-theta)) / sigma_w)**2 ) )
	return g.ravel()

def chisqfunc(test_vars, xy, data, pedvar):
	model = twoD_Gaussian_VEGAS(xy, *test_vars)
#	  in VHFit, it appears the denominator is
#		  (pedvar^2 + (fitCharge-offset)/gammaFactor)
#		  electronic noise + poissonian statistical uncertainty?
#		  gammaFactor = photoelectrons/count.  It's a calibration factor. (0.2 here)
	chisq = np.sum((data - model)**2/(np.sqrt(np.abs(data))**2+pedvar**2))
	pixel_cnt = len(data)
	if fit_coverage == "island":
		if pixel_cnt > 0:
			chisq /= pixel_cnt
	else:
		if pixel_cnt > (6 - 1):
			chisq /= pixel_cnt - (6 - 1)
	return chisq

# generate image storing "pedvar" per pixel
with open(pedvar_fn,'rb') as f:
	images4pedvar = pickle.load(f)
pedvar_image = np.nanstd(images4pedvar, axis=0)


all_unclean_images	   = []
all_clean_images	   = []
converged_clean_images = []		 # will store the cleaned images of events whose fits converged

x_ax_list = []
y_ax_list = []
fit_images = []
fit_params = []
run_evt_pass_list = []
rsq_list       = []
min_chi2_list  = []

for runID in tqdm(runs, leave=True):
	reader = psct_reader.WaveformArrayReader(f"{DATADIR}/cal{runID}.r1", 
											 tracking_file=f"{TRACKDIR}/positionerLog_crab2020.txt",
											 source=source)
	reader.noise_thresh = 2.0
	reader.clean_thresh = 2.0
	reader.nan			= True
	
	culmination_time = reader.culmination_time
	
	for ev in tqdm(data["ev"][data["run"]==runID], leave=True):
		ev = int(ev)
		reader.get_event(ev)
		cpu_s = reader.cpu_s

		charges = reader.charges
		image = reader.image
		all_unclean_images.append(image)
		clean_image = reader.clean_image
		all_clean_images.append(clean_image)
		
		delta_x = utils.get_delta_x(cpu_s, culmination_time)
		delta_y = utils.get_delta_y(cpu_s, culmination_time)
		
		height_guess = np.nanmax(clean_image)
		x_guess		 = data["x"][np.logical_and(data["run"]==runID, data["ev"]==ev)].iloc[0]
		y_guess		 = data["y"][np.logical_and(data["run"]==runID, data["ev"]==ev)].iloc[0]
		length_guess = data["length"][np.logical_and(data["run"]==runID, data["ev"]==ev)].iloc[0]
		width_guess  = data["width"][np.logical_and(data["run"]==runID, data["ev"]==ev)].iloc[0]
		psi_guess	 = data["psi"][np.logical_and(data["run"]==runID, data["ev"]==ev)].iloc[0]
		
		x_test = np.asarray([i - 19.5 + 0.31 * ((i // 8) - 2) + delta_x for i in range(40)]).T
		y_test = np.asarray([j - 19.5 + 0.31 * ((j // 8) - 2) + delta_y for j in range(40)]).T
		
		X_test,Y_test = np.meshgrid(x_test,y_test)
		
		p_guess = (height_guess, x_guess, y_guess, length_guess, width_guess, psi_guess)
		
		if fit_coverage == "island":
			mask = np.logical_and(clean_image!=0, ~np.isnan(clean_image))
			image2fit = np.copy(clean_image)
		elif fit_coverage == "full_unclean": 
			mask = ~np.isnan(clean_image) # can't fit nans, so mask them out
			image2fit = np.copy(image)
		elif fit_coverage == "full_clean":
			mask = ~np.isnan(clean_image)
			image2fit = np.copy(clean_image)
		else:
			sys.exit('fit_coverage string not set correctly')
		try:
			if use_chi2_min:
				chi2_bounds = ((0,None),(None,None),(None,None),(0,None),(0,None),(-6*np.pi,6*np.pi))
				if use_minuit:
					m=Minuit(lambda x: chisqfunc(x, *((X_test[mask], Y_test[mask]), image2fit[mask], pedvar_image[mask])),p_guess, 
					         name=("A", "x", "y", "σ_l", "σ_w", "ψ"))
					m.limits = chi2_bounds
					m.migrad()
					popt = m.values
					min_chi2 = m.fval
				else:
					out = minimize(chisqfunc, p_guess, args=((X_test[mask], Y_test[mask]), image2fit[mask], pedvar_image[mask]), bounds=chi2_bounds)
					popt = out['x']
					min_chi2 = out['fun']
				image_fitted = twoD_Gaussian_VEGAS((X_test, Y_test), *popt)
			else:
				guess_min = (p_guess[0]-30, p_guess[1]-10, p_guess[2]-10, 0, 0, -6*np.pi) # was +/- 3.5 on the x,y pos
				guess_max = (p_guess[0]+30, p_guess[1]+10, p_guess[2]+10, 40, 40, 6*np.pi)
				
				popt, pcov = curve_fit(twoD_Gaussian_VEGAS, (X_test[mask], Y_test[mask]), image2fit[mask], p0=p_guess, bounds=(guess_min, guess_max),maxfev=10000)
				image_fitted = twoD_Gaussian_VEGAS((X_test, Y_test), *popt)

			if popt[4] > popt[3]:
				temp = popt[3]
				popt[3] = popt[4]
				popt[4] = temp
				popt[5] = popt[5] - np.pi/2
				
			residuals = image2fit[mask] - twoD_Gaussian_VEGAS((X_test[mask], Y_test[mask]), *popt) # calc residuals of fit (for R^2)
			
			ss_res = np.sum(residuals**2)									   # residual sum of squares
			ss_tot = np.sum((image2fit[mask]-np.mean(image2fit[mask]))**2) # total sum of squares

			r_sq = 1 - (ss_res / ss_tot)									   # then calc R^2

			image_fitted = image_fitted.reshape((40,40))
			
			fit_images.append(image_fitted)
			fit_params.append(np.array(popt))
			run_evt_pass_list.append((runID,ev))
			x_ax_list.append(x_test)
			y_ax_list.append(y_test)
			converged_clean_images.append(clean_image)
			min_chi2_list.append(min_chi2 if use_chi2_min else np.nan)
			rsq_list.append(r_sq)
			
		except Exception as e:
			print(f"run {runID} evt {ev} failed to converge")
			print(e)
#			  print(p_guess)
			continue
			break
		
# list of indices in all_clean_images that converged
converged_inds = np.zeros(len(run_evt_pass_list),dtype=int)
for i in range(len(run_evt_pass_list)):
	runID = run_evt_pass_list[i][0]
	ev	  = run_evt_pass_list[i][1]
	# match runID and event, flag that index
	flag_evt = np.logical_and(data["run"]==runID, data["ev"]==ev)
	index = np.flatnonzero(flag_evt)[0]
	converged_inds[i] = index
	
unconverged_inds = [x for x in range(0, len(data)) if x not in converged_inds]

fit_params = np.array(fit_params,dtype=float)
fit_params_out = np.copy(fit_params)

# from https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
# analytic integral of 2d gauss with bounds at infinity
# fit_sizes = np.array([2*np.pi*fit_params[:,0]*fit_params[:,3]*fit_params[:,4]]).T OLD VERSION IGNORE
fit_sizes = 2*np.pi*fit_params[:,0]*fit_params[:,3]*fit_params[:,4]
fit_dists = np.sqrt(fit_params[:,1]**2+fit_params[:,2]**2)

psis	= np.rad2deg(np.fmod(fit_params[:,5],np.pi)) # used to be 2*np.pi
psis[psis<-90] += 180
psis[psis>90]  -= 180
phis	= np.rad2deg(np.arctan2(fit_params[:,2], fit_params[:,1]))
phis[phis<-90] += 180
phis[phis>90]  -= 180

alphas	= np.abs(phis - psis)
alphas = np.abs((alphas + 180) % 360 - 180)
alphas[np.logical_and(alphas > 90, alphas < 180)] = 180 - alphas[np.logical_and(alphas > 90, alphas < 180)]

fit_misses = fit_dists*np.sin(np.deg2rad(alphas))

addtl_fit_params = np.column_stack((fit_sizes, fit_dists, fit_misses, psis, phis, alphas, rsq_list, min_chi2_list))
new_fit_params = np.append(fit_params,addtl_fit_params,axis=1)


partial_fit_params_df = pd.DataFrame(new_fit_params, columns=['height', 'x','y','length','width','psi_rad','size','dis','miss','psi','phi','alpha', 'rsq', 'chi2_min'])
run_evt_df = pd.DataFrame(run_evt_pass_list,columns=['run','ev'])
hillas_select_df = pd.merge(data,run_evt_df,how='inner',on=['run','ev'])[['width', 'length', 'miss',
	   'dis', 'azwidth', 'alpha', 'psi', 'size','is_ON','is_OFF']]
hillas_select_df.rename(columns = {'width':'hillas_width', 'length':'hillas_length', 
								   'miss':'hillas_miss', 'dis':'hillas_dis',
								   'azwidth':'hillas_azwidth', 'alpha':'hillas_alpha',
								   'psi':'hillas_psi', 'size':'hillas_size'}, inplace = True)
# on_off_df = pd.merge(data,run_evt_df,how='inner',on=['run','ev'])[['is_ON','is_OFF']]
fit_params_df = pd.concat([run_evt_df, hillas_select_df, partial_fit_params_df],axis=1)

if saving_dict:
	fit_data_dict = {
						"new_fit_params"		 : new_fit_params,
						"run_evt_pass_list"		 : run_evt_pass_list,
						"all_unclean_images"	 : all_unclean_images,
						"all_clean_images"		 : all_clean_images,
						"fit_images"			 : fit_images,
						"converged_clean_images" : converged_clean_images,
						"converged_inds"		 : converged_inds,
						"x_ax_list"				 : x_ax_list,
						"y_ax_list"				 : y_ax_list,
						"rsq_list"				 : rsq_list,
						"min_chi2_list"			 : min_chi2_list
					}
	with open(fit_data_dict_fn,'wb') as f:
		pickle.dump(fit_data_dict,f)

if saving_param_df:
	fit_params_df.to_csv(fit_params_df_fn,index=False)
