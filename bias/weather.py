import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
from tqdm import tqdm
from scipy.optimize import brentq
import matplotlib as mpl
import torch
from math import ceil
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib as mpl
from mg import *

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def get_weighted_quantile(scores,T,alpha):
    score_window = scores[T-K:T]
    return np.quantile(score_window, np.ceil((K+1)*(1-alpha))/K)

def loss_fn(cb, c_yhats, c_ys, t_yhats, t_ys):
    cc_lb, cc_ub, cc_coverages = cb.fit(c_ys, c_yhats)
    ct_lb, ct_ub, ct_coverages = cb.validate(t_ys, t_yhats)
    lengths = cb.interval_lengths(ct_lb, ct_ub)
    return lengths.max()

def get_bias(c_yhats, c_ys, t_yhats, t_ys, method):
    b = torch.tensor([(c_ys-c_yhats).mean()], requires_grad=True)
    opt = torch.optim.Adam([b], lr=1)
    cb = DiffL1(alpha, method)
    # minimize
    loss_prev, loss, iteration = 1e5, 1e4, 0
    while abs(loss_prev-loss)>1e-1:
        loss_prev = loss
        opt.zero_grad()
        loss = loss_fn(cb, c_yhats+b, c_ys, t_yhats+b, t_ys)
        loss.backward(retain_graph=True)
        opt.step()
        iteration +=1
        # print(iteration, b.item(), loss.item())
    # b = b.item()
    return b

def calibrate(c_yhats, c_ys, t_yhats, t_ys, method, T, K, alpha, debias):
    c_yhats = c_yhats[T-K:T]
    c_ys = c_ys[T-K:T]
    t_yhats = torch.Tensor([t_yhats[T]])
    t_ys = torch.Tensor([t_ys[T]])
    if debias:
        b = get_bias(c_yhats, c_ys, t_yhats, t_ys, method)
        c_yhats += b
        t_yhats += b
    else:
        b=torch.Tensor([0])
    cb = DiffL1(alpha=alpha, method=method)
    cc_lb, cc_ub, cc_coverage = cb.fit(c_ys, c_yhats)
    ct_lb, ct_ub, ct_coverage = cb.validate(t_ys, t_yhats)
    length = cb.interval_lengths(ct_lb, ct_ub)
    return ct_lb, ct_ub, ct_coverage, length, b

# Load cached data from Shifts (Yandex Weather Prediction). See https://github.com/Shifts-Project/shifts for details.
# The data has a change point halfway through.
if not os.path.exists('../data'):
    os.system('gdown 1h7S6N_Rx7gdfO3ZunzErZy6H7620EbZK -O ../data.tar.gz')
    os.system('tar -xf ../data.tar.gz -C ../')
    os.system('rm ../data.tar.gz')

data = np.load('../data/weather/weather-catboost.npz')
preds = data['preds']
temperatures = data['temperatures'] # Temperature (degrees Celsius)
times = data['times'] # Times
                    
pred_mean = preds[:,:,0].mean(axis=0)
# pred_uncertainty = np.sqrt(preds[:,:,1].mean(axis=0))

# total_time = len(pred_mean)
total_time = 100000
sort_idx = np.argsort(times)
pred_mean = pred_mean[sort_idx]
temperatures = temperatures[sort_idx]
times = times[sort_idx]

# adjust bias
temperatures0 = temperatures.copy()
bias = np.zeros(temperatures.shape)
start = 0
bias[start:] += 20*np.arange(len(pred_mean)-start)/(len(pred_mean)-start)
pred_mean -= bias

alpha = 0.1
K=1000
c_yhats = torch.Tensor(pred_mean)
c_ys = torch.Tensor(temperatures)
c_ys0 = torch.Tensor(temperatures0)
t_yhats = torch.Tensor(pred_mean)
t_ys = torch.Tensor(temperatures)
t_ys0 = torch.Tensor(temperatures0)

# naive
scores = np.abs(temperatures-pred_mean)
naive_qhats = np.array([np.quantile(scores[:t], np.ceil((t+1)*(1-alpha))/t, interpolation='higher') for t in range(K+1, scores.shape[0])])
naive_lb, naive_ub = pred_mean[K+1:]-naive_qhats, pred_mean[K+1:]+naive_qhats
# Calculate coverage over time
naive_coverage = (temperatures[K+1:] >= naive_lb) & (temperatures[K+1:] <= naive_ub)

# No bias sym
ct_lbs_sym0 = np.zeros(len(c_yhats))
ct_ubs_sym0 = np.zeros(len(c_yhats))
ct_coverages_sym0 = np.zeros(len(c_yhats))
lengths_sym0 = np.zeros(len(c_yhats))
for T in range(K+1, len(c_yhats)):
    ct_lb, ct_ub, ct_coverage, length, _ = calibrate(c_yhats, c_ys0, t_yhats, t_ys0, 'sym', T, K, alpha, False)
    ct_lbs_sym0[T] = ct_lb.item()
    ct_ubs_sym0[T] = ct_ub.item()
    ct_coverages_sym0[T] = ct_coverage.item()
    lengths_sym0[T] = length.item()
    
# Bias Sym
ct_lbs_sym = np.zeros(len(c_yhats))
ct_ubs_sym = np.zeros(len(c_yhats))
ct_coverages_sym = np.zeros(len(c_yhats))
lengths_sym = np.zeros(len(c_yhats))
for T in range(K+1, len(c_yhats)):
    ct_lb, ct_ub, ct_coverage, length, _ = calibrate(c_yhats, c_ys, t_yhats, t_ys, 'sym', T, K, alpha, False)
    ct_lbs_sym[T] = ct_lb.item()
    ct_ubs_sym[T] = ct_ub.item()
    ct_coverages_sym[T] = ct_coverage.item()
    lengths_sym[T] = length.item()
    
# Bias Asym
ct_lbs_asym = np.zeros(len(c_yhats))
ct_ubs_asym = np.zeros(len(c_yhats))
ct_coverages_asym = np.zeros(len(c_yhats))
lengths_asym = np.zeros(len(c_yhats))
for T in range(K+1, len(c_yhats)):
    ct_lb, ct_ub, ct_coverage, length, _ = calibrate(c_yhats, c_ys, t_yhats, t_ys, 'asym', T, K, alpha, False)
    ct_lbs_asym[T] = ct_lb.item()
    ct_ubs_asym[T] = ct_ub.item()
    ct_coverages_asym[T] = ct_coverage.item()
    lengths_asym[T] = length.item()

bs = moving_average(temperatures-pred_mean, K)[:-2]
# bs = (temperatures-pred_mean)[K+1:]
Lsym0 = lengths_sym0[K+1:]
Lsym = lengths_sym[K+1:]
Lasym = lengths_asym[K+1:]

# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['mathtext.fontset'] = 'stix'
# mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 10


fig, ax = plt.subplots(2, 2, figsize=(9,6))
sort_idxs = np.argsort(bs)

x = np.arange(total_time)

ax[0,0].plot(temperatures, label=r'$Ground\;Truth$')
ax[0,0].plot(pred_mean, label=r'$Biased\;Predictions$')
ax[0,0].set_xlabel(r'$Timestamp$')
ax[0,0].set_ylabel(r'$Temperatures\; (^\circ C)$')
ax[0,0].legend()
ax[0,0].set_xlim(0, total_time)
# ax[0,0].set_xlim(total_time-100, total_time)
# ax[0,1].set_ylim(-10, 60)

ax[1,0].fill_between(x, ct_lbs_sym, ct_ubs_sym, label=r'$Symmetric\;Intervals$', color='r', alpha=0.4)
ax[1,0].fill_between(x, ct_lbs_asym, ct_ubs_asym, label=r'$Asymmetric\;Intervals$', color='b', alpha=0.4)
ax[1,0].set_ylabel(r'$Temperature\;(^\circ C)$')
ax[1,0].set_xlabel(r'$Timestamp$')
ax[1,0].legend()
ax[1,0].set_xlim(0, total_time)
# ax[1,0].set_xlim(total_time-10000, total_time)
# ax[1,0].set_ylim(-10, 60)

ax[0,1].plot(moving_average(ct_coverages_asym[K+1:], K), label=r'$Asymmetric\;(weighted)$', c='b')
ax[0,1].plot(moving_average(ct_coverages_sym[K+1:], K), label=r'$Symmetric\;(weighted)$', c='r')
ax[0,1].plot(moving_average(naive_coverage[K+1:], K), label=r'$Symmetric\;(unweighted)$', c='g')
ax[0,1].set_ylim([0, 1])
# ax[0,1].set_xlim(0, total_time)
ax[0,1].set_ylabel(r'$Coverage$')
ax[0,1].set_xlabel(r'$Timestamp$')
ax[0,1].legend(loc='lower left')

# ax[1,1].set_ylim([0,50])
# ax[1,1].set_xlim([-10,50])
ax[1,1].set_xlabel(r'$Bias$')
ax[1,1].set_ylabel(r'$Length$')
ax[1,1].scatter(bs, Lasym, s=10, label=r'$Asymmetric$', c='b')
ax[1,1].scatter(bs, Lsym, s=10, label=r'$Symmetric$', c='r')
# plot Lasym ≈ Lsym (zero bias) for less computation to see if Cor 3.1 is true
# results are valid because Lasym ≥ Lsym
ax[1,1].plot(bs[sort_idxs], Lasym[sort_idxs]+2*abs(bs[sort_idxs]), c='m', label=r'$Upper\;bound\;(Thm.\;2)$')
ax[1,1].legend()

plt.tight_layout()
plt.savefig('output.png', dpi=1000)
