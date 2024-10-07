from scipy import stats
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib as mpl
import sys
sys.path.append('../')
from mg import *

def loss_fn(bias, c_yhats, c_ys, t_yhats, t_ys):
	cc_lb, cc_ub, cc_coverages = cb.fit(c_ys, c_yhats+bias)
	ct_lb, ct_ub, ct_coverages = cb.validate(t_ys, t_yhats+bias)
	lengths = cb.interval_lengths(ct_lb, ct_ub)
	return lengths.max()

alphas = np.round([0.05],2)
data_dists = [stats.norm(loc=10, scale=5)]
dd_labels = [r'$N(10, 5)$']
noise_dists = [ stats.weibull_min(1, loc=0, scale=5),
				stats.norm(loc=0, scale=2),
				stats.weibull_min(1, loc=-2, scale=5)]
nd_labels = [r'$W(1,0,5)$', r'$N(0,2)$', r'$W(1,-2,5)$']

n_c, n_t = 1000, 1000
n_s, n_m = 1000, 1
metric_names = ['Metric ' + str(i) for i in range(n_m)]
c_yhat_dim = (n_c, n_s, n_m)
c_y_dim = (n_c, n_m)
t_yhat_dim = (n_t, n_s, n_m)
t_y_dim = (n_t, n_m)

alpha = 0.1
methods = ['CQR', 'asymCQR']

res = 0.1
lb, ub = -2, 2
biases = np.arange(lb, ub+res, res)
eff_biases = np.zeros((len(noise_dists), len(data_dists), len(methods)))
sample_means = np.zeros((len(noise_dists), len(data_dists)))
data = np.zeros((len(noise_dists), len(data_dists), len(biases), len(methods)))
mpl.rcParams['font.size'] = 18
fig1, ax1 = plt.subplots(len(data_dists), len(noise_dists))
for ni,noise_dist in enumerate(noise_dists):
	print('Noise Dist = ', noise_dist)
	if len(noise_dists)==1:
		pax1 = ax1[di]
	elif len(data_dists)==1:
		pax1 = ax1[ni]
	else:
		pax1 = ax1[di,ni]
	for di,data_dist in enumerate(data_dists):
		c_noise = torch.tensor(noise_dist.rvs(c_yhat_dim))
		t_noise = torch.tensor(noise_dist.rvs(t_yhat_dim))
		if ni==2:
			c_noise = -c_noise
			t_noise = -t_noise
		c_ys = torch.tensor(data_dist.rvs(c_y_dim))
		c_yhats = c_ys.unsqueeze(1)+2*c_noise
		t_ys = torch.tensor(data_dist.rvs(t_y_dim))
		t_yhats = t_ys.unsqueeze(1)+2*t_noise

		sample_means[ni,di] = (c_ys-c_yhats.mean(1)).mean()

		pax1.hist(t_ys.flatten(), label=r'$Y$', alpha=0.4, color='b', bins=20,density=True)
		pax1.hist(t_yhats.flatten(), label=r'$\hat Y^b$', alpha=0.4, color='r', bins=20,density=True)
		pax1.set_title(dd_labels[di]+'+'+nd_labels[ni])
		if ni==0:
			pax1.set_ylabel(r'Density')
		if di==len(data_dists)-1:
			pax1.set_xlabel(r'Value')
		for mi,method in enumerate(methods):
			b = torch.tensor([sample_means[ni,di]], requires_grad=True)
			opt = torch.optim.AdamW([b], lr=1e-2)
			cb = DiffCQR(alpha=alpha, method=method)
			# minimize
			loss_prev, loss, iteration = 1e5, 1e4, 0
			while abs(loss_prev-loss)>1e-5:
				loss_prev = loss
				opt.zero_grad()
				loss = loss_fn(b, c_yhats, c_ys, t_yhats, t_ys)
				loss.backward()
				opt.step()
				iteration +=1
				print(iteration, b.item(), loss.item())

			b = b.item()
			eff_biases[ni, di, mi] = b

			for bi,bias in enumerate(biases):
				print('Effective Bias = ', bias)
				cb = MetricGuidedCalibration(alpha=alpha, method=method)
				cc_lb, cc_ub, cc_coverages = cb.fit(c_ys.numpy(), c_yhats.numpy()+bias+b)
				ct_lb, ct_ub, ct_coverages = cb.validate(t_ys.numpy(), t_yhats.numpy()+bias+b)
				lengths = cb.interval_lengths(ct_lb, ct_ub)
				data[ni, di, bi, mi] = lengths.max()
		
		pax1.hist(t_yhats.flatten()+b, label=r'$\hat Y^b-b_{eff}$', alpha=0.4, color='r', bins=20, density=True, linestyle=(0,(2,2)),fill=False, ec='k')

fig, ax = plt.subplots(data.shape[1], data.shape[0])
for ni in range(data.shape[0]):
	for di in range(data.shape[1]):
		if data.shape[0]==1:
			pax = ax[di]
		elif data.shape[1]==1:
			pax = ax[ni]
		else:
			pax = ax[di, ni]
		
		pax.plot(biases, data[ni,di,:,1], label='Asymmetric', linewidth=3, c='green')
		pax.plot(biases, data[ni,di,:,0], label='Symmetric', linewidth=3, c='red')
		pax.plot(biases, data[ni,di,len(biases)//2,0]+2*abs(biases), label='Upper Bound (Thm. 2)', c='k', linestyle='dashed', linewidth=3)
		Ldiff = data[ni,di,len(biases)//2,1]-data[ni,di,len(biases)//2,0]
		if Ldiff>0:
			pax.axvline(-2*Ldiff,c='dimgrey', linestyle='dashed', label='Condition (Cor. 3.1)',linewidth=3)
			pax.axvline(2*Ldiff, c='dimgrey', linestyle='dashed', linewidth=3)
			if data.shape[0]==1:
				legend_idx = di
			elif data.shape[1]==1:
				legend_idx = ni
			else:
				legend_idx = [di, ni]
		if di==data.shape[1]-1:
			pax.set_xlabel('Bias')
		if ni==0:
			pax.set_ylabel('Max Interval Length')
		pax.set_title(dd_labels[di]+'+'+nd_labels[ni])
		pax.set_xlim([lb,ub])
		pax.grid()

handles, labels = ax[legend_idx].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center',  ncols=6)
fig.tight_layout()
fig.subplots_adjust(hspace=0.35, wspace=0.35)
handles, labels = pax1.get_legend_handles_labels()
fig1.legend(handles, labels, loc='lower center', ncols=3)
fig1.tight_layout()
fig1.subplots_adjust(hspace=0.25, wspace=0.25)
plt.show()
