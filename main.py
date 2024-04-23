from mg import MetricGuidance
import numpy as np
import matplotlib.pyplot as plt

alpha = 0.1
n_calib, n_test = 10000, 20
n_recons = 50
n_metrics = 5
metric_names = ['Metric ' + str(i) for i in range(n_metrics)]
calib_est_dim = (n_calib, n_recons, n_metrics)
calib_gt_dim = (n_calib, n_metrics)
test_est_dim = (n_test, n_recons, n_metrics)
test_gt_dim = (n_test, n_metrics)

# generate reconstructions corresponding to each test scene
# toy example
recons_test = np.random.rand(n_test, n_recons, 256, 256)

# get downstream metrics corresponding to each reconstruction
# toy example
mu_calib_est, sigma_calib_est = 0.1, 1.2
mu_calib_gt, sigma_calib_gt = 0.5, 1.6
mu_test_est, sigma_test_est = 0.3, 1.5
mu_test_gt, sigma_test_gt = 0.4, 1.2
calib_est = np.random.normal(mu_calib_est, sigma_calib_est, calib_est_dim)
calib_gt = np.random.normal(mu_calib_gt, sigma_calib_gt, calib_gt_dim)
test_est = np.random.normal(mu_test_est, sigma_test_est, test_est_dim)
test_gt = np.random.normal(mu_test_gt, sigma_test_gt, test_gt_dim)

# store metrics in dict
calib = {'est': calib_est, 'gt': calib_gt}
test = {'est': test_est, 'gt': test_gt}

# calibrate and test
cb = MetricGuidance(alpha=0.1)
Cc_ub, Cc_lb, calib_calibrated_coverage = cb.fit(calib)
Ct_lb, Ct_ub, test_calibrated_coverage = cb.predict(test)
ub_recons, lb_recons = cb.retrieve_bounds(recons_test)
ub_vals, lb_vals = cb.retrieve_bounds(test['est'])
ub_errs, lb_errs = cb.bound_errors(ub_vals, lb_vals)
print('Calibration Coverages: ', cb.coverages['ct'])
print('Test Coverages: ', cb.coverages['ct'])
print('Avg UB Retrieval Error: ', ub_errs.mean(0))
print('Avg LB Retrieval Error: ', lb_errs.mean(0))

# scatter plot upper and lower bounds for each scene and metric
x_scatter = np.arange(test_gt.shape[0])
if ub_vals.shape[2] != 1:
	ub_scatter = np.diagonal(ub_vals, axis1=2)
	lb_scatter = np.diagonal(lb_vals, axis1=2)
else:
	ub_scatter = ub_vals
	lb_scatter = lb_vals
fig, ax = plt.subplots(1, ub_vals.shape[1])
for i in range(ub_vals.shape[1]):
	if ub_vals.shape[1] != 1:
		ax[i].scatter(x_scatter, ub_scatter[..., i], label='Calibrated Upper Bound')
		ax[i].scatter(x_scatter, lb_scatter[..., i], label='Calibrated Lower Bound')
		ax[i].scatter(x_scatter, test_gt[..., i], label='Ground Truth')
		ax[i].set_title(metric_names[i] + ' (Coverage = ' + str(np.round(cb.coverages['ct'][i], 2))+')')
		ax[i].set_xlabel('Scene Number')
		if i==0:
			ax[i].set_ylabel('Value')
	else:
		ax.scatter(x_scatter, ub_scatter[..., i], label='Calibrated Upper Bound')
		ax.scatter(x_scatter, lb_scatter[..., i], label='Calibrated Lower Bound')
		ax.scatter(x_scatter, test_gt[..., i], label='Ground Truth')
		ax.set_title(metric_names[i] + ' (Coverage = ' + str(np.round(cb.coverages['ct'][i], 2))+')')
		ax.set_xlabel('Scene Number')
		ax.set_ylabel('Value')			

if ub_vals.shape[1] != 1:
	ax[int(ub_vals.shape[1]/2)].legend(loc='center', bbox_to_anchor=(0, -0.625, 0.5, 1.0), ncol=3)
else:
	ax.legend()
plt.show()
