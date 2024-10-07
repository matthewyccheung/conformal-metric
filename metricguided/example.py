from mg import MetricGuidedCalibration
import numpy as np
import matplotlib.pyplot as plt

alpha = 0.2
method = 'CQR'
n_calib, n_test = 10000, 20
n_recons = 50
n_metrics = 3
metric_names = ['Metric ' + str(i) for i in range(n_metrics)]
c_yhats_dim = (n_calib, n_recons, n_metrics)
c_ys_dim = (n_calib, n_metrics)
t_yhats_dim = (n_test, n_recons, n_metrics)
t_ys_dim = (n_test, n_metrics)

# generate reconstructions corresponding to each test scene
# toy example
recons_test = np.random.rand(n_test, n_recons, 256, 256)

# get downstream metrics corresponding to each reconstruction
# toy example
c_yhats = np.random.normal(0.1, 1.2, c_yhats_dim)
c_ys = np.random.normal(0.5, 1.6, c_ys_dim)
t_yhats = np.random.normal(0.3, 1.5, t_yhats_dim)
t_ys = np.random.normal(0.4, 1.2, t_ys_dim)

# calibrate and test
cb = MetricGuidedCalibration(alpha=alpha, method=method)
cc_lb, cc_ub, cc_coverages = cb.fit(c_ys, c_yhats)
ct_lb, ct_ub, ct_coverages = cb.validate(t_ys, t_yhats)
n_metrics_each_sample, n_sample_all_metrics, inlier_bool, outlier_bool = cb.retrieve_in_out(t_yhats)
lb_vals, ub_vals = cb.retrieve_bounds(t_yhats)
lengths = cb.interval_lengths(ct_lb, ct_ub)
retreived_lengths = cb.retrieval_lengths(lb_vals, ub_vals)
ub_errs, lb_errs = cb.bound_errors(lb_vals, ub_vals)
print('Calibration Coverages: ', cc_coverages)
print('Test Coverages: ', ct_coverages)
print('Avg UB Retrieval Error: ', ub_errs.mean(0))
print('Avg LB Retrieval Error: ', lb_errs.mean(0))

# scatter plot upper and lower bounds for each scene and metric
x_scatter = np.arange(t_ys.shape[0])
ub_scatter = ct_ub
lb_scatter = ct_lb
fig, ax = plt.subplots(1, ct_ub.shape[1])
for i in range(ct_ub.shape[1]):
	if ct_ub.shape[1] != 1:
		ax[i].scatter(x_scatter, ub_scatter[..., i], label='Calibrated Upper Bound')
		ax[i].scatter(x_scatter, lb_scatter[..., i], label='Calibrated Lower Bound')
		ax[i].scatter(x_scatter, t_ys[..., i], label='Ground Truth')
		ax[i].set_title(metric_names[i] + ' (Coverage = ' + str(np.round(cb.coverages['ct'][i], 3))+')')
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

if ct_ub.shape[1] != 1:
	ax[int(ct_ub.shape[1]/2)].legend(loc='center', bbox_to_anchor=(0, -0.625, 0.5, 1.0), ncol=3)
else:
	ax.legend()
plt.show()

