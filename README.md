# Metric-guided Image Reconstruction Bounds via Conformal Prediction

This repository contains source code for our [paper](https://arxiv.org/abs/2404.15274).

The application of CP to image reconstruction has been relatively limited.
This is a difficult problem because quantiles in higher dimensional data are M-quantiles, meaning they have infinite solutions and only have unique solutions when a direction is specified. How do we pick such a direction? The conventional (pixel-wise) method is to pick the direction where all pixels are independent. We argue that bounds should be computed in the direction of downstream metrics for more reliable downstream performance.

## Abstract

Recent advancements in machine learning have led to novel imaging systems and algorithms that address ill-posed problems. 
Assessing their trustworthiness and understanding how to deploy them safely at test time remains an important and open problem.
We propose a method that leverages conformal prediction to retrieve upper/lower bounds and statistical inliers/outliers of reconstructions based on the prediction intervals of downstream metrics.
We apply our method to sparse-view CT for downstream radiotherapy planning and show 1) that metric-guided bounds have valid coverage for downstream metrics while conventional pixel-wise bounds do not and 2) anatomical differences of upper/lower bounds between metric-guided and pixel-wise methods.
Our work paves the way for more meaningful reconstruction bounds.

## Example Implementation
```python
from mg import MetricGuidance
import numpy as np

alpha = 0.1 # miscoverage rate
n_calib, n_test = 10000, 20
n_recons = 50
n_metrics = 3
calib_est_dim = (n_calib, n_recons, n_metrics)
calib_gt_dim = (n_calib, n_metrics)
test_est_dim = (n_test, n_recons, n_metrics)
test_gt_dim = (n_test, n_metrics)

#### Toy example
# generate reconstructions corresponding to each test scene
recons_test = np.random.rand(n_test, n_recons, 256, 256)
# get downstream metrics corresponding to each reconstruction
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

#### Calibrate and test
cb = MetricGuidance(alpha=0.1)
Cc_ub, Cc_lb, calib_calibrated_coverage = cb.fit(calib)
Ct_lb, Ct_ub, test_calibrated_coverage = cb.predict(test)
ub_recons, lb_recons = cb.retrieve_bounds(recons_test)
ub_vals, lb_vals = cb.retrieve_bounds(test['est'])
ub_errs, lb_errs = cb.bound_errors(ub_vals, lb_vals)
print('Calibration Coverages: ', cb.coverages['cc'])
print('Test Coverages: ', cb.coverages['ct'])
print('Avg UB Retrieval Error: ', ub_errs.mean(0))
print('Avg LB Retrieval Error: ', lb_errs.mean(0))
```

## Running Example Code
```
python example.py

# Example Output:
# 	Calibration Coverages:  [0.8977 0.9013 0.9014]
# 	Test Coverages:  [1.  0.9 1. ]
# 	Avg UB Retrieval Error:  [-0.05486483 -0.00920602 -0.04949187]
# 	Avg LB Retrieval Error:  [0.03186384 0.02340578 0.00339853]
```

![Example Output](./example_output.png)

## Citation
```
@misc{cheung2024metricguided,
      title={Metric-guided Image Reconstruction Bounds via Conformal Prediction}, 
      author={Matt Y Cheung and Tucker J Netherton and Laurence E Court and Ashok Veeraraghavan and Guha Balakrishnan},
      year={2024},
      eprint={2404.15274},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```