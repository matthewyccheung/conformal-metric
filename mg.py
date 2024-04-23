import numpy as np
import matplotlib.pyplot as plt
import pdb
import pandas as pd

class MetricGuidance:

	"""
		Implements a discrete version of Conformalized Quantile Regression (CQR) [1]
		using sample quantiles instead of regression quantiles. Allows calibration
		for multiple metrics simultaneously. See our paper for more information.

		Usage:
		----------
		cb = MetricGuidance(alpha=0.1)
		cb.fit(calib)
		cb.predict(test)
		print(cb.coverages)	

		Notes:
		----------
		- Adjusted level of confidence: alpha_adj = np.ceil((n_calib+1)*(1-alpha))/n_calib 
			should not be > 1

		References:
		---------
		[1] Romano, Yaniv, Evan Patterson, and Emmanuel Candes. 
			"Conformalized quantile regression." 
			Advances in neural information processing systems 32 (2019).
	"""
	
	def __init__(self, alpha):
		
		"""
			Parameters
			----------
			- alpha: level of confidence in range [0, 1]
				- type: float
		"""
		
		self.alpha = alpha # level of confidence
		self.bounds = {}
		self.coverages = {}

	def fit(self, calib):
		
		"""
			Parameters
			----------
			- calib: contains estimated (est) and ground truth (gt)
					 metric(s) for calibration dataset
				- type: dict with keys 'est' and 'gt'
				
				- calib['est']: estimated metrics
					- type: np.array of size (n_scenes, n_recons, n_metrics)
				- calib['gt']:  ground truth metrics
					- type: np.array of size (n_scenes, n_metrics)

			Returns
			----------
			- Cc_lb: calibration dataset upper bound
				- type: np.array of size (n_scenes, n_metrics)
			- Cc_ub: calibration dataset upper bound
				- type: np.array of size (n_scenes, n_metrics)
			- calib_calibrated_coverage: calibration dataset coverage [0, 1]
				- type: float
		"""

		self.calib = calib
		self.n_calib = calib['est'].shape[0]
		self.adj_alpha = np.ceil((self.n_calib+1)*(1-self.alpha))/self.n_calib
		# calibrate
		calib_lb = np.quantile(calib['est'], q=self.alpha/2, axis=1)
		calib_ub = np.quantile(calib['est'], q=1-self.alpha/2, axis=1)
		scores = (np.maximum(calib_lb - calib['gt'], calib['gt'] - calib_ub)).flatten()
		self.q = np.quantile(scores, self.adj_alpha, method='higher')
		# get prediction set
		Cc_lb, Cc_ub = calib_lb - self.q, calib_ub + self.q
		# compute coverage	
		calib_calibrated_coverage = self._coverage(Cc_lb, Cc_ub, calib['gt'])
		# save data
		self.bounds['cc_ub'] = Cc_ub
		self.bounds['cc_lb'] = Cc_lb
		self.coverages['cc'] = calib_calibrated_coverage
		return Cc_ub, Cc_lb, calib_calibrated_coverage

	def predict(self, test):

		"""
			Parameters
			----------
			- test: contains estimated (est) and ground truth (gt)
					 metric(s) for testing dataset
				- type: dict with keys 'est' and 'gt'
				
				- test['est']: estimated metrics
					- type: np.array of size (n_scenes, n_recons, n_metrics)
				- test['gt']: ground truth metrics
					- type: np.array of size (n_scenes, n_metrics)

			Returns
			----------
			- Ct_lb: testing dataset upper bound
				- type: np.array of size (n_scenes, n_metrics)
			- Ct_ub: testing dataset upper bound
				- type: np.array of size (n_scenes, n_metrics)
			- test_calibrated_coverage: testing dataset coverage [0, 1]
				- type: np.array of size (n_metrics, )
		"""
		
		self.test = test
		test_lb = np.quantile(test['est'], q=self.alpha/2, axis=1)
		test_ub = np.quantile(test['est'], q=1-self.alpha/2, axis=1)
		Ct_lb, Ct_ub = test_lb - self.q, test_ub + self.q
		self.bounds['ct_ub'] = Ct_ub
		self.bounds['ct_lb'] = Ct_lb
		if 'gt' in test.keys():
			test_calibrated_coverage = self._coverage(Ct_lb, Ct_ub, test['gt'])
			self.coverages['ct'] = test_calibrated_coverage
		else:
			test_calibrated_coverage = None
		return Ct_lb, Ct_ub, test_calibrated_coverage

	def retrieve_bounds(self, x):

		"""
			Retrieves reconstructions and indices from upper and lower bounds

			Parameters
			----------
			- x: input tensor containing reconstructions or metrics
				- type: 
					if reconstruction:
						np.array of size (n_scenes, n_recons, images)
						or np.array of size (n_scenes, n_recons, n_metrics)
			
			Returns
			----------
			- ub: upper bound reconstructions or metrics
				- type: 
					if reconstruction:
						np.array of size (n_scenes, n_metrics, dim_1, ..., dim_n)
					if metrics:
						np.array of size (n_scenes, n_metrics, n_metrics)
			- lb: lower bound reconstructions or metrics
				- type: 
					if reconstruction:
						np.array of size (n_scenes, n_metrics, dim_1, ..., dim_n)
					if metrics:
						np.array of size (n_scenes, n_metrics, n_metrics)
			
			Notes
			----------
			- If reconstructions are of size 256x256 will be of size 
				(n_scenes, n_recons, 256, 256)
			- If metrics, the 1st index returns the metric of choice
				and the 2nd index returns the other metrics associated with the
				upper/lower bound of that metric
		"""

		self.ub_idxs = self._get_argmin_idxs(self.bounds['ct_ub'])
		self.lb_idxs = self._get_argmin_idxs(self.bounds['ct_lb'])
		ubs = np.array([])
		lbs = np.array([])
		# loop through each ub and lb index and stack
		for i in range(x.shape[0]):
			ub_r = np.array([])
			lb_r = np.array([])
			for j in range(len(self.ub_idxs[i])):
				ub_i = x[i, self.ub_idxs[i][j], ...][np.newaxis, ...]
				lb_i = x[i, self.lb_idxs[i][j], ...][np.newaxis, ...]
				ub_r = np.vstack((ub_r, ub_i)) if len(ub_r) else ub_i
				lb_r = np.vstack((lb_r, lb_i)) if len(lb_r) else lb_i
			ub_r = ub_r[np.newaxis, ...]
			lb_r = lb_r[np.newaxis, ...]
			ubs = np.vstack((ubs, ub_r)) if len(ubs) else ub_r
			lbs = np.vstack((lbs, lb_r)) if len(lbs) else lb_r
		return ubs, lbs

	def bound_errors(self, ub_vals, lb_vals):
		"""
			Computes retrieval error, defined as (closest estimate - B)/(UB - LB)
			
			Usage
			---------
			ub_vals, lb_vals = cb.retrieve_bounds(test['est'])
			ub_errs, lb_errs = bound_errors(ub_vals, lb_vals)
		
			Parameters
			---------
            - ub: upper bound metrics
                - type: 
                    np.array of size (n_scenes, n_metrics, n_metrics)
            - lb: lower bound metrics
                - type: 
                    np.array of size (n_scenes, n_metrics, n_metrics)

			Returns
			---------
			- ub: upper bound retrieval errors
                - type: 
                    np.array of size (n_scenes, n_metrics)
            - lb: lower bound retrieval errors
                - type: 
                    np.array of size (n_scenes, n_metrics)
		"""
		return (np.diagonal(ub_vals, axis1=2) - self.bounds['ct_ub'])/(self.bounds['ct_ub'] - self.bounds['ct_lb']), (np.diagonal(lb_vals, axis1=2) - self.bounds['ct_lb'])/(self.bounds['ct_ub'] - self.bounds['ct_lb'])

	def lengths(self, ub_vals, lb_vals):
		"""
			Computes prediction interval lengths for each metric and scene

			Usage
			----------
    		ub_vals, lb_vals = cb.retrieve_bounds(test['est'])
			pi_lengths = lengths(ub_vals, lb_vals)
			
			Parameters
			----------
            - ub_vals: upper bound metrics
                - type: np.array of size (n_scenes, n_metrics, n_metrics)
            - lb_vals: lower bound metrics
                - type: np.array of size (n_scenes, n_metrics, n_metrics)

			Returns
			----------
			- prediction intervals lengths
				- type: np.array(n_scenes, n_metrics)
		"""
		return np.diagonal(ub_vals, axis1=2) - np.diagonal(lb_vals, axis1=2)

	def _get_argmin_idxs(self, bounds):
		return np.argmin(abs(self.test['est']-np.tile(np.expand_dims(bounds, axis=1), (1, self.test['est'].shape[1], 1))), axis=1)

	def _coverage(self, lb, ub, gt):
		
		"""
			Parameters
			----------
			- lb: testing dataset upper bound
				- type: np.array of size (n_scenes, n_metrics)
			- ub: testing dataset upper bound
				- type: np.array of size (n_scenes, n_metrics)
			
			Returns
			---------
			- coverage: testing dataset coverage for each metric [0, 1]
				- type: np.array of size (n_metrics, )
		"""

		return np.sum((gt >= lb) & (gt <= ub), axis=0)/gt.shape[0]
