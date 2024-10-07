import numpy as np
import matplotlib.pyplot as plt
import pdb
import pandas as pd
import torch
from math import ceil

class MetricGuidedCalibration:
	"""
		Description
		----------
		Implements metric-guided calibration from our paper: Cheung, Matt Y., et al. "Metric-guided Image Reconstruction Bounds via Conformal Prediction." arXiv preprint arXiv:2404.15274 (2024).https://arxiv.org/abs/2404.15274
		See our paper for more information.

		Usage:
		----------
        cb = MetricGuidedCalibration(alpha=alpha, method=method)
		cc_lb, cc_ub, cc_coverages = cb.fit(c_ys, c_yhats)
		ct_lb, ct_ub, ct_coverages = cb.validate(t_ys, t_yhats)
		n_metrics_each_sample, n_sample_all_metrics, inlier_bool, outlier_bool = cb.retrieve_in_out(t_yhats)
		lb_vals, ub_vals = cb.retrieve_bounds(t_yhats)
		lengths = cb.interval_lengths(ct_lb, ct_ub)
		retreived_lengths = cb.retrieval_lengths(lb_vals, ub_vals)
		ub_errs, lb_errs = cb.bound_errors(lb_vals, ub_vals)
	"""
	
	def __init__(self, alpha, method):
		"""
			Description
			----------
			

			Parameters
			----------
			- alpha (float): miscoverage rate [0, 1]
			- method (string): CQR method ['asymCQR', 'CQR', 'CQRub', 'CQRlb']

			Notes
			----------
			- Adjusted level of confidence: alpha_adj = np.ceil((n_c+1)*(1-alpha))/n_c should not be > 1
			- asymCQR: implements CQR with asymmetric adjustments
            - CQR: implements vanilla CQR from [1]
            - CQRub: implements CQR with only the 1st term as confomity score 
						Q_{\alpha/2}(\hat Y^{n+1})-Y^{n+1}
            - CQRlb: implements CQR with only 2nd term as conformity score
						Y^{n+1} - Q_{1-\alpha/2}(\hat Y^{n+1})
		"""
		
		self.alpha = alpha # level of confidence
		self.method = method
		self.bounds = {}
		self.coverages = {}

	def fit(self, y, yhat):
		
		"""
			Description
			----------
			Calibrates model according to data provided in yhat (estimates) and yhat, and method. Saves quantile adjustments as q or (q_lb, q_ub) corresponding to asymmetric adjustments.
	
			Parameters
			----------
			- yhat (np.array, size=(n_scenes, n_recons, n_metrics)): contains estimated metric(s)
			- y (np.array, size=(n_scenes, n_metrics)):  ground truth metrics

			Returns
			----------
			- Cc_lb (np.array, size=(n_scenes, n_metrics)): calibration data adjusted lower bounds
			- Cc_ub (np.array, size=(n_scenes, n_metrics)): calibration data adjusted upperbounds
			- cc_coverage (float): calibrated coverage on calibration dataset [0, 1]
		"""

		self.n_c = yhat.shape[0]
		self.adj_alpha = np.ceil((self.n_c+1)*(1-self.alpha))/self.n_c
		# calibrate
		yhat_lb = np.quantile(yhat, q=self.alpha/2, axis=1)
		yhat_ub = np.quantile(yhat, q=1-self.alpha/2, axis=1)
		if self.method=='CQR':
			scores = np.maximum(yhat_lb-y, y-yhat_ub)
			self.q = np.quantile(scores, self.adj_alpha, interpolation='higher', axis=0)
		elif self.method=='CQRlb':
			scores = yhat_lb-y
			self.q = np.quantile(scores, self.adj_alpha, interpolation='higher', axis=0)
		elif self.method=='CQRub':
			scores = y-yhat_ub
			self.q = np.quantile(scores, self.adj_alpha, interpolation='higher', axis=0)
		elif self.method=='CQR-m':
			yhat_med = np.median(yhat, axis=1)
			scores = np.maximum((yhat_lb-y)/(yhat_med-yhat_lb),(y-yhat_ub)/(yhat_ub-yhat_med))
			self.q = np.quantile(scores, self.adj_alpha, interpolation='higher', axis=0)
		elif self.method=='CQR-r':
			scores = (np.maximum((yhat_lb-y)/(yhat_ub-yhat_lb),(y-yhat_ub)/(yhat_ub-yhat_lb)))
			self.q = np.quantile(scores, self.adj_alpha, interpolation='higher', axis=0)
		elif self.method=='asymCQR':
			scores_lb = yhat_lb-y
			scores_ub = y-yhat_ub
			self.adj_alpha = np.ceil((self.n_c+1)*(1-self.alpha/2))/self.n_c
			self.q_lb = np.quantile(scores_lb, self.adj_alpha, interpolation='higher', axis=0)
			self.q_ub = np.quantile(scores_ub, self.adj_alpha, interpolation='higher', axis=0)

		cc_lb, cc_ub, cc_coverages = self.validate(y, yhat)
		self.coverages['cc'] = cc_coverages
		return cc_lb, cc_ub, cc_coverages

	def validate(self, y, yhat):

		"""
			Description
			----------
			Validates model according to data provided in yhat (estimates) and yhat, and method. 
    
            Parameters
            ----------
            - yhat (np.array, size=(n_scenes, n_recons, n_metrics)): contains estimated metric(s)
            - y (np.array, size=(n_scenes, n_metrics)):  ground truth metrics		
			Parameters
			----------
			- yhat (np.array, size=(n_scenes, n_recons, n_metrics)): contains estimated metric(s)
            - y (np.array, size=(n_scenes, n_metrics)):  ground truth metrics

			Returns
			----------
			- ct_lb (np.array, size=(n_scenes, n_metrics)): testing data adjusted upper bound
			- ct_ub (np.array, size (n_metrics, )): testing data adjusted lower bound
			- ct_coverage (float): calibrated coverage on calibration dataset [0, 1]
		"""
		
		yhat_lb = np.quantile(yhat, q=self.alpha/2, axis=1)
		yhat_ub = np.quantile(yhat, q=1-self.alpha/2, axis=1)
		if (self.method=='CQR')|(self.method=='CQRub')|(self.method=='CQRlb'):
			self.q_lb = self.q
			self.q_ub = self.q
		elif self.method=='CQR-m':
			yhat_med = np.median(yhat, axis=1)
			self.q_lb = self.q*(yhat_med-yhat_lb)
			self.q_ub = self.q*(yhat_ub-yhat_med)
		elif self.method=='CQR-r':
			test_med = np.median(yhat, axis=1)
			self.q_lb = self.q*(yhat_ub-yhat_lb)
			self.q_ub = self.q_lb
		elif self.method=='asymCQR':
			pass
		ct_lb, ct_ub = yhat_lb-self.q_lb, yhat_ub+self.q_ub
		self.bounds['ct_ub'] = ct_ub
		self.bounds['ct_lb'] = ct_lb
		ct_coverages = self._coverage(ct_lb, ct_ub, y)
		self.coverages['ct'] = ct_coverages
		return ct_lb, ct_ub, ct_coverages

	def retrieve_bounds(self, x):

		"""
			Description
			----------
			Retrieves reconstructions and indices based on calibrated upper and lower bounds

			Parameters
			----------
			- x (np.array, size=(n_scenes, n_recons, n_pixels) for reconstructions or (n_scenes, n_recons, n_metrics) for metrics): input tensor containing reconstructions or metrics you want to retrieve closest upper and lower bounds of.
			
			Returns
			----------
			- ub (np.array, size=(n_scenes, n_metrics, dim_1, ..., dim_n) for reconstructions or size=(n_scenes, n_metrics, n_metrics) for metrics): upper bound reconstructions or metrics
			- lb(np.array, size=(n_scenes, n_metrics, dim_1, ..., dim_n) for reconstructions or size=(n_scenes, n_metrics, n_metrics) for metrics): lower bound reconstructions or metrics
		
			Notes
			----------
			- If reconstructions are of size 256x256 will be of size 
				(n_scenes, n_recons, 256, 256)
			- If metrics, the 1st index returns the metric of choice
				and the 2nd index returns the other metrics associated with the
				upper/lower bound of that metric
		"""

		self.ub_idxs = self._get_argmin_idxs(self.bounds['ct_ub'], x)
		self.lb_idxs = self._get_argmin_idxs(self.bounds['ct_lb'], x)
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
		return lbs, ubs

	def retrieve_in_out(self, x):
		"""
			Description
			----------
			Determine 1) how many metrics each sample's metrics are contained within the calibrated bounds and 2) how many samples' metrics are contained within all calibration bounds
			
			Parameters
			----------
			- x (np.array, size=(n_scenes, n_recons, n_pixels) for reconstructions or (n_scenes, n_recons, n_metrics) for metrics): input tensor containing reconstructions or metrics you want to retrieve closest upper and lower bounds of.
		
		    Returns
            ----------
			- n_metrics_each_samples (np.array[int], size=(n_scenes, n_recons, 1)): how many metrics each sample's metrics are contained within the calibrated bounds
			- n_sample_all_metrics (np.array[bool], size=(n_scenes, n_recons, 1))
			- inlier_bool (np.array[float], size=(n_scenes, n_recons, n_metrics)): boolean array showing whether each sample in each scene have metrics contained within the calibrated bounds
			- outlier_bool (np.array[float], size=(n_scenes, n_recons, n_metrics)):boolean array showing whether each sample in each scene have one or more metrics not contained within the calibrated bounds
		"""

		# determine whether estimates lie in bounds
		inlier_bool = (x<np.tile(np.expand_dims(self.bounds['ct_ub'],axis=1),(1,x.shape[1],1)))&(x>np.tile(np.expand_dims(self.bounds['ct_lb'], axis=1), (1,x.shape[1],1)))
		outlier_bool = ~inlier_bool
		# how many metrics each sample meets
		n_metrics_each_sample = inlier_bool.sum(0)
		# how many samples meet all requirements
		n_sample_all_metrics = inlier_bool.sum(1)
		return n_metrics_each_sample, n_sample_all_metrics, inlier_bool, outlier_bool

	def bound_errors(self, lb_vals, ub_vals):
		"""
			Description
			----------
			Computes retrieval error, defined as (closest estimate - B)/(UB - LB)
		
			Parameters
			----------
            - ub_vals (np.array, size=(n_scenes, n_metrics, n_metrics)): metrics from samples with closest estimate to calibrated upper bound
            - lb_vals (np.array, size=(n_scenes, n_metrics, n_metrics)): metrics from samples with closest estimate to calibrated lower bound

			Returns
			----------
			- ub_errs (np.array, size=(n_scenes, n_metrics)): upper bound retrieval errors
            - lb_errs (np.array, size=(n_scenes, n_metrics)): lower bound retrieval errors
		"""

		return (np.diagonal(ub_vals, axis1=2) - self.bounds['ct_ub'])/(self.bounds['ct_ub'] - self.bounds['ct_lb']), (np.diagonal(lb_vals, axis1=2) - self.bounds['ct_lb'])/(self.bounds['ct_ub'] - self.bounds['ct_lb'])

	def interval_lengths(self, lb, ub):
		return ub-lb

	def retrieval_lengths(self, lb_vals, ub_vals):
		"""
			Description
			----------
			Computes prediction interval lengths for each metric and scene
			
			Parameters
			----------
            - ub_vals (np.array, size=(n_scenes, n_metrics, n_metrics)): upper bound metrics
            - lb_vals (np.array, size=(n_scenes, n_metrics, n_metrics)): lower bound metrics

			Returns
			----------
			- prediction intervals lengths (np.array(n_scenes, n_metrics))
		"""
		return np.diagonal(ub_vals, axis1=2) - np.diagonal(lb_vals, axis1=2)

	def _get_argmin_idxs(self, bounds, yhat):
		# internal function
		return np.argmin(abs(yhat-np.tile(np.expand_dims(bounds, axis=1), (1, yhat.shape[1], 1))), axis=1)

	def _coverage(self, lb, ub, gt):
		
		"""
			Description
			----------
			Computes coverage defined by % of time lb<=gt<=ub where lb and ub are calibrated upper and lower bounds and gt is the ground truth metrics

			Parameters
			----------
			- lb (np.array, size=(n_scenes, n_metrics)): testing dataset calibrated lower bound
			- ub (np.array, size=(n_scenes, n_metrics)): testing dataset calibrated upper bound
			- gt (np.array, size=(n_scenes, n_metrics)): testing dataset ground truth values
			
			Returns
			---------
			- coverage (np.array, size=(n_metrics, )): testing dataset coverage for each metric [0, 1]
		"""
		return np.sum((gt >= lb) & (gt <= ub), axis=0)/gt.shape[0]

class DiffCQR:
	def __init__(self, alpha, method):	
		self.alpha = alpha # level of confidence
		self.method = method
		self.bounds = {}
		self.coverages = {}

	def fit(self, y, yhat):
		self.n_c = yhat.shape[0]
		self.adj_alpha = np.ceil((self.n_c+1)*(1-self.alpha))/self.n_c
		# calibrate
		yhat_lb = torch.quantile(yhat, q=self.alpha/2, axis=1)
		yhat_ub = torch.quantile(yhat, q=1-self.alpha/2, axis=1)
		if self.method=='CQR':
			scores = torch.maximum(yhat_lb-y, y-yhat_ub)
			self.q = torch.quantile(scores, self.adj_alpha, interpolation='higher', axis=0)
		elif self.method=='CQRlb':
			scores = yhat_lb-y
			self.q = torch.quantile(scores, self.adj_alpha, interpolation='higher', axis=0)
		elif self.method=='CQRub':
			scores = y-yhat_ub
			self.q = torch.quantile(scores, self.adj_alpha, interpolation='higher', axis=0)
		elif self.method=='CQR-m':
			yhat_med = torch.median(yhat, axis=1).values
			scores = torch.maximum((yhat_lb-y)/(yhat_med-yhat_lb),(y-yhat_ub)/(yhat_ub-yhat_med))
			self.q = torch.quantile(scores, self.adj_alpha, interpolation='higher', axis=0)
		elif self.method=='CQR-r':
			scores = (torch.maximum((yhat_lb-y)/(yhat_ub-yhat_lb),(y-yhat_ub)/(yhat_ub-yhat_lb)))
			self.q = torch.quantile(scores, self.adj_alpha, interpolation='higher', axis=0)
		elif self.method=='asymCQR':
			scores_lb = yhat_lb-y
			scores_ub = y-yhat_ub
			self.adj_alpha = np.ceil((self.n_c+1)*(1-self.alpha/2))/self.n_c
			self.q_lb = torch.quantile(scores_lb, self.adj_alpha, interpolation='higher', axis=0)
			self.q_ub = torch.quantile(scores_ub, self.adj_alpha, interpolation='higher', axis=0)

		cc_lb, cc_ub, cc_coverages = self.validate(y, yhat)
		self.coverages['cc'] = cc_coverages
		return cc_lb, cc_ub, cc_coverages

	def validate(self, y, yhat):	
		yhat_lb = torch.quantile(yhat, q=self.alpha/2, axis=1)
		yhat_ub = torch.quantile(yhat, q=1-self.alpha/2, axis=1)
		if (self.method=='CQR')|(self.method=='CQRub')|(self.method=='CQRlb'):
			self.q_lb = self.q
			self.q_ub = self.q
		elif self.method=='CQR-m':
			yhat_med = torch.median(yhat, axis=1).values
			self.q_lb = self.q*(yhat_med-yhat_lb)
			self.q_ub = self.q*(yhat_ub-yhat_med)
		elif self.method=='CQR-r':
			test_med = torch.median(yhat, axis=1).values
			self.q_lb = self.q*(yhat_ub-yhat_lb)
			self.q_ub = self.q_lb
		elif self.method=='asymCQR':
			pass
		ct_lb, ct_ub = yhat_lb-self.q_lb, yhat_ub+self.q_ub
		self.bounds['ct_ub'] = ct_ub
		self.bounds['ct_lb'] = ct_lb
		ct_coverages = self._coverage(ct_lb, ct_ub, y)
		self.coverages['ct'] = ct_coverages
		return ct_lb, ct_ub, ct_coverages

	def interval_lengths(self, lb, ub):
		return ub-lb

	def _coverage(self, lb, ub, gt):
		return torch.sum((gt >= lb) & (gt <= ub), axis=0)/gt.shape[0]

class DiffL1:
    def __init__(self, alpha, method):
        self.alpha = alpha # level of confidence
        self.method = method
        self.bounds = {}
        self.coverages = {}

    def fit(self, y, yhat):
        self.n_c = len(yhat)
        scores = abs(y-yhat)
        scores_lb = yhat-y
        scores_ub = y-yhat
        if self.method == 'sym':
            self.adj_alpha = ceil((self.n_c+1)*(1-self.alpha))/self.n_c
            q = torch.quantile(scores, self.adj_alpha, interpolation='higher')
            self.q_lb = q
            self.q_ub = q
        elif self.method == 'asym':
            self.adj_alpha = ceil((self.n_c+1)*(1-self.alpha/2))/self.n_c
            self.q_lb = torch.quantile(scores_lb, self.adj_alpha, interpolation='higher')
            self.q_ub = torch.quantile(scores_ub, self.adj_alpha, interpolation='higher')
        cc_lb, cc_ub, cc_coverages = self.validate(y, yhat)
        self.coverages['cc'] = cc_coverages
        return cc_lb, cc_ub, cc_coverages

    def validate(self, y, yhat):
        ct_lb, ct_ub = yhat-self.q_lb, yhat+self.q_ub
        self.bounds['ct_ub'] = ct_ub
        self.bounds['ct_lb'] = ct_lb
        ct_coverages = self._coverage(ct_lb, ct_ub, y)
        self.coverages['ct'] = ct_coverages
        return ct_lb, ct_ub, ct_coverages

    def interval_lengths(self, lb, ub):
        return ub-lb

    def _coverage(self, lb, ub, gt):
        return ((gt>=lb)&(gt<=ub)).sum()/len(gt)
