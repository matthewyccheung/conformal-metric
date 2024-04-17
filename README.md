# Metric-guided Reconstruction Bounds at Test Time via Conformal Prediction

This repository contains source code and toy dataset for our [paper](https://arxiv.org). The implementation is self-contained in `mg.py`.

## Abstract

Recent advancements in machine learning have led to novel image reconstruction algorithms that address inherently ill-posed problems. Assessing their trustworthiness and understanding how to deploy them safely at test time remains an important and open problem. We propose a method that leverages conformal prediction to retrieve upper/lower bounds and statistical inliers/outliers of reconstructions based on the prediction intervals of downstream metrics. We apply our method to sparse-view CT for downstream radiotherapy planning and show 1) that metric-guided bounds have valid coverage for downstream metrics while conventional pixel-wise bounds do not and 2) anatomical differences of upper/lower bounds between metric-guided and pixel-wise methods. Our work paves the way for more meaningful reconstruction bounds.

## Citation
Cheung, Matt Y., et al. "Metric-guided Reconstruction Bounds at Test Time via Conformal Prediction." arXiv preprint arXiv: (2024).