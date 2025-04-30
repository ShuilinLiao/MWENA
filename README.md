# MWENA

The source code for [**MWENA: a novel sample re-weighting-based algorithm for disease classification and data interpretation using extracellular vesicles omics data**](https://xx/), accepted at xx.



## Requirements

- python 3.8

- scikit-learn   1.3.2

- torch 2.2.1

  

## How to Run

This directory contains three Jupyter notebook files for various tasks:

- **stim_MWENA_CompareModel.ipynb**: This notebook is used for simulating data generation, training MWENA, evaluating its performance on test data, and comparing the results with other models.
- **real_exo_MWENA_train.ipynb**: This notebook trains MWENA using real exosome dataset (Task: CRC-Detection), with the processed data located at ./data/exoRBase_Benign_vs_CRC_merge_full_data.csv. The raw data is sourced from the ExoRBase 2.0 database, accessible at http://www.exoRBase.org.
- **stim_MWENA_CompareModel.ipynb**: This notebook evaluates the performance of MWENA and comparison models on the CRC-Detection task.

