# MWENA

The official source code for [**MWENA: a novel sample re-weighting-based algorithm for disease classification and data interpretation using extracellular vesicles omics data**](https://xx), accepted at xx.

## Requirements

- python                    3.9.7
- scikit-learn              1.2.2
- torch                     1.10.1+cpu

## How to Run

```shell
cd assay-upload
## Simulated Data
python stim_binary.py --num_features 600 --num_examples 100 --set_seed 42 --sigma 2 --IR 1 --lmbda 0.01 --lmbda2 0.1

```

