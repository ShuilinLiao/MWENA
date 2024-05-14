## Simulated Data
python stim_binary.py --num_features 600 --num_examples 100 --set_seed 42 --sigma 2 --IR 1 --lmbda 0.01 --lmbda2 0.1

## Real data
## Feature filtering
Rscript S1.exoRBase_merge.R

## MWENA
python S2_exo_binary.py --proj "exoRBase" --proj1 "Benign" --proj2 "CRC" --lmbda 0.008 --lmbda2 0.01
python S3_exo_multi.py  --proj 'exoRBase' --lmbda 0.01 --lmbda2 0.1

## feature selection
python S4_featureSelection_binary.py --item 'b' --proj 'exoRBase' --proj1 "Benign" --proj2 "CRC" --num_epochs '49'
python S5_featureSelection_multi.py --item 'm' --proj "exoRBase" --num_epochs '49'