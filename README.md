# DoubleLingo

This repository contains the code for "DoubleLingo: Causal Estimation with Large Language Models".

## Setup
```
conda create -y --name DoubleLingo python==3.9
conda activate DoubleLingo
pip install -r requirements.in
pip install -e .
pip uninstall transformers
pip install adapter-transformers==3.2.1
```

Setup is slightly tricky since ```adapter-transformers``` is a direct fork of ```transformers```, but ```sentence-transformers``` automatically installs ```transformers``` which ideally should not be installed in the same environment as ```adapter-transformers```. Doing the above is a simple quick fix.

## View & Recreate Results
To view results presented in the paper in detail, run the notebook ```explore_results.ipynb```, where ```path="paper"```.

To recreate results, run ```estimate.py```, and then run the above notebook with ```path="recreate"```

Make sure to set the appropriate device at the top of this script. It is currently set to ```torch.device("cuda:0")``` which is automatically parallelized together with a ```torch.device("cuda:1")``` device to train the adapter's 128 batch size split across two 64 batch sizes.

If only using 1 GPU, set ```BERTAdapter(... batch_size=128 ...)``` within ```estimate.py``` 


## Using adapter and FFN estimators
Our adapter and feed-forward relu network are also ```scikit-learn``` style predictors, with the following methods:
```
fit(self, X, y)
predict(self, X)
predict_proba(self, X)
```
which can be used outside of causal estimation. The classes are found in ```utils/adapter.py``` and ```utils/feed_forward.py```.


## Data
The ```data/subpopA_physics_medicine.csv``` and ```causal_eval``` are taken from the RCT Rejection Sampling codebase (https://github.com/kakeith/rct_rejection_sampling/blob/main/README.md). These implement the RCT rejection sampling algorithm and contain the baseline Logistic Regression and CatBoost 
