import os
import pickle
import re
import numpy as np 
import pandas as pd 
import warnings
from copy import deepcopy
from causal_eval.ml_utils import COUNTVECTORIZER
from causal_eval.utils import *
from causal_eval.sampling import *

import statsmodels.api as sm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

from sentence_transformers import SentenceTransformer
from datasets import Dataset
from transformers import BertTokenizer
import torch
from utils.feed_forward import hyperparam_search
from utils.adapter import BERTAdapter

# Set Device ##########################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#######################################################################

# Load Data
fname = './data/subpopA_physics_medicine.csv'
data = pd.read_csv(fname)

# Bag of words
vec = deepcopy(COUNTVECTORIZER)
all_texts = data['X'].to_numpy()
X = vec.fit_transform(all_texts).toarray()

# Load bert tokenizer, sentence transformers & TF-IDF vectorizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
mpnetv2 = SentenceTransformer("all-mpnet-base-v2").to(device)
specter = SentenceTransformer("allenai-specter").to(device)

# Calculate ATE according to the Double Machine Learning estimation procedure
def DML_ate(Y, Y_hat, T, T_hat):
    # Calculate residuals
    Y_resid, T_resid  = Y - Y_hat, T - T_hat

    # Fit linear regression model
    lin_reg = sm.OLS(Y_resid, T_resid).fit()
    return lin_reg.params[0]

# Function to results obtained during estimation
def store_results(current_results, Y_train, Y_test, T_train, T_test, location):
    rct = 0.09557700480435129
    ate = DML_ate(Y, Y_test, T, T_test)
    results = {"ate":ate, "relative_abs_error":(np.abs(ate - rct)/rct),
               "outcome_train":Y_train,
               "outcome_test":Y_test,
               "treatment_train":T_train,
               "treatment_test":T_test,
    }
    current_results.append(results)
    with open(location, 'wb') as file:
        pickle.dump(current_results, file)

# Initialize the TF-IDF vectorizer
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # get rid of numbers
    return text
vectorizer = TfidfVectorizer(max_features=2000, lowercase=True, strip_accents="unicode", 
                             stop_words="english", max_df=0.9, min_df=5, binary=True, preprocessor=preprocess_text)

# Set rejection sampling parameters
num_seeds_one_setting = 100
zeta0, zeta1, random_state = 0.85, 0.15, 0
x, y, data_resampled_all_seeds = one_hyperparam_setting(data, zeta0, zeta1, num_seeds_one_setting)
data_resampled_dict_all_seeds = resampled_data_cleanup(data_resampled_all_seeds, vec)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
results_BA, results_MP, results_SP, results_TF = [], [], [], []
for index in range(len(data_resampled_dict_all_seeds)):
    torch.cuda.empty_cache()

    # Load & encode data subset
    texts = data_resampled_all_seeds[index]["X"].values
    Y = data_resampled_dict_all_seeds[index]["Y"]
    T = data_resampled_dict_all_seeds[index]["T"]
    X = data_resampled_dict_all_seeds[index]["X"]

    # Prepare BERT adapter data
    tokenized = Dataset.from_pandas(data_resampled_all_seeds[index]).rename_column("X", "decoded").remove_columns("__index_level_0__")
    def encode_batch(batch):
        return tokenizer(batch["decoded"], truncation=True, padding="max_length")
    tokenized = tokenized.map(encode_batch, batched=True)

    # Encode embeddings for sentence transformers and the TF-IDF representation
    X_MP = mpnetv2.encode(texts)
    X_SP = specter.encode(texts)
    X_TF = vectorizer.fit_transform(texts).toarray()

    # Initialize result arrays
    L = Y.shape[0]
    Y_train_BA, Y_test_BA, T_train_BA, T_test_BA = np.empty(L), np.empty(L), np.empty(L), np.empty(L)
    Y_train_MP, Y_test_MP, T_train_MP, T_test_MP = np.empty(L), np.empty(L), np.empty(L), np.empty(L)
    Y_train_SP, Y_test_SP, T_train_SP, T_test_SP = np.empty(L), np.empty(L), np.empty(L), np.empty(L)
    Y_train_TF, Y_test_TF, T_train_TF, T_test_TF = np.empty(L), np.empty(L), np.empty(L), np.empty(L)
    
    # Create a crossfit split
    split_var = [f"{i}_{j}" for i, j in zip(T, Y)]
    crossfit_split = list(
        StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state).split(X, split_var)
    )
    for crossfit_number, (train_inds, test_inds) in enumerate(crossfit_split):

        # BERT Adapter
        Y_model_BA = BERTAdapter(dataset=tokenized, num_epochs=5, batch_size=64, learning_rate=3e-4, category="C", group="Y", device=device)
        Y_model_BA.fit(train_inds, None)
        Y_test_BA[test_inds] = Y_model_BA.predict_proba(test_inds)[:, 1]
        Y_train_BA[train_inds] = Y_model_BA.predict_proba(train_inds)[:, 1]

        T_model_BA = BERTAdapter(dataset=tokenized, num_epochs=5, batch_size=64, learning_rate=3e-4, category="C", group="T", device=device)
        T_model_BA.fit(train_inds, None)
        T_test_BA[test_inds] = T_model_BA.predict_proba(test_inds)[:, 1]
        T_train_BA[train_inds] = T_model_BA.predict_proba(train_inds)[:, 1]

        learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
        # MPNet v2
        Y_test_MP[test_inds], Y_train_MP[train_inds]  = hyperparam_search(train_inds, test_inds, X_MP, Y, 50, 128, learning_rates, "C", False, device=device)
        T_test_MP[test_inds], T_train_MP[train_inds] = hyperparam_search(train_inds, test_inds, X_MP, T, 50, 128, learning_rates, "C", False, device=device)

        # SPECTER
        Y_test_SP[test_inds], Y_train_SP[train_inds]  = hyperparam_search(train_inds, test_inds, X_SP, Y, 50, 128, learning_rates, "C", False, device=device)
        T_test_SP[test_inds], T_train_SP[train_inds] = hyperparam_search(train_inds, test_inds, X_SP, T, 50, 128, learning_rates, "C", False, device=device)

        # TF-IDF
        Y_test_TF[test_inds], Y_train_TF[train_inds]  = hyperparam_search(train_inds, test_inds, X_TF, Y, 50, 128, learning_rates, "C", True, device=device)
        T_test_TF[test_inds], T_train_TF[train_inds] = hyperparam_search(train_inds, test_inds, X_TF, T, 50, 128, learning_rates, "C", True, device=device)

    # Store Results
    store_results(results_BA, Y_train_BA, Y_test_BA, T_train_BA, T_test_BA, "results/recreate/adapter.pkl")
    store_results(results_MP, Y_train_MP, Y_test_MP, T_train_MP, T_test_MP, "results/recreate/mpnet.pkl")
    store_results(results_SP, Y_train_SP, Y_test_SP, T_train_SP, T_test_SP, "results/recreate/specter.pkl")
    store_results(results_TF, Y_train_TF, Y_test_TF, T_train_TF, T_test_TF, "results/recreate/tf-idf.pkl")