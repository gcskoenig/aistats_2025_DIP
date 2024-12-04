import pandas as pd
import numpy as np
import random

seed = 10
random.seed(seed)
np.random.seed(seed)
savepath = 'experiments/superconductivity/'

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
superconductivty_data = fetch_ucirepo(id=464) 
  
# data (as pandas dataframes) 
X = superconductivty_data.data.features 
y = superconductivty_data.data.targets 
target_variable = y.columns[0]

df = pd.concat([X, y], axis=1)

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score


from dipd import DIP
from dipd.learners import EBM
from sklearn.model_selection import KFold

folds = 10

wrk = DIP(df, target_variable, EBM)

kf = KFold(n_splits=folds, shuffle=True)
splits = kf.get_n_splits(df)
X, y = df.drop(target_variable, axis=1), df[target_variable]

results = []
for i, (train_index, test_index) in enumerate(kf.split(df)):
    print(f'Fold {i+1}/{folds}')
    X_train, y_train, X_test, y_test = X.iloc[train_index], y.iloc[train_index], X.iloc[test_index], y.iloc[test_index]
    wrk.set_split(X_train, X_test, y_train, y_test)
    ex_loo = wrk.get_all_loo()
    scores = ex_loo.scores
    results.append(ex_loo)

    # save result
    resultsscores = [result.scores for result in results]
    result_df = pd.concat(resultsscores, axis=0)

    # group by index and average within groups
    result_df_mean = result_df.groupby(result_df.index).mean().copy()
    result_df_std = result_df.groupby(result_df.index).std()
    result_df_mean.to_csv(savepath + f'superconductivity_loo_{folds}fold_{seed}_mean.csv')
    result_df_std.to_csv(savepath + f'superconductivity_loo_{folds}fold_{seed}_std.csv')
