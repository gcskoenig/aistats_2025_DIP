import random
import pandas as pd
import numpy as np

seed = 10
random.seed(seed)
np.random.seed(seed)

savepath = 'experiments/online-news/'

df = pd.read_csv(savepath + 'OnlineNewsPopularity.csv')

# remove leading whitespace from column names
df.columns = df.columns.str.lstrip()

target_name = 'shares'
drop_vars = ['url', 'timedelta']
df = df.drop(drop_vars, axis=1)

## kfold
from sklearn.model_selection import KFold
from dipd import DIP
from dipd.learners import EBM

folds = 10
wrk = DIP(df, target_name, EBM)

kf = KFold(n_splits=folds, shuffle=True)
splits = kf.get_n_splits(df)
X, y = df.drop(target_name, axis=1), df[target_name]

results = []
for i, (train_index, test_index) in enumerate(kf.split(df)):
    print(f'Fold {i+1}/{folds}')
    X_train, y_train, X_test, y_test = X.iloc[train_index], y.iloc[train_index], X.iloc[test_index], y.iloc[test_index]
    wrk.set_split(X_train, X_test, y_train, y_test)
    ex_loo = wrk.get_all_loo()
    scores = ex_loo.scores
    results.append(ex_loo)

resultsscores = [result.scores for result in results]
result_df = pd.concat(resultsscores, axis=0)

# correct for negative pure interactions
pure_interactions = result_df['pure_interactions']
pure_interactions.iloc[np.where(result_df['pure_interactions'] <= 0)] = 0.0
result_df['pure_interactions'] = pure_interactions

# group by index and average within groups
result_df_mean = result_df.groupby(result_df.index).mean().copy()
result_df_std = result_df.groupby(result_df.index).std()
result_df_mean.to_csv(savepath + f'popularity_loo_{folds}fold_{seed}_mean.csv')
result_df_std.to_csv(savepath + f'popularity_loo_{folds}fold_{seed}_std.csv')