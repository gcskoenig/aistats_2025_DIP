import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dipd.learners import EBM, LinearGAM
from dipd import DIP
from dipd.consts import PLOTS_FONT_AESTHETICS
from dipd.plots import forceplot


savepath = 'experiments/appendix_examples/figures/'

## dgp

N = 10**6
x = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=N)
df = pd.DataFrame(x, columns=['x1', 'x2'])

# dgp 1
df1 = df.copy()
df1['y'] = -4.3 * df['x1'] - 0.9 * df['x2'] - 3.9 * df['x1']**2 + 3.0 * df['x2']**2

# dgp 1
df2 = df.copy()
df2['y'] = -1.3 * df['x1'] - 4.7 * df['x2'] + 3.6 * df['x1']**2 - 3.0 * df['x2']**2 + 4.7 * df['x1'] * df['x2']

# dgp 1
df3 = df.copy()
df3['y'] = 10.9 * df['x1'] + 2.4 * df['x2'] - 5.1 * df['x1']**2 - 5.3 * df['x2']**2 + 11.3 * df['x1'] * df['x2']


## dip decomposition

wrk = DIP(df1, 'y', EBM)
ex_loo1 = wrk.get(['x1', 'x2'])

wrk = DIP(df2, 'y', EBM)
ex_loo2 = wrk.get(['x1', 'x2'])

wrk = DIP(df3, 'y', EBM)
ex_loo3 = wrk.get(['x1', 'x2'])

scores = pd.DataFrame({'DGP 1': ex_loo1, 'DGP 2': ex_loo2, 'DGP 3': ex_loo3})
scores.to_csv(savepath + '3DGPs.csv')

## plot

scores = pd.read_csv(savepath + '3DGPs.csv', index_col=0)

# get axes
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
forceplot(scores, '', figsize=(8, 4), ylabel='Normalized Scores', **PLOTS_FONT_AESTHETICS, 
          hline_thickness=0.8, hline_width=0.4, ax=ax, bar_width=0.3,
          split_additive=True)
# remove legend
plt.legend().remove()
plt.tight_layout()
plt.savefig(savepath + '3DGPs.pdf', bbox_inches='tight')
plt.close()
