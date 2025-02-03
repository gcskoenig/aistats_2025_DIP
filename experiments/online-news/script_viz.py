import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dipd.plots import forceplot
from dipd.consts import PLOTS_FONT_AESTHETICS

savepath = 'experiments/online-news/'
seed = 10
folds = 10

from dipd.plots import forceplot

result_df_mean = pd.read_csv(savepath + f'popularity_loo_{folds}fold_{seed}_mean.csv', index_col=0)

font_aesthetics = PLOTS_FONT_AESTHETICS.copy()
font_aesthetics['fontsize'] = 6.5

plt.figure()
ax = forceplot(result_df_mean.T, 'Online News Popularity', figsize=(11, 7), 
               ylabel='Normalized Scores',
               separator_ident_prop=0.01,
               explain_surplus=True, xticklabel_rotation=45,
               hline_thickness=0.8, hline_width=0.8, **font_aesthetics)
ax.get_legend().remove()
plt.savefig(savepath + f'popularity_loo_{folds}fold_{seed}.pdf', bbox_inches='tight')
