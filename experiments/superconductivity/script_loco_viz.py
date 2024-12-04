import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dipd.plots import forceplot
from dipd.consts import PLOTS_FONT_AESTHETICS

seed = 10
folds = 10
savepath = 'experiments/superconductivity/'

result_df_mean = pd.read_csv(savepath + f'superconductivity_loo_{folds}fold_{seed}_mean.csv', index_col=0)

font_aesthetics = PLOTS_FONT_AESTHETICS.copy()
font_aesthetics['fontsize'] = 6.5

plt.figure()
ax = forceplot(result_df_mean.T, 'Superconductivity', figsize=(11, 8.2), 
               ylabel='Normalized Scores',
               separator_ident_prop=0.01,
               explain_surplus=True, xticklabel_rotation=45,
               hline_thickness=0.8, hline_width=0.8, **font_aesthetics)
ax.get_legend().remove()
plt.savefig(savepath + f'superconductivity_loo_{folds}fold_{seed}.pdf', bbox_inches='tight')
