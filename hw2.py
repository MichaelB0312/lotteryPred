import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.nonparametric.smoothers_lowess as sn
from scipy.stats import linregress
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
universal = './universal-20240227'
out_put = './clean_file'
if os.path.exists(out_put) == False:
    os.mkdir(out_put)
plt.figure(figsize=(10,10))
i = 0
for file in os.listdir(universal):
    file_path = os.path.join(universal, file)
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['gene_sim'])
    i = i+1
    plt.subplot(3,3,i)
    plt.scatter(x='genome_sim', y='gene_sim', data=df, label='Data Points')
    lowest = sn.lowess(df['gene_sim'], df['genome_sim'], frac=0.2)
    plt.plot(lowest[:,0], lowest[:,1], color='red', label='LOWESS')
    slope, intercept, _, _, _ = linregress(df['genome_sim'], df['gene_sim'])
    reg_line = slope*df['genome_sim']+intercept
    plt.plot(df['genome_sim'].to_numpy(), reg_line.to_numpy(), color='orange', label='Lin. Reg.')
    r2 = r2_score(df['genome_sim'], df['gene_sim'])
    # Define font properties using fontdict
    font = {'family': 'Arial', 'color': 'red', 'weight': 'bold', 'size': 12}
    plt.title(f'Gene name: {file[:-4]} \n, adjusted R-squared: {r2:.3f}', fontdict=font)
    plt.xlabel('genome_sim', fontsize=12)
    plt.ylabel('gene_sim', fontsize=12)
    clean_file = 'clean_' + file
    clean_file_path = os.path.join(out_put, clean_file)
    plt.tight_layout()
    plt.legend()

plt.tight_layout()
plt.show()

    #df.to_csv(out_put, index=False)
