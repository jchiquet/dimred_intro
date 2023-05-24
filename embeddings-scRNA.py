import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## scRNA liver cell lines
meta_scRNA = pd.read_csv("data/scRNA_metadata.tsv", delimiter="\t")
cell_type = pd.Categorical(meta_scRNA['line'])
counts_scRNA = pd.read_csv("data/scRNA_counts.tsv", delimiter="\t")

log_offset = np.log(1 + meta_scRNA['offset']).to_numpy()
log_scRNA = (np.log(1 + counts_scRNA) - log_offset.reshape((-1,1)) ).to_numpy()    

from sklearn.manifold import MDS, Isomap, SpectralEmbedding

embedding = MDS(n_components=2, normalized_stress='auto')
X_MDS = embedding.fit_transform(log_scRNA)

embedding = Isomap(n_components=2)
X_Isomap = embedding.fit_transform(log_scRNA)

embedding = SpectralEmbedding(n_components=2, affinity='rbf')
X_Spectral = embedding.fit_transform(log_scRNA)

sns.scatterplot(x=X_MDS[:, 0], y=X_MDS[:, 1], hue=cell_type)

sns.scatterplot(x=X_Isomap[:, 0], y=X_Isomap[:, 1], hue=cell_type)

sns.scatterplot(x=X_Spectral[:, 0], y=X_Spectral[:, 1], hue=cell_type)
