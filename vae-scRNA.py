## basic modules
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## modules for data importation
import scanpy as sc
import anndata as ad
from data import datasets
from anndata.experimental.pytorch import AnnLoader

## scRNA liver cell lines
meta_scRNA = pd.read_csv("scRNA_metadata.tsv", delimiter="\t")
with open("scRNA_counts.tsv") as scRNA:
    adata = ad.read_csv(scRNA, delimiter="\t")
adata.obs_names = meta_scRNA['name']
adata.obs["cell_type"] = pd.Categorical(meta_scRNA['line'])

## Set torch device to cuda
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader = AnnLoader(adata, batch_size=256, shuffle=True,use_cuda=True)

## Homemade VAE modules
from models.vae import VAE
from models.distributions.nb import NegativeBinomial
from models.distributions.cb import ContinuousBernoulli
from models.distributions.zinb import ZeroInflatedNegativeBinomial
from models.distributions.poisson import Poisson
from models.distributions.zipoisson import ZIPoisson

## _________________________________________________________________
## VAE with Poisson loss
hidden_layer_list = [256, 128, 64]
latent_dim = 2
learning_rate = 1e-3
nb_epochs = 150

my_vae_poisson = VAE(
    input_dim=adata.n_vars,
    hidden_dims=hidden_layer_list,
    latent_dim=latent_dim,
    pxz_distribution=ZeroInflatedNegativeBinomial,
)
my_vae_poisson.to(device)

optimizer = torch.optim.Adam(my_vae_poisson.parameters(), lr=learning_rate)

vae_poisson_loss, vae_poisson_rec_x = my_vae_poisson.train_(
    dataloader, device, optimizer, nb_epochs
)

plt.plot(vae_poisson_loss)
plt.xlabel("Epochs")
plt.ylabel("Loss value")
plt.title("Loss value over the epochs of the training")
plt.show()

Z_vae = my_vae_poisson.encode(dataloader.dataset[:].X)[0].data.cpu().numpy()
labels = adata.obs["cell_type"]
sns.scatterplot(x=Z_vae[:, 0], y=Z_vae[:, 1], hue=labels)

_______________________________________________________________________
## Embedding: PCA, UMPA, t-SNE
adata_log = adata.copy()
log_offset = np.log(1 + meta_scRNA['offset']).to_numpy()
adata_log.X = np.log(1 + adata_log.X) - log_offset.reshape((-1,1))

sc.pp.pca(adata_log)
sc.pp.neighbors(adata_log)
sc.tl.pca(adata_log, n_comps=2)
sc.tl.umap(adata_log, n_components=2)
sc.tl.tsne(adata_log)

plt.figure(figsize=(16, 12))
fig, axs = plt.subplots(1, 3)
sc.pl.pca(adata_log, show=False, color=["cell_type"], title="PCA", ax=axs[0])
sc.pl.tsne(adata_log, show=False, color=["cell_type"], title="t-SNE", ax=axs[1])
sc.pl.umap(adata_log, show=False, color=["cell_type"], title="UMAP", ax=axs[2])
plt.show()

sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.pca(adata, n_comps=2)
sc.tl.umap(adata, n_components=2)
sc.tl.tsne(adata)

plt.figure(figsize=(16, 12))
fig, axs = plt.subplots(1, 3)
sc.pl.pca(adata, show=False, color=["cell_type"], title="PCA", ax=axs[0])
sc.pl.tsne(adata, show=False, color=["cell_type"], title="t-SNE", ax=axs[1])
sc.pl.umap(adata, show=False, color=["cell_type"], title="UMAP", ax=axs[2])
plt.show()
