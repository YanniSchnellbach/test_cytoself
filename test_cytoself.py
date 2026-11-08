#train cytoself model on training data
print("STEP 1: train cytoself model on training data")
from cytoself.trainer.cytoselflite_trainer import CytoselfFullTrainer

from cytoself.datamanager.opencell import DataManagerOpenCell

print("STEP 2: initiate data manager")
data_ch = ['pro', 'nuc']
datapath = 'Testdata'  # path to download sample data
#DataManagerOpenCell.download_sample_data(datapath)  # donwload data
datamanager = DataManagerOpenCell(datapath, data_ch, fov_col=None)
datamanager.const_dataloader(batch_size=32, num_workers=1, label_name_position=1)
datamanager.train_loader.num_workers=0
datamanager.val_loader.num_workers=0
datamanager.test_loader.num_workers=0

model_args = {
    'input_shape': (2, 100, 100),
    'emb_shapes': ((25, 25), (4, 4)),
    'output_shape': (2, 100, 100),
    'fc_output_idx': [2],
    'vq_args': {'num_embeddings': 512, 'embedding_dim': 64},
    'num_class': len(datamanager.unique_labels),
    'fc_input_type': 'vqvec',
}
train_args = {
    'lr': 1e-3,
    'max_epoch': 1,
    'reducelr_patience': 3,
    'reducelr_increment': 0.1,
    'earlystop_patience': 6,
}


import torch
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print("Using device:", device)
trainer = CytoselfFullTrainer(train_args, homepath='demo_output', model_args=model_args, device=device)
trainer.fit(datamanager, tensorboard_path='tb_logs')


#plot UMAP
print("STEP 3: Plot UMAP")
from cytoself.analysis.analysis_opencell import AnalysisOpenCell

analysis = AnalysisOpenCell(datamanager, trainer)
umap_data = analysis.plot_umap_of_embedding_vector(
    data_loader=datamanager.test_loader,
    group_col=2,
    output_layer=f'{model_args["fc_input_type"]}2',
    title=f'UMAP {model_args["fc_input_type"]}2',
    xlabel='UMAP1',
    ylabel='UMAP2',
    s=0.3,
    alpha=0.5,
    show_legend=True,
)

#Plot feature spectrum

print("STEP 4: Plot feature spectrum")
# Compute bi-clustering heatmap
analysis.plot_clustermap(num_workers=4)

# Prepare image data
img = next(iter(datamanager.test_loader))['image'].detach().cpu().numpy()[:1]

# Compute index histogram
vqindhist1 = trainer.infer_embeddings(img, 'vqindhist1')

# Reorder the index histogram according to the bi-clustering heatmap
ft_spectrum = analysis.compute_feature_spectrum(vqindhist1)

# Generate a plot
import numpy as np
import matplotlib.pyplot as plt

x_max = ft_spectrum.shape[1] + 1
x_ticks = np.arange(0, x_max, 50)
fig, ax = plt.subplots(figsize=(10, 3))
ax.stairs(ft_spectrum[0], np.arange(x_max), fill=True)
ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel('Feature index')
ax.set_ylabel('Counts')
ax.set_xlim([0, x_max])
ax.set_xticks(x_ticks, analysis.feature_spectrum_indices[x_ticks])
fig.tight_layout()
fig.savefig('demo_output/analysis/feature_spectra_figures/feature_spectrum.png', dpi=300)
plt.close()