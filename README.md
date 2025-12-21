# Cross-Subject ERP Classification via Multi-View Based Contrastive Learning
## Overview
MVCLDG (Multi-View Contrastive Learning Domain Generalization) is a cross-subject domain generalization model for electroencephalogram (EEG) signal classification tasks. This model significantly enhances the generalization performance of event-related potential (ERP) recognition tasks on new subjects through multi-view feature extraction and domain-invariant representation learning.

## Core Innovations
Multi-view feature fusion: Simultaneously utilizes amplitude information from raw EEG signals and phase information derived from Hilbert Transform (HT) to enhance feature discriminability

Domain-invariant representation learning: Minimizes cross-domain feature distribution differences through domain alignment loss and contrastive learning loss

Multi-view contrastive learning: Simultaneously optimizes contrastive learning on raw, HT, and fused views

## Installation Dependencies
```python
pip install torch numpy scipy scikit-learn matplotlib tqdm
```

## Usage
An example execution is as follows:

```python
from mvclg import MVCLDGModel, MVCLDGTrainer, EEGDatasetWithHT

# Initialize model
model = MVCLDGModel(input_shape=(1, 64, 256), num_classes=2, num_domains=4)

# Prepare data
dataset = EEGDatasetWithHT(data_path='path/to/data', dataset_id=1, include_ht=True)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
domain_labels = dataset.get_domain_labels()

# Create trainer and train
trainer = MVCLDGTrainer(model, device='cuda', config={'learning_rate': 1e-3, ...})
trainer.train(data_loader, domain_labels, epochs=50)
```
Parameter num_domains specifies the number of source domains for domain generalization. Parameter include_ht controls whether to include the Hilbert Transform view. The model is trained using a combination of classification loss, domain alignment loss, and contrastive learning loss, with weights controlled by tradeoff_align and tradeoff_contrast in the config dictionary.

To use the train function, the data should be provided via a DataLoader that yields batches of EEG data and corresponding labels. The EEGDatasetWithHT class handles loading and preprocessing of EEG data, including optional Hilbert Transform computation for the phase view. The domain_labels are used to identify which domain each sample belongs to, which is essential for computing the domain alignment loss.

## Citation
If you use the MVCLDG model in your research, please cite this repository:

```text
@software{mvcldg_code_2025,
  title = {MVCLDG: Multi-View Contrastive Learning Domain Generalization for EEG},
  author = {[]},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ExtenChen3/MVCLDG}}
}
```
