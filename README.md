
##  HCR + DLR Fusion for Binary Classification with MedMNIST

This project implements a **Contrastive Variational Autoencoder (cVAE)** architecture that combines two types of representations:

- **HCR** (Hand-Crafted Radiomics): extracted via PyRadiomics from 2D or 3D MedMNIST images,
    
- **DLR** (Deep-Learned Representations): extracted via a convolutional encoder.
    

---

## Project Context

This project was conducted as part of our training at **École Centrale de Nantes (ECN)** in 2025. It focuses on **learning non-redundant features from medical images**, particularly through the **fusion of handcrafted radiomics (HCR)** with **deep-learned features (DLR)**.

The goal was to evaluate whether combining both types of descriptors can improve binary classification performance on datasets from the **MedMNIST** benchmark, using a **contrastive VAE (cVAE)** framework.

Main steps include:

- extracting **analytical radiomic features (shape, intensity)** using `pyradiomics` from 2D and 3D medical images,
    
- learning deep representations using a **Variational Autoencoder**,
    
- combining both representations while minimizing their redundancy via **Mutual Information minimization**,
    
- evaluating the results with various performance metrics (AUC, reconstruction error, LDA, PCA, etc.).
    

---

## Acknowledgments

We would like to sincerely thank our two supervisors:

- **Diana Mateus**, Professor at École Centrale de Nantes and researcher at the IRCCyN Laboratory,
    
- **Oriane Thiery**, researcher at Guerbet Research,
    

for their guidance, scientific input, and availability throughout the project.

---

##  Bibliography

This project is directly inspired by several recent works on **contrastive VAEs** and **radiomics fusion** in medical imaging:

- **Vétil et al., MICCAI 2023**  
    _Non-redundant combination of hand-crafted and deep learning radiomics: Application to the early detection of pancreatic cancer_  
    → Introduces a supervised VAE combining HCR and DLR, with Mutual Information minimization to ensure non-redundancy.  
    📄 [PDF link](https://arxiv.org/abs/2308.11389)
    
- **Abid & Zou, 2019**  
    _Contrastive Variational Autoencoder Enhances Salient Features_  
    → Proposes the cVAE model to separate salient latent factors in a target dataset compared to a reference group.  
    📄 [GitHub](https://github.com/abidlabs/contrastive_vae)
    📄 [PDF link](https://arxiv.org/abs/1902.04601)
    
- **Louiset et al., ICML 2023 Workshop**  
    _SepVAE: a contrastive VAE to separate pathological patterns from healthy ones_  
    → Introduces SepVAE, a more advanced cVAE variant with saliency-focused classification loss and stronger independence constraints.  
    📄 [GitHub](https://github.com/neurospin-projects/2023_rlouiset_sepvae)


---

## Project Structure

```
├── 2D/
│   ├── medmnist_hcr.py               # HCR feature extraction (2D shape features)
│   ├── cnn_model.py                  # 2D CNN: encoder, decoder, classifier
│   ├── c_vae.py                      # Contrastive VAE architecture (2D)
│   ├── train_model.py                # Training pipeline (2D)
│   ├── test_model.py                 # Evaluation and visualizations (2D)
│   └── Architecture Use Case (2D).ipynb  # Colab / local interactive example
│
├── 3D/
│   ├── medmnist_hcr.py               # HCR feature extraction (3D shape features)
│   ├── cnn_model.py                  # 3D CNN: encoder, decoder, classifier
│   ├── c_vae.py                      # Contrastive VAE architecture (3D)
│   ├── train_model.py                # Training pipeline (3D)
│   ├── test_model.py                 # Evaluation and visualizations (3D)
│   └── Architecture Use Case (3D).ipynb  # Colab / local interactive example
│
└── README.md
```


---
## Dataset: MedMNIST (2D & 3D)

We use several datasets from **MedMNIST v2**, a lightweight and standardized benchmark for biomedical image classification. It includes 2D and 3D medical images from various modalities such as X-rays, CT, and MRI.

Depending on the version used (`2D` or `3D`), the data may consist of:

-  2D images: size `(1, 64, 64)` or `(3, 64, 64)` 
    
- Volumetric 3D images: size `(1, 64, 64, 64)`
    

MedMNIST datasets are automatically downloaded using the `medmnist` Python API.

### Examples of datasets used

| Name             | Type | Description                        | Task                  |
| ---------------- | ---- | ---------------------------------- | --------------------- |
| `PathMNIST`      | 2D   | Colon cancer histology             | Multi-class (9)       |
| `PneumoniaMNIST` | 2D   | Chest X-ray pneumonia detection    | Binary classification |
| `AdrenalMNIST3D` | 3D   | Abdominal CT scan (adrenal glands) | Binary classification |
| `OrganMNIST3D`   | 3D   | Organ segmentation from MRI        | Multi-class           |

### Custom Dataset Class

To integrate **Hand-Crafted Radiomics (HCR)** into the MedMNIST pipeline, we implemented a custom PyTorch `Dataset`:

```python
from medmnist_hcr import MedMNISTwHCR
dataset = MedMNISTwHCR(split='train', data_flag='pathmnist', hcr_folder='features/')
```

This class handles:

- downloading MedMNIST data (if not already cached),
    
- computing or loading radiomic features (shape-based descriptors via `pyradiomics`),
    
- returning each sample as a tuple `(image, label, hcr_features)`.
    
---

## Main Dependencies

- `torch`
    
- `torchvision`
    
- `medmnist`
    
- `scikit-learn`
    
- `SimpleITK`
    
- `pyradiomics`
    
- `matplotlib`
    
- `scikit-image`
    

---

**All dependencies are listed in `requirements.txt`**

---

##  Interactive Example (Colab / Notebook)

To better understand and reproduce the pipeline, an explanatory notebook is available in the `2D/` folder:

```
2D/Architecture Use Case (2D).ipynb
```

This notebook walks through:

- loading MedMNIST data,
    
- extracting HCR features,
    
- training the `ContrastiveVAE` model,
    
- evaluating and visualizing the results.
    

📌 **We recommend starting with this notebook** to replicate the experiments or adapt the pipeline to another dataset.




