import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import medmnist
from medmnist import INFO
import SimpleITK as sitk
from radiomics import shape, firstorder
from skimage.filters import threshold_otsu

class MedMNISTwHCR(Dataset):
    def __init__(self, split='train', transform=None, size=64, hcr_folder=None, data_flag=None):
        info = INFO[data_flag]
        self.dataset = getattr(medmnist, info['python_class'])(
            split=split,
            transform=transform,
            download=True,
            size=size
        )

        self.hcr_path = os.path.join(hcr_folder, split, 'hcr.npy') if hcr_folder else None

        if not os.path.isfile(self.hcr_path):
            print(f"Fichier {self.hcr_path} introuvable. Calcul des HCR features en cours...")
            all_features = []
            for i in range(len(self.dataset)):
                img, label = self.dataset[i]
                feats = self.compute_hcr_features(img[0])  # (1,H,W)
                all_features.append(feats)

            all_features = np.array(all_features, dtype=np.float32)
            all_features = np.squeeze(all_features)

            scaler = StandardScaler()
            all_features = scaler.fit_transform(all_features)

            os.makedirs(os.path.dirname(self.hcr_path), exist_ok=True)
            np.save(self.hcr_path, all_features)

        self.hcr_features = np.load(self.hcr_path)
        assert len(self.dataset) == len(self.hcr_features), \
            "Nombre de HCR features ne correspond pas au nombre d'images."

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        hcr_feat = torch.from_numpy(self.hcr_features[idx]).float()
        return img, label, hcr_feat

    def compute_hcr_features(self, tensor_image):
        arr = tensor_image.numpy()
        sitk_image = sitk.GetImageFromArray(arr)

        # Créer un masque via Otsu
        otsu_threshold = threshold_otsu(arr)
        mask = (arr > otsu_threshold).astype(np.uint8)
        sitk_mask = sitk.GetImageFromArray(mask)

        # Shape features
        shape_features = shape.RadiomicsShape(sitk_image, sitk_mask)
        shape_features.enableAllFeatures()
        shape_features.execute()

        # First-order features
        fo = firstorder.RadiomicsFirstOrder(sitk_image, sitk_mask)
        fo.enableAllFeatures()
        fo.execute()

        features = [
            shape_features.getMeshVolumeFeatureValue(),
            shape_features.getVoxelVolumeFeatureValue(),
            shape_features.getSurfaceAreaFeatureValue(),
            shape_features.getSurfaceVolumeRatioFeatureValue(),
            shape_features.getSphericityFeatureValue(),
            shape_features.getMaximum3DDiameterFeatureValue(),
            shape_features.getMaximum2DDiameterSliceFeatureValue(),
            shape_features.getMaximum2DDiameterColumnFeatureValue(),
            shape_features.getMaximum2DDiameterRowFeatureValue(),
            shape_features.getMajorAxisLengthFeatureValue(),
            shape_features.getMinorAxisLengthFeatureValue(),
            shape_features.getLeastAxisLengthFeatureValue(),
            shape_features.getElongationFeatureValue(),
            shape_features.getFlatnessFeatureValue(),
            fo.getEnergyFeatureValue(),
            fo.getTotalEnergyFeatureValue(),
            fo.getEntropyFeatureValue(),
            fo.getMinimumFeatureValue(),
            fo.get10PercentileFeatureValue(),
            fo.get90PercentileFeatureValue(),
            fo.getMaximumFeatureValue(),
            fo.getMeanFeatureValue(),
            fo.getMedianFeatureValue(),
            fo.getInterquartileRangeFeatureValue(),
            fo.getRangeFeatureValue(),
            fo.getMeanAbsoluteDeviationFeatureValue(),
            fo.getRobustMeanAbsoluteDeviationFeatureValue(),
            fo.getRootMeanSquaredFeatureValue(),
            fo.getSkewnessFeatureValue(),
            fo.getKurtosisFeatureValue(),
            fo.getVarianceFeatureValue(),
            fo.getUniformityFeatureValue()
        ]

        # Conversion propre
        features = np.array([f.item() if isinstance(f, np.ndarray) else f for f in features], dtype=np.float32)
        return features

