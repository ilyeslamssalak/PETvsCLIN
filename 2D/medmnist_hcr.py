import os

import torch
from torch.utils.data import Dataset

import medmnist
from medmnist import INFO 

import numpy as np
from sklearn.preprocessing import StandardScaler

import radiomics
from radiomics import  shape2D

import SimpleITK as sitk

import matplotlib.pyplot as plt

class MedMNISTwHCR(Dataset):
    def __init__(self, split='train', transform=None, size=64, hcr_folder=None , data_flag = None, download = True, save_directory = None):


        info = INFO[data_flag]


        self.dataset = getattr(medmnist, info['python_class'])(
            split=split,
            transform=transform,
            download=download,
            size=size
        )


        if hcr_folder != None :

          self.hcr_path = os.path.join(hcr_folder, split ,'hcr.npy')

        else :

          self.hcr_path = None

        if not os.path.isfile(self.hcr_path):

            print(f"Fichier {self.hcr_path} introuvable, calcul des hcr_features en cours...")

            # Calculer les features et les sauver

            all_features = []

            for i in range(len(self.dataset)):

                img, label = self.dataset[i]

                feats = self.compute_hcr_features(img[0])  # par ex. si shape = (1,H,W)

                all_features.append(feats)

            #conversion en array et réalignement
            all_features = np.array(all_features, dtype=np.float32)
            print(f"Forme avant standardisation : {all_features.shape}")

            # Suppression des dimensions inutiles
            all_features = np.squeeze(all_features)

            # Standardisation
            scaler = StandardScaler()
            all_features = scaler.fit_transform(all_features)

            #  Créer le dossier parent si nécessaire
            os.makedirs(os.path.dirname(self.hcr_path), exist_ok=True)

            np.save(self.hcr_path, all_features)



        # Ensuite, on charge (le fichier a été créé si besoin)
        self.hcr_features = np.load(self.hcr_path)


        # Sécurité : vérifier que le nombre d'images correspond
        assert len(self.dataset) == len(self.hcr_features), \
            "La taille du Dataset medmnist et le .npy des HCR ne correspondent pas."

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retourne un tuple (image, label, hcr)
        ou (image, label) si vous préférez concaténer plus tard.
        """
        img, label = self.dataset[idx]  # img = Tensor, label = Tensor
        hcr_feat = self.hcr_features[idx]        # type: np.array

        # Convertir les HCR en tensor si nécessaire
        hcr_feat = torch.from_numpy(hcr_feat).float()

        return img, label, hcr_feat




    def compute_hcr_features(self,tensor_image):
        sitk_image = sitk.GetImageFromArray(tensor_image.numpy())

        
        otsu_filter = sitk.OtsuThresholdImageFilter()
        sitk_mask = otsu_filter.Execute(sitk_image)
        

        # fo = firstorder.RadiomicsFirstOrder(sitk_image, sitk_mask)
        # fo.enableAllFeatures()
        # fo.execute()
        # features = [
        #     fo.getMeanFeatureValue(),
        #     fo.getStandardDeviationFeatureValue(),
        #     fo.getEntropyFeatureValue(),
        #     fo.getEnergyFeatureValue(),
        #     fo.getKurtosisFeatureValue(),
        #     fo.getRobustMeanAbsoluteDeviationFeatureValue(),
        #     fo.getUniformityFeatureValue(),
        #     fo.getSkewnessFeatureValue()
        # ]


        # features = np.array(features, dtype=np.float32)
        # scaler = StandardScaler()
        # features_transformed = scaler.fit_transform(features)

        # return features_transformed
        # Initialiser l'extracteur de caractéristiques de forme
        
        shape_features = radiomics.shape2D.RadiomicsShape2D(sitk_image, sitk_mask)
        
        shape_features.enableAllFeatures()
        shape_features.execute()

        # Extraire les caractéristiques de forme
        features = [
            shape_features.getMeshSurfaceFeatureValue(),
            shape_features.getPixelSurfaceFeatureValue(),
            shape_features.getPerimeterFeatureValue(),
            shape_features.getPerimeterSurfaceRatioFeatureValue(),
            shape_features.getSphericityFeatureValue(),
            shape_features.getSphericalDisproportionFeatureValue(),
            shape_features.getMaximumDiameterFeatureValue(),
            shape_features.getMajorAxisLengthFeatureValue(),
            shape_features.getMinorAxisLengthFeatureValue(),
            shape_features.getElongationFeatureValue(),
        ]

        # Convertir les caractéristiques en tableau numpy
        features = np.array(features, dtype=np.float32)

        # Standardiser les caractéristiques
        scaler = StandardScaler()
        features_transformed = scaler.fit_transform(features.reshape(-1, 1)).flatten()

        return features_transformed



