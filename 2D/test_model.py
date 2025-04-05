import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.distributions import Normal, kl_divergence


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

class TestModel(nn.Module):

    def __init__(self, trainer):

        super().__init__()

        self.__dict__.update(vars(trainer))


    def test(self, test_loader):
        y_s = []
        y_preds = []
        y_preds_rep = []
        factor_s = []
        factor_pred = []
        self.model.eval()
        for data, y, hcr in test_loader:
            data = data.cuda()
            y = y.cuda()
            y=(y>0.5).float()
            hcr_feat = hcr.cuda()

            reconstructed_z, z, z_mean, z_log_var, y_pred  = self.model.inference(data, hcr_feat)


            joint_predictions = self.factor_classifier(z)


            product_of_marginals_predictions = self.factor_classifier(torch.cat((z[:, :self.dlr_size], torch.cat((z[1:, self.dlr_size:], z[0, self.dlr_size:][None]), dim=0)), dim=1))
            factor_pred.append(joint_predictions[:, 0].round().int())
            factor_pred.append(product_of_marginals_predictions[:, 0].round().int())
            factor_s.append(torch.ones_like(joint_predictions[:, 0]))
            factor_s.append(torch.zeros_like(joint_predictions[:, 0]))

            y_pred = self.final_classifier(z)

            y_preds.append(y_pred)
            y_s.append(y.int())

            y_preds_rep.append(y_pred[:, 0])

        y_preds = torch.cat(y_preds, dim=0).cpu().detach().numpy()
        y_s = torch.cat(y_s, dim=0).cpu().numpy()
        bacc = roc_auc_score(y_s, y_preds)
        print("TEST B-ACC : ", bacc)

        factor_pred = torch.cat(factor_pred, dim=0)
        factor_s = torch.cat(factor_s, dim=0)
        bacc = (factor_s == factor_pred).float().mean()
        print("TEST FACTOR B-ACC : ", bacc.item())





    def test_linear_probe(self, train_loader, test_loader):
        # compute the representation of the normal set
        with torch.no_grad():
            X_train = []
            X_test = []

            y_train_subtype = []
            y_test_subtype = []

            hcr_train_subtype = []
            hcr_test_subtype = []

            data_train_original = []

            target_original = []

            data_test_original = []

            for data, target, hcr in train_loader:
                data = data.cuda()

                reconstructed_z, z, z_mean, z_log_var, y_pred  = self.model.inference(data, hcr)

                X_train.extend(z.cpu().numpy())

                y_train_subtype.extend(target.cpu().numpy())

                hcr_train_subtype.extend(hcr.cpu().numpy())

                data_train_original.extend(data.cpu().numpy())

                target_original.extend(target.cpu().numpy())


            for data, target, hcr in test_loader:

                data = data.cuda()

                reconstructed_z, z, z_mean, z_log_var, y_pred  = self.model.inference(data, hcr)

                X_test.extend(z.cpu().numpy())

                y_test_subtype.extend(target.cpu().numpy())

                data_test_original.extend(data.cpu().numpy())

                hcr_test_subtype.extend(hcr.cpu().numpy())

            # Construction des arrays numpy
            X_train = np.array(X_train)
            X_test = np.array(X_test)
            y_train_subtype = np.array(y_train_subtype)
            y_test_subtype = np.array(y_test_subtype)
            data_train_original = np.array(data_train_original)
            data_test_original = np.array(data_test_original)
            target_original = np.array(target_original)

            #Réduction de dimensionss
            data_train_original = np.squeeze(data_train_original)
            data_test_original = np.squeeze(data_test_original)
            target_original = np.squeeze(target_original)
            data_train_original = data_train_original.reshape(data_train_original.shape[0],-1)
            data_test_original = data_test_original.reshape(data_test_original.shape[0],-1)


            hcr_train_subtype = np.array(hcr_train_subtype)
            hcr_test_subtype = np.array(hcr_test_subtype)

        # Standardisation
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        data_train_original = scaler.fit_transform(data_train_original)
        data_test_original = scaler.fit_transform(data_test_original)

        #applatir y
        y_train_subtype = y_train_subtype.flatten()
        y_test_subtype = y_test_subtype.flatten()

        # Compute performances---------------------



        # Regression logistique space latent
        log_reg = LogisticRegression(max_iter=100).fit(X_train, y_train_subtype)
        log_reg_score = log_reg.score(X_test, y_test_subtype)
        print("Linear probe trained space latent, latents HCR + DLR : ", log_reg_score)


        # Récupération des poids (pour la classe 1 si binaire)
        weights = log_reg.coef_[0]

        # Générer des labels pour les dimensions (HCR + DLR)
        n_features = len(weights)
        feature_names = [f"z{i+1}" for i in range(n_features)]  # z1, z2, ..., zn

        # Tracer le barplot
        plt.figure(figsize=(10, 5))
        plt.bar(range(n_features), weights, tick_label=feature_names)
        # Ajout d’un trait vertical en pointillés après les DLR
        plt.axvline(x=self.dlr_size - 0.5, color='gray', linestyle='--', linewidth=1)
        plt.xticks(rotation=45)
        plt.xlabel("Dimensions latentes")
        plt.ylabel("Poids de la régression logistique")
        plt.title("Importance des dimensions latentes (HCR + DLR)")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()


        # Regression logistique space original
        log_reg_original = LogisticRegression(max_iter=1000).fit(data_train_original, y_train_subtype)
        log_reg_score_original = log_reg_original.score(data_test_original, y_test_subtype)
        print("Linear probe trained space original, latents HCR + DLR : ", log_reg_score_original)



        # SVM
        svm_test = svm.SVC(max_iter=100).fit(X_train, y_train_subtype)
        svm_score = svm_test.score(X_test, y_test_subtype)
        print("Linear probe trained, latents HCR + DLR (SVM) : ", svm_score)

        # Random forest
        rf = RandomForestClassifier().fit(X_train, y_train_subtype)
        rf_score = rf.score(X_test, y_test_subtype)
        print("Linear probe trained, latents HCR + DLR (Random forest) : ", rf_score)

        # PCA sur l'espace latent en 2D
        pca_latent = PCA(n_components=2)
        X_train_pca = pca_latent.fit_transform(X_train)

        plt.figure()
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1],
                            c=y_train_subtype, cmap='viridis', s=50, alpha=0.7)
        ax.set_title("PCA Latent Space 2D")
        ax.set_xlabel("Composante 1")
        ax.set_ylabel("Composante 2")

        print('Variance expliquée du PCA sur les données latentes:', pca_latent.explained_variance_ratio_)

        # PCA sur les données originales en 2D
        pca_original = PCA(n_components=2)
        X_train_pca_data_original = pca_original.fit_transform(data_train_original)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X_train_pca_data_original[:, 0], X_train_pca_data_original[:, 1],
                            c=y_train_subtype, cmap='viridis', s=50, alpha=0.7)
        ax.set_title("PCA Données Originales 2D")
        ax.set_xlabel("Composante 1")
        ax.set_ylabel("Composante 2")




        print('Variance du PCA du sur les données initiales :', pca_original.explained_variance_ratio_)

        #LDA donnés original

        clf = LDA()
        images_transf = clf.fit_transform(data_train_original, target_original)
        print(images_transf.shape)

        #plot
        plt.figure(figsize=(8, 5))

        # Scatter plot des points projetés sur l'unique dimension LDA
        plt.scatter(images_transf, [0] * len(images_transf), c=target_original, cmap='viridis', alpha=0.7)

        plt.xlabel("LDA Component 1")
        plt.yticks([])  # Supprime l'axe Y (car tout est projeté sur une seule ligne)
        plt.title("Projection donnés originaux LDA - 1D")
        plt.grid()
        plt.show()

        #LDA space latent

        clf = LDA()
        images_transf_latent = clf.fit_transform(X_train, y_train_subtype)
        print(images_transf.shape)

        #plot
        plt.figure(figsize=(8, 5))

        # Scatter plot des points projetés sur l'unique dimension LDA
        plt.scatter(images_transf_latent, [0] * len(images_transf_latent), c=y_train_subtype, cmap='viridis', alpha=0.7)

        plt.xlabel("LDA Component 1")
        plt.yticks([])  # Supprime l'axe Y (car tout est projeté sur une seule ligne)
        plt.title("Projection space latent LDA - 1D")
        plt.grid()
        plt.show()








