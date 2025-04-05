import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn_model import Encoder, Decoder, Classifier

class ContrastiveVAE(nn.Module):
    """ The VAE architecture.
    """

    def __init__(self, hyperparams):
        super(ContrastiveVAE, self).__init__()

        # Hand Crafted Radiomics

        #self.hcr = HCR()
        self.hcr_size =  hyperparams["hcr_size"]

        #Hyperparameters

        self.dlr_size = hyperparams["dlr_size"]
        self.total_latent_size = self.dlr_size + self.hcr_size


        self.encoder = Encoder(self.dlr_size).float()
        self.decoder = Decoder(self.dlr_size, self.hcr_size).float()
        self.classifier = Classifier(self.total_latent_size).float()

    def forward(self, x, y, hcr):

        device = x.get_device()
        batch_size = x.size()[0]

        # DLR space
        mean, logvar = self.encoder(x)
        std = (logvar * 0.5).exp()

        #DLR vector latent (z = mean +std*erreur)
        reparameterized_latent = torch.randn((batch_size, self.dlr_size), device=device)
        dlr_features = mean + std * reparameterized_latent

        #HCR space
        #Mettre dans la GPU
        hcr_features = hcr.float().to(device)
        hcr_features = hcr_features.squeeze(-1)

        #concatenation
        z = torch.cat([dlr_features,hcr_features], dim = 1)

        # Classification de l'espace latent
        y_pred = self.classifier(z)

        #Reconstruction de l'image x
        reconstructed_x = self.decoder(z)

        # Y a t il vraiment besoin de retourner mean et logvar ?
        return reconstructed_x, mean, logvar, y_pred, z

    def inference(self, x, hcr):
        device = x.get_device()

        # DLR space (pas d'échantillonnage aléatoire)
        mean, logvar = self.encoder(x)
        std = (logvar * 0.5).exp()

        #DLR vector latent (z = mean +std*erreur)
        reparameterized_latent = torch.randn((1, self.dlr_size), device=device)
        dlr_features = mean + std * reparameterized_latent  

        # HCR space
        hcr_features = hcr.float().to(device)
        hcr_features = hcr_features.squeeze(-1)


        z = torch.cat([dlr_features, hcr_features], dim=1)

        # Classification (basée sur la représentation latente)
        y_pred = self.classifier(z)

        #Reconstruction de z
        reconstructed_z = self.decoder(z)


        return reconstructed_z, z, mean, logvar, y_pred
