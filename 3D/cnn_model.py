import torch
import torch.nn as nn
import torch.nn.functional as F

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Encoder(nn.Module):
    def __init__(self, dlr_size):
        super(Encoder, self).__init__()

        hidden_dim = 256

        self.encoder = nn.Sequential(
        nn.Conv3d(1, 32, 4, 2, 1),
        nn.BatchNorm3d(32),  # Normalisation pour éviter de Nan dans la sortie "h" de "forward"
        nn.ReLU(True),
        nn.Conv3d(32, 32, 4, 2, 1),
        nn.BatchNorm3d(32),
        nn.ReLU(True),
        nn.Conv3d(32, 64, 4, 2, 1),
        nn.BatchNorm3d(64),
        nn.ReLU(True),
        nn.Conv3d(64, 128, 4, 2, 1),
        nn.BatchNorm3d(128),
        nn.ReLU(True),
        nn.Conv3d(128, hidden_dim, 4, 1),
        nn.BatchNorm3d(hidden_dim),
        nn.ReLU(True),
        View((-1, hidden_dim * 1 * 1 * 1)),
        nn.Linear(hidden_dim, dlr_size * 2),
        )


        self.mean = nn.Linear(dlr_size * 2, dlr_size)
        self.varlog = nn.Linear(dlr_size * 2, dlr_size)

    def forward(self, x):
        if torch.isnan(x).any():
          print("NaN detected in input data!")
        h = F.relu(self.encoder(x))
        return self.mean(h), self.varlog(h)


class Decoder(nn.Module):
    def __init__(self, dlr_size, hdr_size):
        super(Decoder, self).__init__()

        hidden_dim = 256
        self.decoder = nn.Sequential(
            nn.Linear(dlr_size + hdr_size, hidden_dim),
            View((-1, hidden_dim, 1, 1, 1)),  # Ajuster les dimensions pour 3D
            nn.ReLU(True),
            nn.ConvTranspose3d(hidden_dim, 128, 4),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.ConvTranspose3d(128, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.ConvTranspose3d(64, 32, 4, 2, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 32, 4, 2, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)


class Classifier(nn.Module):
    def __init__(self, latent_dim):
        super(Classifier, self).__init__()

        self.z_dim = latent_dim

        self.fc = nn.Sequential(nn.Linear(self.z_dim, self.z_dim),
                                nn.ReLU(),
                                nn.Linear(self.z_dim, 1))

    def forward(self, latent):
        latent = latent.view(-1, self.z_dim)
        h = self.fc(latent)
        pred = torch.sigmoid(h)
        return pred


  



class FactorClassifier(nn.Module):
    def __init__(self, latent_dim):
        super(FactorClassifier, self).__init__()
        self.z_dim = latent_dim
        self.fc = nn.Sequential(nn.Linear(self.z_dim, self.z_dim), nn.ReLU(), nn.Linear(self.z_dim, self.z_dim), nn.ReLU(), nn.Linear(self.z_dim, 1))

    def forward(self, latent):

        latent = latent.view(-1, self.z_dim)
        h = self.fc(latent)
        pred = torch.sigmoid(h)
        return pred
        
