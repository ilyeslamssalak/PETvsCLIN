import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.distributions import Normal, kl_divergence

from cnn_model import FactorClassifier, Classifier

import matplotlib.pyplot as plt

class TrainModel(nn.Module):

    def __init__(self, hyperparams, model):
        
        super().__init__() 
        self.hyperparams = hyperparams
        self.dlr_size = self.hyperparams["dlr_size"]
        self.hcr_size = self.hyperparams["hcr_size"]

        self.loss_list = []

        # define the models

        self.bce = nn.BCELoss(reduction="mean")
        
        self.factor_classifier = FactorClassifier(self.dlr_size + self.hcr_size).float().cuda()
        self.final_classifier = Classifier(self.dlr_size+self.hcr_size).float().cuda()
        self.model = model.to("cuda")


    ## create loss function
    def compute_loss(self, x_reconstructed, x, z_mean, z_log_var, y, y_pred):

        #notre y est de la forme (batch_size, 1), mais dans le code original il n'y a que un dimension
        y = y.squeeze()


        # Hyperparameters
        beta_c  = self.hyperparams["beta_c"]
        dlr_size = self.hyperparams["dlr_size"]
        alpha = self.hyperparams["alpha"]

        # Reconstruction
        reconstruction_loss = (x[y==1] - x_reconstructed[y==1]).pow(2).sum()
        reconstruction_loss += (x[y==0] - x_reconstructed[y==0]).pow(2).sum()

        # KL
        # kl_div_loss = - beta_c * 0.5 * torch.sum(1 + z_log_var[y==0, :dlr_size] - z_mean[y==0, :dlr_size].pow(2) - z_log_var[y==0, :dlr_size].exp(), dim=-1).sum()
        # kl_div_loss += - beta_c * 0.5 * torch.sum(1 + z_log_var[y==1, :dlr_size] - z_mean[y==1, :dlr_size].pow(2) - z_log_var[y==1, :dlr_size].exp(), dim=-1).sum()
        
        # Classe 0
        q0 = Normal(loc=z_mean[y==0, :dlr_size], scale=(0.5 * z_log_var[y==0, :dlr_size]).exp())
        p0 = Normal(loc=torch.zeros_like(q0.loc), scale=torch.ones_like(q0.scale))
        kl0 = kl_divergence(q0, p0).sum()

        # Classe 1
        q1 = Normal(loc=z_mean[y==1, :dlr_size], scale=(0.5 * z_log_var[y==1, :dlr_size]).exp())
        p1 = Normal(loc=torch.zeros_like(q1.loc), scale=torch.ones_like(q1.scale))
        kl1 = kl_divergence(q1, p1).sum()


        # Total KL
        kl_div_loss = beta_c * (kl0 + kl1)


        # kl_div_loss += - beta_s * 0.5 * torch.sum(1 + z_log_var[y==1, dlr_size:] - z_mean[y==1, dlr_size:].pow(2) - z_log_var[y==1, dlr_size:].exp(), dim=-1).sum()
        # kl_div_loss += beta_s * 
        kl_div_loss += alpha * 0.5 * z_mean[y==0, dlr_size:].pow(2).sum()

        clsf_loss = self.bce(y_pred, y[:, None])
        return reconstruction_loss, kl_div_loss, clsf_loss

    



    def train_step(self, optimizer, factor_optimizer, train_loader):
        #hyperparametres
        dlr_size = self.hyperparams["dlr_size"]
        kappa = self.hyperparams["kappa"]
        gamma = self.hyperparams["gamma"]
        fader_param = self.hyperparams["fader_param"]
        reconstruction_param = self.hyperparams["reconstruction_param"]

        self.model.train()
        train_loss = 0

        loss_list = []
        factor_loss_list = []
        MI_loss_list = []
        reconstruction_loss_list = []
        kl_loss_list= []
        classif_loss_list = []

        for batch_idx, (data, y, hcr_feat) in enumerate(train_loader):

            #Data et label
            data = data.cuda()
            y = y.cuda()
            y=(y>0.5).float() #shape: (32,1) au lieu de (batch_size) comme dans le code original
            hcr_feat = hcr_feat.cuda()



            # independent optimizer training
            factor_optimizer.zero_grad()
            _, _, _, _, z = self.model(data, y, hcr_feat)


            # Classification training
            y_pred = self.final_classifier(z)
            fader_clsf_loss = self.bce(y_pred, y.float())

            # Entraînement du discriminator
            joint_predictions = self.factor_classifier(z)
            product_of_marginals_predictions = self.factor_classifier(torch.cat((z[:, :dlr_size], torch.cat((z[1:, dlr_size:], z[0, dlr_size:][None]), dim=0)), dim=1))
            
            factor_input = torch.cat((joint_predictions[:, 0], product_of_marginals_predictions[:, 0]), dim=0)
            
            factor_target = torch.cat((torch.ones_like(joint_predictions[:, 0]), torch.zeros_like(product_of_marginals_predictions[:, 0])), dim=0)
            
            factor_clsf_loss = self.bce(factor_input, factor_target)

            # parameters update
            loss_step1= factor_clsf_loss + fader_param*fader_clsf_loss
            factor_loss_list.append(loss_step1.item())

            loss_step1.backward()
            factor_optimizer.step()
           

            # training of the self.model model
            optimizer.zero_grad()
            reconstructed_x, z_mean, z_log_var, y_pred, z = self.model(data, y, hcr_feat)
            reconstruction_loss, kl_div_loss, clsf_loss = self.compute_loss(x_reconstructed=reconstructed_x, x = data, z_mean=z_mean, z_log_var= z_log_var, y= y.float(), y_pred=y_pred)
            joint_predictions = self.factor_classifier(z)
            factor_clsf_loss = F.relu(torch.log(joint_predictions / (1 - joint_predictions))).sum()

            # parameters training
            loss_step2 = reconstruction_param*reconstruction_loss + kl_div_loss + kappa*factor_clsf_loss + gamma*clsf_loss
            
            loss_list.append(loss_step2.item())
            kl_loss_list.append(kl_div_loss.item())
            MI_loss_list.append(factor_clsf_loss.item()) 
            classif_loss_list.append(clsf_loss.item())
            reconstruction_loss_list.append(reconstruction_loss.item())

            loss_step2.backward()
            train_loss += loss_step2.item()
            optimizer.step()

        return loss_list,factor_loss_list, kl_loss_list, MI_loss_list, classif_loss_list, reconstruction_loss_list

    def train(self, train_loader):
        
        # optimizer = torch.optim.Adam(self.model.parameters(), lr = self.hyperparams["learning_rate"])
        # # factor_optimizer = torch.optim.Adam(list(self.factor_classifier.parameters()) + list(self.final_classifier.parameters()))
        # factor_optimizer = torch.optim.Adam(list(self.factor_classifier.parameters()), lr=self.hyperparams["factor_learning_rate"])

        loss_list = []
        factor_loss_list = []
        KL_loss_list = []
        MI_loss_list = []
        classif_loss_list = []
        reconstruction_loss_list = []
        
        for epoch in range(1,self.hyperparams["epochs"]):
            # redefine optimizers at each epoch lead to better results
            # (provably because it re-init internal states at each epoch)
               
            optimizer = torch.optim.Adam(self.model.parameters(), lr = self.hyperparams["learning_rate"])
            # factor_optimizer = torch.optim.Adam(list(self.factor_classifier.parameters()) + list(self.final_classifier.parameters()))
            factor_optimizer = torch.optim.Adam(list(self.factor_classifier.parameters()), lr=self.hyperparams["factor_learning_rate"])

            new_loss, new_factor_loss, new_kl, new_MI, new_classif,new_reconstruction  = self.train_step(optimizer, factor_optimizer, train_loader)
            
            loss_list.extend(new_loss)
            factor_loss_list.extend(new_factor_loss)
            KL_loss_list.extend(new_kl)
            MI_loss_list.extend(new_MI)
            classif_loss_list.extend(new_classif)
            reconstruction_loss_list.extend(new_reconstruction)

        
        self.loss_list = loss_list
        self.factor_loss_list = factor_loss_list
        self.KL_loss_list = KL_loss_list
        self.MI_loss_list = MI_loss_list
        self.classif_loss_list = classif_loss_list
        self.reconstruction_loss_list = reconstruction_loss_list







    