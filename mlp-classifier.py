##Embedding avec un MLP

#Importation des librairies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt

##On va créer un MLP sous forme de classe héritant de nn.Module

class Net(torch.nn.Module):
    def __init__(self, n_in, n_h1, n_h2, n_out):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(n_in,n_h1) # couche cachée 1
        self.fc2 = torch.nn.Linear(n_h1,n_h2) # couche cachée 2
        self.fc3 = torch.nn.Linear(n_h2,n_out)  # couche de sortie
            
#Définition des fonctions d'activation
    def forward(self, X):
        A0 = X
        A1 = self.fc1(A0).clamp(0)   #fonction d'activation pour la couche cachée 1
        A2 = self.fc2(A1).clamp(0)   #fonction d'activation pour la couche cachée 2
        A3 = torch.nn.functional.sigmoid(self.fc3(A2))   #fonction d'activation de la couche de sortie
        return A3

# --- START CODE HERE
  #We initialise our network class
my_model = Net(n_in, n_h1, n_h2, n_out)
# --- END CODE HERE

