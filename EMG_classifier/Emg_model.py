import torch
from torch import nn 
import numpy as np  
import pandas as pd 
class EMGClassfier(nn.Module): 
    def __init__(self,sequence_length,features,num_classes): 
        super().__init__()
        self.flatten = nn.Flatten() 
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(sequence_length*features,128),
            nn.ReLU(),
            nn.Linear(128,num_classes),
        ) 
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x,target=None):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x) 
        if target is not None : 
            loss = self.loss(logits,target) 
            return logits,loss
        return logits 
    
data = np.loadtxt('../features.csv',delimiter=',') 
data_tensor = torch.tensor(data, dtype=torch.float32,device='cuda:0')
model = EMGClassfier(len(data),5,8).to('cuda:0')
output=model(data)  
print(output.shape)