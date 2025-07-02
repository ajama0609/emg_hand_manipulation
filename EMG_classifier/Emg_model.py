import torch
from torch import nn 
import numpy as np  
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
features = data.shape[1]  
sample = 1
data_tensor = torch.tensor(data, dtype=torch.float32,device='cuda:0')
model = EMGClassfier(sample,features,num_classes=8).to('cuda:0')
output=model(data_tensor)  
print(output.shape)