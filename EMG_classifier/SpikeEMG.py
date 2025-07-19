import torch
from torch import nn ,optim 
from torch.utils.data import TensorDataset, DataLoader
import numpy as np    
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix   
from sklearn.preprocessing import StandardScaler
import seaborn as sns 
import matplotlib.pyplot as plt   
import snntorch as snn 
from snntorch import surrogate
from imblearn.over_sampling import SMOTE   
from torchsummary import summary
from ipdb import set_trace
from spikingjelly.clock_driven import neuron, surrogate, functional, layer


device = 'cuda:0'


scaler = StandardScaler()
data = np.loadtxt('../s1/s1_feat.csv',delimiter=',')    
sm = SMOTE(random_state=42)
X = data[:, :-1]  
n_features=X.shape[1] 
labels=data[:,-1] 

Time = X.shape[0]
num_classes = len(np.unique(labels))


X=scaler.fit_transform(X)
X, labels = sm.fit_resample(X, labels)   

X = np.repeat(X[:, np.newaxis, :], Time, axis=1)  


X_train,X_test,labels_train,labels_test=train_test_split(X,labels,test_size=0.20,random_state=42)   
X_test,X_valid,labels_test,labels_valid=train_test_split(X_test,labels_test,test_size=0.50,random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
labels_train_tensor = torch.tensor(labels_train, dtype=torch.int64, device=device)

X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32, device=device)
labels_valid_tensor = torch.tensor(labels_valid, dtype=torch.int64, device=device)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
labels_test_tensor = torch.tensor(labels_test, dtype=torch.int64, device=device)   

X_train_tensor = X_train_tensor

class SimpleSNN(nn.Module):
    def __init__(self,features,num_classes):
        super().__init__()
        self.fc = nn.Sequential( 
            nn.Linear(features,64),  

            neuron.LIFNode( surrogate_function=surrogate.PiecewiseLeakyReLU(),  
            tau=2.0),
        
            nn.Linear(64,num_classes) ,

            neuron.LIFNode(    surrogate_function=surrogate.PiecewiseLeakyReLU(),  
            tau=2.0),
        
        )  
        self.loss = nn.CrossEntropyLoss()


    def forward(self, x, target=None):
        batch_size, Time, _ = x.shape
        spike_sum = torch.zeros(batch_size, self.fc[-2].out_features, device=x.device)        
    
        functional.reset_net(self) 

        for t in range(Time):
            xt = x[:, t, :]
            spk = self.fc(xt) 
            spike_sum += spk
        output = spike_sum / Time

        if target is not None:
            loss = self.loss(output, target)
            return output, loss
        else:
            return output

model = SimpleSNN(features=n_features,num_classes=num_classes).to('cuda:0') 

num_epochs = 20 
optimizer = optim.Adam(model.parameters(), lr=1e-3) 

train_dataset = TensorDataset(X_train_tensor, labels_train_tensor) 
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) 

valid_dataset=  TensorDataset(X_test_tensor, labels_test_tensor)  
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True) 


for epoch in range(num_epochs): 
    model.train()
    train_loss = 0
    train_acc = 0
    for X_batch, labels_batch in train_loader: 
        optimizer.zero_grad()
        output, loss = model(X_batch, labels_batch)   
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0) 
        preds = torch.argmax(output, dim=1)
        train_acc += (preds == labels_batch).float().mean().item() 
    avg_loss = train_loss / len(train_loader.dataset)
    avg_acc = train_acc / len(train_loader.dataset)
    
    model.eval()
    with torch.no_grad(): 
        for valid_batch,valid_labels_batch in valid_loader:
            valid_output = model(valid_batch) 
            valid_preds = torch.argmax(valid_output, dim=1)
            valid_acc = (valid_preds == valid_labels_batch).float().mean().item()

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, Valid Acc: {valid_acc:.4f}")