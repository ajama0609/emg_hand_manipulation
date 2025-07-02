import torch
from torch import nn ,optim
import numpy as np    
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix  
import seaborn as sns 
import matplotlib.pyplot as plt 
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
X = data[:,:-1]  
features=X.shape[1]
sample = 1 
labels=data[:,-1] 

device = 'cuda:0'

X_train,X_test,labels_train,labels_test=train_test_split(X,labels,test_size=0.20,random_state=42)   
X_test,X_valid,labels_test,labels_valid=train_test_split(X_test,labels_test,test_size=0.50,random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
labels_train_tensor = torch.tensor(labels_train, dtype=torch.int64, device=device)

X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32, device=device)
labels_valid_tensor = torch.tensor(labels_valid, dtype=torch.int64, device=device)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
labels_test_tensor = torch.tensor(labels_test, dtype=torch.int64, device=device) 


epochs=20
lr=1e-3

model = EMGClassfier(sample,features,num_classes=8).to('cuda:0') 
optimzer = optim.Adam(model.parameters(),lr=lr) 
for epoch in range(epochs): 
    model.train() 
    optimzer.zero_grad()     
    logits,loss = model(X_train_tensor,labels_train_tensor)   
    loss.backward() 
    optimzer.step() 

        
    # Calculate training accuracy
    preds = torch.argmax(logits, dim=1)
    train_acc = (preds == labels_train_tensor).float().mean()  

    model.eval() 
    with torch.no_grad(): 
        val_logits, val_loss = model(X_valid_tensor, labels_valid_tensor)
        val_preds = torch.argmax(val_logits, dim=1)
        val_acc = (val_preds == labels_valid_tensor).float().mean() 

    print(f"Epoch {epoch+1}/{epochs} | Train loss: {loss.item():.4f} | Train acc: {train_acc.item():.4f} | Val loss: {val_loss.item():.4f} | Val acc: {val_acc.item():.4f}")
model.eval() 
with torch.no_grad():  
    test_logits=model(X_test_tensor) 
    test_preds = torch.argmax(test_logits,dim=1) 
    test_acc = (test_preds == labels_test_tensor).float().mean() 
print(f"Test accuracy: {test_acc.item():.4f}")  
y_true = labels_valid_tensor.cpu().numpy() 
y_pred = test_preds.cpu().numpy() 

labels = np.unique(np.concatenate([y_true, y_pred]))

cm = confusion_matrix(y_true,y_pred) 

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
