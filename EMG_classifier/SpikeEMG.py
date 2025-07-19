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
        self.MLP = nn.Sequential(
            nn.Linear(features, 64), 
            nn.ReLU(),
            nn.Linear(64, num_classes)
        ) 
        self.lif1 = neuron.LIFNode(surrogate_function=surrogate.PiecewiseLeakyReLU(), tau=2.0)   
        self.loss = nn.CrossEntropyLoss()


    def forward(self, x,target=None):
        batch_size, Time,features = x.shape
        spike_sum = torch.zeros(batch_size, self.MLP[-1].out_features, device=x.device)        
    
        functional.reset_net(self) 

        xt = x.view(batch_size * Time,features)  
        xt=self.MLP(xt) 
        xt = xt.view(batch_size, Time, -1)
        for t in range(Time):
            x = xt[:, t, :]
            spk = self.lif1(x) 
            spike_sum += spk
        output = spike_sum / Time

        if target is not None:
            loss = self.loss(output, target)
            return output, loss
        else:
            return output

model = SimpleSNN(features=n_features,num_classes=num_classes).to('cuda:0') 

summary(model,input_size=(X.shape[1],X.shape[2]))

num_epochs = 20 
optimizer = optim.Adam(model.parameters(), lr=1e-3) 

train_dataset = TensorDataset(X_train_tensor, labels_train_tensor) 
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) 

valid_dataset=  TensorDataset(X_valid_tensor, labels_valid_tensor)  
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True) 


test_dataset=  TensorDataset(X_test_tensor, labels_test_tensor)  
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True) 


for epoch in range(num_epochs): 
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for X_batch, labels_batch in train_loader:
        optimizer.zero_grad()
        output, loss = model(X_batch, labels_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(output, dim=1)
        correct += (preds == labels_batch).sum().item()  # count correct predictions
        total += labels_batch.size(0)                     # count total samples

    avg_loss = train_loss / total
    avg_acc = correct / total
    
    model.eval()
    valid_loss_sum = 0
    valid_acc_sum = 0
    num_batches = 0

    with torch.no_grad():
        for valid_batch, valid_labels_batch in valid_loader:
            valid_output, val_loss = model(valid_batch, valid_labels_batch)
            valid_preds = torch.argmax(valid_output, dim=1)
            
            valid_acc_batch = (valid_preds == valid_labels_batch).float().mean()
            
            valid_loss_sum += val_loss.item() * valid_batch.size(0)
            valid_acc_sum += valid_acc_batch.item() * valid_batch.size(0)
            num_batches += valid_batch.size(0)

    valid_loss = valid_loss_sum / num_batches
    valid_acc = valid_acc_sum / num_batches


    print(f"Epoch {epoch+1}/{num_epochs}, "
      f"Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, "
      f"Valid Loss: {val_loss:.4f}, Valid Acc: {valid_acc:.4f}") 
    
model.eval()   
all_preds = []
all_labels = []
all_losses = []

with torch.no_grad():
    for test_batch, test_labels_batch in test_loader:
        test_output, test_loss = model(test_batch, test_labels_batch)
        test_preds = torch.argmax(test_output, dim=1)

        all_preds.append(test_preds)
        all_labels.append(test_labels_batch)
        all_losses.append(test_loss.item() * test_batch.size(0))

y_pred_tensor = torch.cat(all_preds)
y_true_tensor = torch.cat(all_labels)
avg_test_loss = sum(all_losses) / len(test_dataset)
test_acc = (y_pred_tensor == y_true_tensor).float().mean()

print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_acc.item():.4f}")

y_true = y_true_tensor.cpu().numpy()
y_pred = y_pred_tensor.cpu().numpy()

labels = np.unique(np.concatenate([y_true, y_pred]))
cm = confusion_matrix(y_true, y_pred, normalize='true')

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
