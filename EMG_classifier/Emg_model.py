import torch
from torch import nn ,optim 
from torch.utils.data import TensorDataset, DataLoader
import numpy as np    
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix   
from sklearn.preprocessing import StandardScaler
import seaborn as sns 
import matplotlib.pyplot as plt  
from imblearn.over_sampling import SMOTE
class EMGClassfier(nn.Module): 
    def __init__(self,features,sequence_length,num_classes): 
        super().__init__()
       # self.flatten = nn.Flatten() 
        self.linear_relu_stack = nn.Sequential( 
            nn.Linear(sequence_length*features,64), 
            nn.BatchNorm1d(64),
            nn.ReLU(), 
            nn.Dropout(0.1),  

            nn.Linear(64,128), 
            nn.BatchNorm1d(128),
            nn.ReLU(), 
            nn.Dropout(0.1),  

            nn.Linear(128,256), 
            nn.BatchNorm1d(256),
            nn.ReLU(), 
            nn.Dropout(0.1),   

            nn.Linear(256,512), 
            nn.BatchNorm1d(512),
            nn.ReLU(), 
            nn.Dropout(0.1),  
            
            nn.Linear(512,num_classes), 
        )  
        self.deep_emg_model = nn.Sequential( 
            nn.Conv1d(sequence_length*features,64,1), 
            nn.BatchNorm1d(64), 
            nn.ReLU(), 
            nn.Dropout(0.2),  


            nn.Conv1d(64,128,1), 
            nn.BatchNorm1d(128), 
            nn.ReLU(), 
            nn.Dropout(0.2),  

            nn.Conv1d(128,256,1), 
            nn.BatchNorm1d(256), 
            nn.ReLU(), 
            nn.Dropout(0.2), 


            nn.Conv1d(256,512,1), 
            nn.BatchNorm1d(512), 
            nn.ReLU(), 
            nn.Dropout(0.2), 

            nn.Flatten(),    
            nn.Linear(512,num_classes),

        ) 
        self.lstm = nn.LSTM(features , 64, batch_first=True, bidirectional=True)    
        self.lstm2 = nn.LSTM(128 , 256, batch_first=True, bidirectional=True)  

        self.dropout=nn.Dropout(0.2)
        self.fc = nn.Linear(256 * 2, num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x,target=None):
        #x = self.flatten(x)
        lstm_out, _ = self.lstm(x) 
        lstm_out,_ = self.lstm2(lstm_out) 
        last_layer = lstm_out[: ,-1 ,:] 
        last_layer =self.dropout(last_layer) 
        logits = self.fc(last_layer)   
        if target is not None : 
            loss = self.loss(logits,target) 
            return logits,loss
        return logits 
    

scaler = StandardScaler()
data = np.loadtxt('../features1.csv',delimiter=',')   
sm = SMOTE(random_state=42)
X = data[:,:-1]  
features=X.shape[1] 
sample = 32 
labels=data[:,-1] 

device = 'cuda:0'

X=scaler.fit_transform(X)
X, labels = sm.fit_resample(X, labels) 
X_train,X_test,labels_train,labels_test=train_test_split(X,labels,test_size=0.20,random_state=42)   
X_test,X_valid,labels_test,labels_valid=train_test_split(X_test,labels_test,test_size=0.50,random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
labels_train_tensor = torch.tensor(labels_train, dtype=torch.int64, device=device)

X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32, device=device)
labels_valid_tensor = torch.tensor(labels_valid, dtype=torch.int64, device=device)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
labels_test_tensor = torch.tensor(labels_test, dtype=torch.int64, device=device) 

epochs=30
lr=1e-3

history ={ 
    'train_acc': [] ,
    'test_acc': [] , 
    'train_loss' :[], 
    'val_loss':[], 
    'val_acc':[]
}

X_train_tensor = X_train_tensor.unsqueeze(1)  #this is for the deep model
X_valid_tensor = X_valid_tensor.unsqueeze(1)
X_test_tensor = X_test_tensor.unsqueeze(1)

model = EMGClassfier(features,sample,num_classes=len(np.unique(labels))).to('cuda:0') 
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
    history['train_acc'].append(train_acc) 
    history['train_loss'].append(loss) 
    history['val_loss'].append(val_loss) 
    history['val_acc'].append(val_acc)
model.eval()   
with torch.no_grad():  
    test_logits, test_loss = model(X_test_tensor, labels_test_tensor)
    test_preds = torch.argmax(test_logits, dim=1)
    test_acc = (test_preds == labels_test_tensor).float().mean()  
y_true_tensor = torch.tensor(labels_test).to(device)
y_pred_tensor = torch.tensor(test_preds).to(device)
test_acc = (y_pred_tensor == y_true_tensor).float().mean()
print(f"Test accuracy: {test_acc.item():.4f}")    
history['test_acc'].append(test_acc) 

def save(history): 
    path=input("Please write something to save this training log to") 
    torch.save(history, f'training_history_{path}.pth')
    print(f"Training history saved as training_history_{path}.pth")  

save(history=history)
y_true = torch.tensor(labels_test).cpu().numpy()
y_pred = torch.tensor(test_preds).cpu().numpy()

labels = np.unique(np.concatenate([y_true, y_pred]))

cm = confusion_matrix(y_true,y_pred,normalize='true') 

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()   
print(model)


