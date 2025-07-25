import torch
from torch import nn ,optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np    
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix , f1_score 
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import seaborn as sns 
import matplotlib.pyplot as plt   
import snntorch as snn 
from snntorch import surrogate
from imblearn.over_sampling import SMOTE   
from torchsummary import summary
from ipdb import set_trace
from spikingjelly.clock_driven import neuron, surrogate, functional, layer,encoding
import time 
import json   
from torch.utils.tensorboard import SummaryWriter

device = 'cuda:0'

subject=1

num_epochs=150  

training_number = input("Enter the training number: ") 
writer =SummaryWriter(log_dir=f'logs/{training_number}')

scaler = StandardScaler()
data = np.loadtxt('s1/total_feats.csv',delimiter=',',skiprows=1)    
sm = SMOTE(random_state=42)
X = data[:, :-1]  
n_features=X.shape[1]  

labels_a = data[0:505, -1]
labels_b = data[505:1010, -1]
labels_c = data[1010:1515, -1]

# Offset labels to avoid overlapping classes
index_1 = labels_a.max()
labels_b = labels_b + index_1 + 1

index_2 = labels_b.max()
labels_c = labels_c + index_2 + 1

# Concatenate all into a single label array
labels = np.concatenate([labels_a, labels_b, labels_c])


num_classes = len(np.unique(labels))

#set_trace()

X=scaler.fit_transform(X)
X, labels = sm.fit_resample(X, labels) 

Time= X.shape[1] 

X = X[:, np.newaxis , :]  # shape: (samples, 1, features)

X_train,X_test,labels_train,labels_test=train_test_split(X,labels,test_size=0.20,random_state=42)   
X_test,X_valid,labels_test,labels_valid=train_test_split(X_test,labels_test,test_size=0.50,random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
labels_train_tensor = torch.tensor(labels_train, dtype=torch.int64, device=device)

X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32, device=device)
labels_valid_tensor = torch.tensor(labels_valid, dtype=torch.int64, device=device) 

X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
labels_test_tensor = torch.tensor(labels_test, dtype=torch.int64, device=device)    
 
class DeepNeuralNetwork(nn.Module): 
      def __init__(self,features,num_classes):
        super().__init__()  
        self.CNN = nn.Sequential(
            nn.Conv1d(features,128,kernel_size=1), 
            nn.ReLU(), 
            nn.Dropout(0.3), 

            nn.Conv1d(128,256,kernel_size=1), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Flatten(), 

            nn.Linear(256,num_classes)
        )
        self.loss = nn.CrossEntropyLoss()

      def forward(self, x):
            batch_size, Time, features = x.shape 
            x = x.permute(0, 2, 1)  
            #set_trace() 
            logits=self.CNN(x) 
            return logits  
      def compute_loss(self,logits,labels): 
        return self.loss(logits, labels) 

model = DeepNeuralNetwork(features=n_features,num_classes=num_classes).to('cuda:0') 

summary(model,input_size=(X.shape[1],Time))

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
        X_batch = torch.tensor(X_batch, dtype=torch.float32).to(device)
        labels_batch = torch.tensor(labels_batch, dtype=torch.long).to(device)

        optimizer.zero_grad()
        output = model(X_batch) 
        loss = model.compute_loss(output,labels_batch)
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
            valid_batch = torch.tensor(valid_batch, dtype=torch.float32).to(device)
            valid_labels_batch = torch.tensor(valid_labels_batch, dtype=torch.long).to(device)

            valid_output= model(valid_batch) 
            val_loss = model.compute_loss(valid_output,valid_labels_batch)
            valid_preds = torch.argmax(valid_output, dim=1)
            
            valid_acc_batch = (valid_preds == valid_labels_batch).float().mean()
            
            valid_loss_sum += val_loss.item() * valid_batch.size(0)
            valid_acc_sum += valid_acc_batch.item() * valid_batch.size(0)
            num_batches += valid_batch.size(0)

    valid_loss = valid_loss_sum / num_batches
    valid_acc = valid_acc_sum / num_batches

    writer.add_scalar('Train/Loss', avg_loss, epoch + 1)
    writer.add_scalar('Train/Accuracy', avg_acc, epoch + 1)
    writer.add_scalar('Valid/Loss', valid_loss, epoch + 1)
    writer.add_scalar('Valid/Accuracy', valid_acc, epoch + 1)



    print(f"Epoch {epoch+1}/{num_epochs}, "
      f"Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, "
      f"Valid Loss: {val_loss:.4f}, Valid Acc: {valid_acc:.4f}") 

writer.close()


torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, f'model/{subject}/model_{training_number}checkpoint.pth')



model.eval()
all_preds = []
all_labels = []
all_losses = []

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()  # start timing

with torch.no_grad():
    for test_batch, test_labels_batch in test_loader:  
        test_batch = torch.tensor(test_batch, dtype=torch.float32).to(device)
        test_labels_batch = torch.tensor(test_labels_batch, dtype=torch.long).to(device)

        test_output = model(test_batch) 
        test_loss = model.compute_loss(test_output,test_labels_batch)
        test_preds = torch.argmax(test_output, dim=1)

        all_preds.append(test_preds)
        all_labels.append(test_labels_batch)
        all_losses.append(test_loss.item() * test_batch.size(0))

end_event.record()  # end timing

# Wait for events to be recorded
torch.cuda.synchronize()

elapsed_time_ms = start_event.elapsed_time(end_event)  # milliseconds
elapsed_time_s = elapsed_time_ms / 1000  # seconds

total_samples = len(test_dataset)
total_batches = len(test_loader)

print(f"Total inference time for test set: {elapsed_time_s:.4f} seconds")
print(f"Average inference time per batch: {elapsed_time_s / total_batches:.6f} seconds")
print(f"Average inference time per sample: {elapsed_time_s / total_samples:.6f} seconds")

# Continue with your evaluation as before
y_pred_tensor = torch.cat(all_preds)
y_true_tensor = torch.cat(all_labels)
avg_test_loss = sum(all_losses) / total_samples
test_acc = (y_pred_tensor == y_true_tensor).float().mean()

print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_acc.item():.4f}") 

results = {
    "test_loss": avg_test_loss,
    "test_accuracy": test_acc.item(),
    "total_inference_time_s": elapsed_time_s,
    "avg_inference_time_per_batch_s": elapsed_time_s / total_batches,
    "avg_inference_time_per_sample_s": elapsed_time_s / total_samples
}

with open(f"logs/inference_results_{training_number}.json", "w") as f:
    json.dump(results, f, indent=4)


y_true = y_true_tensor.cpu().numpy()
y_pred = y_pred_tensor.cpu().numpy() 

f1_macro = f1_score(y_true, y_pred, average='macro')  # average over classes equally
f1_weighted = f1_score(y_true, y_pred, average='weighted')  # weighted by support

print(f"Macro F1-score: {f1_macro:.4f}")
print(f"Weighted F1-score: {f1_weighted:.4f}")

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

