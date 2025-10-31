import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import json
from tqdm import tqdm

# ==============================================================================
# 1. data process and split train set
# ==============================================================================

def prepare_data(features, labels, test_size=0.2, random_state=42):

    print("--- 1. split dataset ---")
    print(f"shape of feature: {features.shape}")
    print(f"shape of label: {labels.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, 
        labels, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=labels
    )
    
    print(f"train set: {len(X_train)}")
    print(f"test set: {len(X_test)}")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor


# ==============================================================================
# 2. model
# ==============================================================================

class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# ==============================================================================
# 3. train
# ==============================================================================

def train_model(model, X_train, y_train, X_test, y_test,epochs=10, batch_size=256, learning_rate=0.0001):
    print("\n--- 3. training ---")
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"device: {device}")
    f1_list = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        f1 = evaluate_model(model, X_test, y_test)
        f1_list.append(f1)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, F1: {f1:.4f}")
    print(max(f1_list))
# ==============================================================================
# 4. evaluation
# ==============================================================================
def evaluate_model(model, X_test, y_test):
    print("\n--- 4. evaluate ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    
    with torch.no_grad():
        outputs = model(X_test.to(device))
        preds = (outputs > 0.5).float().cpu().numpy()
        labels = y_test.cpu().numpy()
        
    f1 = f1_score(labels, preds)

    return f1
# ==============================================================================
# main
# ==============================================================================

if __name__ == "__main__":
    # index file, span id to row id
    data_index_path = ''
    # test file used in eval_dataset_vllm.py
    data_test_path = ''
    # embedding file generated in eval_dataset_vllm.py
    data_embedding_path = ''

    with open(data_index_path, 'r') as r:
        data_index = json.load(r)

    with open(data_test_path, 'r') as r:
        data_test = json.load(r)

    data_ndarray = np.memmap(data_embedding_path, dtype='float16', mode='r', shape=(2209638, 4096))

    label_list = []
    numpy_list = []
    for key in tqdm(data_index.keys()):
        tmp_rowid_list = data_index[key]
        tmp_label = data_test[tmp_rowid_list[0]]['label']
        label_list.append(tmp_label)
        tmp_numpy_list = []
        for i in tmp_rowid_list:
            tmp_numpy_list.append(data_ndarray[i])
        tmp_numpy = np.array(tmp_numpy_list)
        tmp_numpy_mean = tmp_numpy.mean(axis=0)
        numpy_list.append(tmp_numpy_mean)
    FEATURES = np.array(numpy_list)
    LABELS = np.array(label_list)

    X_train, X_test, y_train, y_test = prepare_data(FEATURES, LABELS, test_size=0.8)
    
    input_dim = X_train.shape[1]
    hidden_dim = 5120
    model = SimpleClassifier(input_size=input_dim, hidden_size=hidden_dim)
    print("\n--- model ---")
    print(model)
    
    train_model(model, X_train, y_train, X_test, y_test,epochs=500, batch_size=512)


