import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from src.legacy.models.ann_classifier import ANNClassifier

# Load data
df = pd.read_csv("data/processed/node_features.csv")

# TEMP: create dummy labels for pipeline testing
df["label"] = (df["fraud_ratio"] > 0).astype(int)

X = df.drop(columns=[
    "account",
    "label",
    "fraud_ratio",
    "fraud_count"
]).values

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df["label"].values

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = ANNClassifier(input_dim=X.shape[1])

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    total_loss = 0

    for batch_X, batch_y in loader:
        optimizer.zero_grad()

        outputs, embeddings = model(batch_X)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss={total_loss:.4f}")
    
torch.save(model.state_dict(), "models_ann.pth")
print("ANN model saved.")