import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.legacy.models.lstm_model import LSTMClassifier

X = np.load("data/processed/sequences.npy")
y = np.load("data/processed/sequence_labels.npy")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = LSTMClassifier()

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

torch.save(model.state_dict(), "models_lstm.pth")
print("LSTM model saved.")