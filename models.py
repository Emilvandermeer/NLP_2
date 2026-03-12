from __future__ import annotations
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class LSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool = False,
    ) -> None:

        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            bidirectional=bidirectional, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layer
        fc_input_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_input_dim, output_dim)
    
    def forward(self, input_ids, attention_mask=None):
        
        # Pass tokens through the embedding layer
        embedded = self.dropout(self.embedding(input_ids))
        
        # Pass embeddings through the LSTM
        _, (hidden, _) = self.lstm(embedded)
        
        # Extract the final hidden state
        if self.lstm.bidirectional:
            # Concatenate the final forward and backward hidden states
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
            
        # Pass the hidden state through the linear layer
        return self.fc(hidden)

class CNNModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_filters: int = 100,
        kernel_sizes: tuple = (3, 4, 5),
        dropout: float = 0.5,
        pad_idx: int = 0,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
             for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        emb = self.embedding(x) 

        emb_t = emb.transpose(1, 2)
        pooled = []
        for conv_block in self.convs:
            z = conv_block(emb_t) 
            # 3. Global Max Pooling
            p = torch.max(z, dim=2).values 
            pooled.append(p)

        rep = torch.cat(pooled, dim=1)  
        rep = self.dropout(rep)
        return self.fc(rep)  

def train_model(
        model: LSTMModel | CNNModel,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        epochs: int = 15,
        lr: float = 1e-3,
        patience: int = 3,
        clip_grad: float | None = None,
    ) -> tuple[LSTMModel, dict]:

    # Try to train on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model = model.to(device)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val = float('inf')
    bad_epochs = 0
    best_weights = None

    history = {'train_loss': [], 'dev_loss': []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for input_ids, _, labels in train_loader:
            # Move the data to the correct device
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids) 
            loss = loss_function(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            
            # Update weights
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        model.eval()
        total_dev_loss = 0
        
        with torch.no_grad():
            for input_ids, _, labels in dev_loader:
                # Move the data to the correct device
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                
                outputs = model(input_ids)
                loss = loss_function(outputs, labels)
                total_dev_loss += loss.item()
                
        avg_dev_loss = total_dev_loss / len(dev_loader)
        
        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Dev Loss: {avg_dev_loss:.4f}")
        
        history['train_loss'].append(avg_train_loss)
        history['dev_loss'].append(avg_dev_loss)
        
        # Early stopping
        if avg_dev_loss < best_val:
            # Save a copy if the model is better
            best_val = avg_dev_loss
            bad_epochs = 0
            best_weights = deepcopy(model.state_dict())
        else:
            # The model didn't improve
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"\n Early stopping triggered at epoch {epoch}!")
                break
                
    # Restore the weights
    if best_weights is not None:
        print(f"\nRestoring best model weights (Dev Loss: {best_val:.4f})")
        model.load_state_dict(best_weights)
        
    return model, history

def get_predictions(model: LSTMModel | CNNModel, data_loader: DataLoader) -> list:
    """Returns predictions so they can be passed in the calculate metrics function"""
    # Try to test on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    all_preds = []
    
    with torch.no_grad():
        for input_ids, _, _ in data_loader:
            input_ids = input_ids.to(device)
            
            # Get raw score
            logits = model(input_ids)
            
            # Pick the highest score
            preds = torch.argmax(logits, dim=1)
            
            # Convert to a Python list
            all_preds.extend(preds.cpu().numpy().tolist())
            
    return all_preds