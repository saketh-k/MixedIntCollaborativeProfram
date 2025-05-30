import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def _translate_to_tensor_terrain(terrain_str): #member
    return torch.tensor([1 if c == 'W' else 0 for c in terrain_str])
def _translate_to_tensor_assignment(assignment_str):
    return torch.tensor([int(c)-1 for c in assignment_str])

class TerrainDataset(Dataset):
    def __init__(self, in_file, soln_file):
        self.df_input = pd.read_csv(in_file, header=None)
        self.df_soln  = pd.read_csv(soln_file)
        self.terrain_data = torch.stack([_translate_to_tensor_terrain(self.df_input.transpose()[i]) for i in range(self.df_input.shape[0])])
        self.assignments = torch.stack([_translate_to_tensor_assignment(self.df_soln.transpose()[i]) for i in range(self.df_soln.shape[0])])

    def __len__(self):
        """Returns number of entries in dataset"""
        return len(self.df_input)

    def __getitem__(self,idx):
        """
        Returns following:
        - terrain
        - assignment
        """
        return self.terrain_data[idx], self.assignments[idx]

class CNNAttnNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(216,4)
        self.conv = nn.Sequential(
            nn.Conv1d(4,8,5,padding=2),
            nn.BatchNorm1d(8),
            nn.Sigmoid(),
            nn.Conv1d(8,4,3,padding=1),
            nn.BatchNorm1d(4)
        )

        self.attention = nn.MultiheadAttention(4,num_heads=4,batch_first=True)
        self.head = nn.Linear(4,3)

    def forward(self,x):
        x = self.embed(x)
        x = x.permute(0,2,1)
        x = self.conv(x)
        x = x.permute(0,2,1)
        x, _ = self.attention(x,x,x)
        return(self.head(x))

def train_model(model,train_loader,val_loader,optimizer,criterion,epochs=10):

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        for x,y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1,3), y.view(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

            optimizer.step()
            epoch_train_loss += loss.item()

        #validation
        model.eval()
        epoch_val_loss = 0
        correct=0
        total=0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                logits = model(x_val)
                loss = criterion(logits.view(-1,3), y_val.view(-1))
                epoch_val_loss += loss.item()
                preds = torch.argmax(logits,dim=-1)
                correct += (preds == y_val).sum().item
                total += y_val.numel()

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_acc = correct/total

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(),'best_model.nn')
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if epoch % 10 == 9:
            print(f"Epoch {epoch+1}/ {epochs}: ",
                  f"Train Loss: {avg_train_loss:.4f} |",
                  f"Val Loss: {avg_val_loss:.4f} |",
                  f"Val Acc: {val_acc: .7%}")
        return train_losses, val_losses

    # NN solve non linearity
if __name__ == "__main__":
    dataset = TerrainDataset("NN_input_1000_Realizations_L1.csv","NN_solution_1000_Realizations_L1.csv")
    # Create validation loader (use 20% of training data)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Initialize fresh model
    device = torch.device("cpu")
    model = CNNAttnNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()


    # Train with validation
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, optimizer, criterion, epochs=100
    )

