import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from model import Seq2SeqAttention
from data_generation import generate_nonlinear_nonstationary
from sklearn.preprocessing import StandardScaler
from tqdm import trange

def create_supervised(series, input_len=60, output_len=12):
    arr = series.values.astype(float)
    X, Y = [], []
    for i in range(len(arr) - input_len - output_len + 1):
        X.append(arr[i:i+input_len])
        Y.append(arr[i+input_len:i+input_len+output_len])
    X = np.array(X)[:,:,None]
    Y = np.array(Y)[:,:,None]
    return X, Y

def train_main(epochs=10, batch_size=32, lr=1e-3, device='cpu'):
    s = generate_nonlinear_nonstationary(1500)
    X, Y = create_supervised(s, input_len=60, output_len=12)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1,1)
    X_train = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1,1)).reshape(X_val.shape)
    Y_train = scaler.transform(Y_train.reshape(-1,1)).reshape(Y_train.shape)
    Y_val = scaler.transform(Y_val.reshape(-1,1)).reshape(Y_val.shape)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train,dtype=torch.float32),
                                            torch.tensor(Y_train,dtype=torch.float32)),
                              batch_size=batch_size, shuffle=True)

    model = Seq2SeqAttention(enc_params={'input_size':1,'hidden_size':64,'num_layers':1},
                             dec_params={'hidden_size':64,'output_size':1,'num_layers':1},
                             device=device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for ep in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            preds = model(xb, target_len=yb.size(1))
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        print(f'Epoch {ep+1}/{epochs} - Train MSE: {epoch_loss/len(train_loader):.6f}')
    torch.save(model.state_dict(), 'seq2seq_attention.pth')
    print('Saved model: seq2seq_attention.pth')

if __name__ == '__main__':
    train_main(epochs=5)
