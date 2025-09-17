"""
    Example non-parametric nuisance estimator using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from torch.nn.functional import sigmoid
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
import numpy as np
import random


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.clone().detach().requires_grad_(False)
        self.y = y.clone().detach().requires_grad_(False)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]





class MLPOutcome(nn.Module):
    def __init__(self, input_size, do_rate=0.2, out_size=32):
        super(MLPOutcome, self).__init__()
        mid_size = int(out_size / 2)
        self.model = nn.Sequential(
            nn.Linear(input_size, out_size),
            nn.ReLU(),
            nn.Dropout(do_rate),
            nn.Linear(out_size, mid_size),
            nn.ReLU(),
            nn.Dropout(do_rate),
            nn.Linear(mid_size, mid_size),
            nn.ReLU(),
            nn.Dropout(do_rate),
            nn.Linear(mid_size, 1)  # Output remains scalar
        )

    def forward(self, x):
        return self.model(x)


class MLPPropensity(nn.Module):
    def __init__(self, input_dim=4, do_rate = 0.1, hidden_dim=256, out_dim=4):
        super(MLPPropensity, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(do_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(do_rate),
            nn.Linear(hidden_dim // 2, out_dim)  # logits for 4 classes
        )

    def forward(self, x):
        return self.net(x)  # outputs raw logits



class PropensityEstimator:
    def __init__(self, input_dim, do_rate=0.15, hidden_dim=64, lr=1e-4, batch_size=64, out_dim=4, non_dichotomous=None):
        self.model = MLPPropensity(input_dim, do_rate, hidden_dim, out_dim)
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_losses = None
        self.val_losses = None
        self.non_dichotomous = non_dichotomous

    def fit(self, X, y, num_epochs=800, batch_size=None, id=None, random_state=42):
        if batch_size is None:
            batch_size = self.batch_size
        # Set the seed
        tgen = torch.Generator()
        tgen.manual_seed(random_state)
        torch.manual_seed(random_state)
        random.seed(random_state)
        np.random.seed(random_state)
        x_train, x_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=random_state)
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        # copy x_train to x_train_cat
        x_train_cat = x_train.clone().detach().requires_grad_(False)
        # add interaction terms to the training data
        if self.non_dichotomous is None:
            self.non_dichotomous = list(range(X.shape[1]))  # [0, 1, 2, 3]
        for i in self.non_dichotomous:  # [0, 1, 2, 3]:
            # for j in range(i, x_train.shape[1]):
            for j in range(i, i + 1):
                x_train_cat = torch.cat((x_train_cat, (x_train[:, i:i + 1] * x_train[:, j:j + 1])), dim=1)
        # x_val, y_val = torch.tensor(X[val_indices], dtype=torch.float32), torch.tensor(y[val_indices], dtype=torch.float32)
        x_val = torch.tensor(x_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)
        # copy x_val to x_val_cat
        x_val_cat = x_val.clone().detach().requires_grad_(False)
        # add interaction terms to the validation data
        # for i in range(x_val.shape[1]):
        for i in self.non_dichotomous:
            # for j in range(i, x_val.shape[1]):
            for j in range(i, i + 1):
                x_val_cat = torch.cat((x_val_cat, (x_val[:, i:i + 1] * x_val[:, j:j + 1])), dim=1)
        # Initialize early stopping parameters
        best_val_loss = float('inf')
        best_model_state = None
        patience = 100  # Number of epochs to wait for improvement
        epochs_without_improvement = 0


        # Create DataLoader for batching
        train_dataset = TabularDataset(x_train_cat, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=tgen)

        self.train_losses = []
        self.val_losses = []
        learning_rates = []
        scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.75)
            #torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5))
        for epoch in range(num_epochs):
            # if epoch % 50 == 0:
            #     print(f'Epoch [{epoch + 1}/{num_epochs}]')
            # Forward pass
            self.model.train()
            epoch_loss = 0.0
            for x_batch, y_batch in train_loader:
                # print(x_batch[:3])
                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()

                self.optimizer.step()
                epoch_loss += loss.item() * x_batch.size(0)
            avg_epoch_loss = epoch_loss / len(train_loader.dataset)
            self.train_losses.append(avg_epoch_loss)
            # Validation loop
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(x_val_cat)
                val_loss = self.criterion(val_outputs, y_val).item()

            self.val_losses.append(val_loss)
            scheduler.step()
            learning_rates.append(self.optimizer.param_groups[0]['lr'])
            # ---- Early Stopping ----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    # print(f"Early stopping at epoch {epoch + 1}")
                    break
            # if (epoch + 1) % 50 == 0:
            #     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        # print(f'Best validation loss: {best_val_loss:.4f}')
        # Load the best model state
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        # copy X_tensor to X_tensor_cat
        X_tensor_cat = X_tensor.clone().detach().requires_grad_(False)
        # add interaction terms to the test data
        # for i in range(X_tensor.shape[1]):
        for i in self.non_dichotomous:  # [0, 1, 2, 3]:
            # for j in range(i, X_tensor.shape[1]):
            for j in range(i, i + 1):
                X_tensor_cat = torch.cat((X_tensor_cat, (X_tensor[:, i:i + 1] * X_tensor[:, j:j + 1])), dim=1)
        with torch.no_grad():
            logits = self.model(X_tensor_cat)
            return torch.softmax(logits, dim=1).numpy()  # Convert to probabilities
            # return self.model(X_tensor_cat).numpy()[:, 0]


class OutcomeEstimator:
    def __init__(self, input_size, do_rate=0.02, hidden_layer=64, lr=1e-3, w_decay=0.0, batch_size=64, setting='panel', non_dichotomous=None):
        self.model = MLPOutcome(input_size, do_rate, hidden_layer)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=w_decay)
        self.batch_size = batch_size
        self.non_dichotomous = non_dichotomous
        if setting == 'panel':
            self.non_covariate_features = 2  # G and D
            self.num_epochs = 800
        else:
            self.non_covariate_features = 3 # G, D, and T
            self.num_epochs = 1400

    def fit(self, X, y, num_epochs=None, batch_size=None, random_state=42):
        if num_epochs is None:
            num_epochs = self.num_epochs
        if batch_size is None:
            batch_size = self.batch_size
        # Set the seed
        tgen = torch.Generator()
        tgen.manual_seed(random_state)
        torch.manual_seed(random_state)
        random.seed(random_state)
        np.random.seed(random_state)
        x_train, x_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=random_state)
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        # copy x_train to x_train_cat
        x_train_cat = x_train.clone().detach().requires_grad_(False)
        # add interaction terms to the training data
        if self.non_dichotomous is None:
            self.non_dichotomous = list(range(self.non_covariate_features, X.shape[1]))
        for i in self.non_dichotomous:
            # for j in range(i, x_train.shape[1]):
            for j in range(i, i + 1):
                x_train_cat = torch.cat((x_train_cat, (x_train[:, i:i + 1] * x_train[:, j:j + 1])), dim=1)
        # x_val, y_val = torch.tensor(X[val_indices], dtype=torch.float32), torch.tensor(y[val_indices], dtype=torch.float32)
        x_val = torch.tensor(x_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        # copy x_val to x_val_cat
        x_val_cat = x_val.clone().detach().requires_grad_(False)
        # add interaction terms to the validation data
        # for i in range(x_val.shape[1]):
        for i in self.non_dichotomous:  # [2, 3, 4, 5]. we exclude the first two features because they are binary
            # for j in range(i, x_val.shape[1]):
            for j in range(i, i + 1):
                x_val_cat = torch.cat((x_val_cat, (x_val[:, i:i + 1] * x_val[:, j:j + 1])), dim=1)
        # Initialize early stopping parameters
        best_val_loss = float('inf')
        best_model_state = None
        patience = 100  # Number of epochs to wait for improvement
        epochs_without_improvement = 0

        # Create DataLoader for batching
        train_dataset = TabularDataset(x_train_cat, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=tgen)

        self.train_losses = []
        self.val_losses = []
        learning_rates = []
        scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.75)
        # torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5))
        for epoch in range(num_epochs):
            # if epoch % 50 == 0:
            #     print(f'Epoch [{epoch + 1}/{num_epochs}]')
            # Forward pass
            self.model.train()
            epoch_loss = 0.0
            for x_batch, y_batch in train_loader:
                # print(x_batch[:3])
                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch.unsqueeze(1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)


                self.optimizer.step()
                epoch_loss += loss.item() * x_batch.size(0)
            avg_epoch_loss = epoch_loss / len(train_loader.dataset)
            self.train_losses.append(avg_epoch_loss)
            # Validation loop
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(x_val_cat)
                val_loss = self.criterion(val_outputs, y_val.unsqueeze(1)).item()

            self.val_losses.append(val_loss)
            scheduler.step()
            learning_rates.append(self.optimizer.param_groups[0]['lr'])
            # ---- Early Stopping ----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience and epoch > 300:
                    # print(f"Early stopping at epoch {epoch + 1}")
                    break
            # if (epoch + 1) % 50 == 0:
            #     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        # print(f'Best validation loss: {best_val_loss:.4f}')
        # Load the best model state
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        # copy X_tensor to X_tensor_cat
        X_tensor_cat = X_tensor.clone().detach().requires_grad_(True)
        # add interaction terms to the test data
        for i in self.non_dichotomous:
            # for j in range(i, X_tensor.shape[1]):
            for j in range(i, i+1):
                X_tensor_cat = torch.cat((X_tensor_cat, (X_tensor[:, i:i + 1] * X_tensor[:, j:j + 1])), dim=1)
        with torch.no_grad():
            return self.model(X_tensor_cat).numpy()[:, 0]