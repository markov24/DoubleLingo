import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score
)
from sklearn.model_selection import train_test_split

# Early stop class
class EarlyStopper:
  def __init__(self, patience=2, min_delta=0):
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.min_validation_loss = float('inf')

  def early_stop(self, validation_loss):
    if validation_loss < self.min_validation_loss:
      self.min_validation_loss = validation_loss
      self.counter = 0
    elif validation_loss > (self.min_validation_loss + self.min_delta):
      self.counter += 1
      if self.counter >= self.patience:
        return True
    return False

# Data Class
class Data(torch.utils.data.Dataset):
  def __init__(self, X, y, scale_data=False):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      self.X = torch.from_numpy(X)
      self.y = torch.from_numpy(y)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, i):
    return self.X[i], self.y[i]

# Multilayer Perceptron Class
class MLP(nn.Module):
  def __init__(self, category, norm=False, size=768):
    super(MLP, self).__init__()
    self.category = category
    self.norm = norm
    if self.norm:
      self.linear_relu = nn.Sequential(
        nn.Linear(size, size),
        nn.BatchNorm1d(size),
        nn.ReLU()
      )
    else:
      self.linear_relu = nn.Sequential(
        nn.Linear(size, size),
        nn.ReLU()
      )
    self.linear = nn.Linear(size, 1)

  def forward(self, x):
    out = self.linear_relu(x)
    y_pred = self.linear(out)
    if self.category == "C":
      return torch.sigmoid(y_pred)
    elif self.category == "R":
      return y_pred

class FeedForward():
  def __init__(self, num_epochs, batch_size, learning_rate, category, norm, size, device):
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.category = category
    self.norm = norm
    self.size = size
    self.device = torch.device(device if torch.cuda.is_available() else "cpu")
    self.model = MLP(category=self.category, norm=self.norm, size = self.size).to(self.device)
  
  def fit(self, X, y, X_val=None, y_val=None):
    # Define the loss function according to category
    if self.category == "C":
      loss_function = nn.BCELoss()
    elif self.category == "R":
      loss_function = nn.MSELoss()

    # Define early stopper if validation required
    if X_val is not None:
      stopper = EarlyStopper()

    # Define optimizer
    optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    # Run the training loop
    for epoch in range(self.num_epochs):

      # Set current loss value
      current_loss = 0.0

      # Iterate over the DataLoader for training data
      self.model.train()
      trainloader = torch.utils.data.DataLoader(Data(X, y), batch_size=self.batch_size, shuffle=True, num_workers=1, drop_last=False)
      for i, data in enumerate(trainloader, 0):

        # Get and prepare inputs
        inputs, targets = data
        inputs, targets = inputs.float().to(self.device), targets.float().to(self.device)
        targets = targets.reshape((targets.shape[0], 1))

        # Zero the gradients
        optimizer.zero_grad()

        # Perform forward pass
        outputs = self.model(inputs)
        
        # Compute loss
        loss = loss_function(outputs, targets).to(self.device)

        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()
        current_loss += loss.item()

      # Validation
      if X_val is not None:
        self.model.eval()
        inputs_val = torch.from_numpy(X_val).float().to(self.device)
        with torch.no_grad():
          outputs_val = self.model(inputs_val)

        validation_loss = loss_function(outputs_val, torch.from_numpy(y_val).float().to(self.device).reshape((-1, 1))).item()
        # if self.category == "C": 
        #   validation_acc = accuracy_score(y_val, (outputs_val >= 0.5).to(int).cpu())
        #   print(f"Epoch : {epoch+1}/{self.num_epochs} | Loss : {current_loss},{validation_loss} | Accuracy : {validation_acc}")
        # else:
        #   print(f"Epoch : {epoch+1}/{self.num_epochs} | Loss : {current_loss},{validation_loss}")

        best_epoch, lowest_loss = epoch, validation_loss

        if stopper.early_stop(validation_loss):
           break

    # Process is complete.
    # print(f'Training process has finished with {epoch+1} epochs')
    if X_val is not None: return (lowest_loss, best_epoch+1)

  def predict(self, X):
    # Put inputs on device
    inputs = torch.from_numpy(X).float().to(self.device)

    # Set the model to evaluation mode
    self.model.eval()

    # Disable gradient computation
    with torch.no_grad():
        # Forward pass to compute predictions
        predictions = self.model(inputs)

    # Move predictions back to the CPU and convert to numpy array
    predictions = predictions.cpu().numpy()
    return predictions
  
  def predict_proba(self, X):
    predictions = self.predict(X)
    difference = np.ones(predictions.shape) - predictions
    return np.concatenate((difference, predictions), axis=1)
  
def hyperparam_search(train_inds, test_inds, X, y, num_epochs, batch_size, learning_rates, category, norm, device):
  # Perform a search over provided learning rates utilizing the EarlyStopper class
  X_train, X_val, y_train, y_val = train_test_split(X[train_inds], y[train_inds], test_size=0.20, random_state=0)

  best_loss, best_epoch, best_lr = float('inf'), None, None
  for lr in learning_rates:
    model = FeedForward(num_epochs=num_epochs, batch_size=batch_size, learning_rate=lr, 
                        category=category, norm=norm, size=X.shape[1], device=device)
    loss, epoch = model.fit(X_train, y_train, X_val, y_val)
    # print(f"Learning Rate : {lr} | Loss : {loss} | Epoch : {epoch}")

    if loss < best_loss:
      best_loss, best_epoch, best_lr = loss, epoch, lr
  # print(f"BEST LR : {best_lr} | BEST Epoch : {best_epoch}")
  model = FeedForward(num_epochs=best_epoch, batch_size=batch_size, learning_rate=best_lr, 
                      category=category, norm=norm, size=X.shape[1], device=device)
  model.fit(X[train_inds], y[train_inds])
  test = model.predict(X[test_inds]).flatten()
  train = model.predict(X[train_inds]).flatten()
  return test, train
