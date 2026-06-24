# %% BLACK SCHOLES BARENBLATT: No EXTRA LOSS
"""
Created on Thu Feb 27 12:07:02 2025

@author: Alberto Gámez González
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time  # Import time module

# Constants
T = 1  # Time horizon
N = 80  # Number of time steps
dt = T / N  # Time step
d = 100  # Dimension of X
sigma = 0.4  # Volatility
r = 0.05  # Interest rate
M = 100  # Number of trajectories (batch size)
BATCH_LIMIT = 2000  # Maximum epochs
ERROR_THRESHOLD = 1e-3  # Stopping criterion

# Define initial condition X_0 = (1, 1/2,  ..., 1, 1/2)
x_0 = np.tile([1, 1/2], d // 2)
X_0 = torch.tensor(x_0, dtype=torch.float32).view(1, d)

# Terminal condition (NumPy version)
g = lambda x: np.linalg.norm(x, axis=1, keepdims=True)**2  # NumPy operations

# Exact solution
def exact_solution(t, x):
    return np.exp((r + sigma**2) * (T - t)) * np.linalg.norm(x)**2  # NumPy norm


# Neural network
#n_neurons = 256  # Number of neurons in hidden layers: 2/3 inputs + number of output.
n_neurons = int(2/3*d)+1
#n_neurons=d+10

'''
class Net(nn.Module):
    def __init__(self, d):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(d + 1, n_neurons)
        self.in1 = nn.InstanceNorm1d(n_neurons)  # Instance normalization
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.in2 = nn.InstanceNorm1d(n_neurons)  # Instance normalization
        self.fc3 = nn.Linear(n_neurons, n_neurons)
        self.in3 = nn.InstanceNorm1d(n_neurons)  # Instance normalization
        self.fc4 = nn.Linear(n_neurons, 1)
        self.relu = nn.ReLU()
        
    def forward(self, t, x):
        tx = torch.cat((t, x), dim=1)
        x = self.relu(self.in1(self.fc1(tx)))  # Always works, any batch size
        x = self.relu(self.in2(self.fc2(x)))   # Same
        x = self.relu(self.in3(self.fc3(x)))   # Same
        return self.fc4(x)
'''
class Net(nn.Module):
    def __init__(self, d):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(d + 1, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, n_neurons)
        self.fc4 = nn.Linear(n_neurons, n_neurons)
        self.fc5 = nn.Linear(n_neurons, 1)
        self.relu = nn.ReLU()
    
    def forward(self, t, x):
        tx = torch.cat((t, x), dim=1)
        x = self.relu(self.fc1(tx))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.fc5(x)

# Training model
net = Net(d)

# Optimizers!!: SGD, RMSprop, Adagrad, Adadelta Adam/W and ASGD (+- Nesterov)
lr = 0.0015 # lr=0.0015 or 0.015 by default
optimizer = torch.optim.AdamW(net.parameters(), lr)  # Adam optimizer better than W in this case
#optimizer = torch.optim.ASGD(net.parameters(), lr) # Better for batch norm
#optimizer = torch.optim.SGD(net.parameters(), lr, momentum=0.9,nesterov=False) # Better for batch norm
#optimizer = torch.optim.RMSprop(net.parameters(), lr)  # Adam optimizer better than W in this case

#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.7)  # Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=200, min_lr=1e-5)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=50)
losses, Y_0_vals, Y_T_vals = [], [], []

start_time = time.time()  # Record start time

for epoch in range(BATCH_LIMIT):
    optimizer.zero_grad()
    W = torch.randn(M, N, d) * np.sqrt(dt)
    X = X_0.repeat(M, 1)  # Initial condition X_0
    Y = net(torch.zeros((M, 1)), X)
    for n in range(N - 1):
        t = (n * dt) * torch.ones((M, 1))
        Xt = X.clone().detach().requires_grad_(True)
        Yt = net(t, Xt)
        Zt = torch.autograd.grad(Yt.sum(), Xt, create_graph=True)[0]
        dX = sigma * torch.bmm(torch.diag_embed(Xt), W[:, n, :].unsqueeze(-1)).squeeze(-1)
        X = X + dX
        Y = Y + r * (Yt - (Zt * Xt).sum(dim=1, keepdim=True)) * dt + sigma * (Zt * dX).sum(dim=1, keepdim=True)

    # Final prediction
    t_final = T * torch.ones(M, 1)
    Z_T = net(t_final, X)
    # Loss function (terminal condition)
    # Convert X to NumPy, compute g(X), and convert back to tensor
    g_X = torch.norm(X, dim=1, keepdim=True) ** 2
    #loss = torch.mean((Y - g_X) ** 2 + 0.1 * torch.mean((Z_T - Zt) ** 2))
    loss = torch.mean((Y - g_X) ** 2)
    
    loss.backward()
    optimizer.step() 
    #scheduler.step(loss) # Comment this step if unused scheduler
    losses.append(loss.item())
    
    # Predictions
    Y_0_pred = net(torch.zeros((1, 1)), X_0).item()
    Y_T_pred = Yt.mean().item()
    Y_0_vals.append(Y_0_pred)
    Y_T_vals.append(Y_T_pred)
    
    # Exact solutions
    Y_0_exact = exact_solution(0, X_0) # Here was x_0 instead of X_=
    x_T = X[-1].detach().numpy()  # Use the final state X_T
    Y_T_exact = exact_solution(T, x_T)
    
    # Relative errors
    rel_error_0 = abs(Y_0_pred - Y_0_exact) / abs(Y_0_exact)
    rel_error_T = abs(Y_T_pred - Y_T_exact) / abs(Y_T_exact)
    
    # Elapsed time
    elapsed_time = time.time() - start_time
    
    
    Y_0_vals.append(Y_0_pred)
    Y_T_vals.append(Y_T_pred)
    # Print progress
    if epoch % 25 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}, Y_0 = {Y_0_pred:.6f}, Y_T = {Y_T_pred:.6f}, Elapsed Time = {elapsed_time:.2f} seconds")
        print(f"Rel Error Y_0: {rel_error_0:.6f}, Rel Error Y_T: {rel_error_T:.6f}")
        print("-" * 50)
    
    # Stopping criterion
    if loss.item() < ERROR_THRESHOLD:
        print("Stopping criterion met.")
        break


print("Training complete.")

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(losses, label='Training Loss', color='blue', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Time')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(['Y_0', 'Y_T'], [rel_error_0, rel_error_T], color=['blue', 'orange'])
plt.ylabel('Relative Error')
plt.title('Relative Error in Y_0 and Y_T')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(Y_0_vals, label='Y_0 (Predicted)', color='blue', linewidth=2)
plt.axhline(Y_0_exact, color='red', linestyle='--', linewidth=2, label='Y_0 (Exact)')
plt.plot(Y_T_vals, label='Y_T (Predicted)', color='green', linewidth=2)
plt.axhline(Y_T_exact, color='orange', linestyle='--', linewidth=2, label='Y_T (Exact)')
plt.xlabel('Epoch')
plt.ylabel('Values')
plt.legend(loc='lower right', fontsize=12)
plt.title('Y_0 and Y_T Over Training')
plt.grid(True)
plt.show()



# %% MAIN PROBLEM TFM: ELOY with No EXTRA LOSS

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time

# Constants
T = 1  # Time horizon
N = 80  # Number of time steps
dt = T / N  # Time step
d = 100  # Dimension of X
sigma_bar = 0.2  # Volatility
mu_bar = 0.02  # Drift
delta = 2 / 3  # Recovery rate
R = 0.02  # Interest rate
gamma_h = 0.2  # Upper bound for gamma
gamma_l = 0.02  # Lower bound for gamma
v_h = 50  # Upper bound for v
v_l = 70  # Lower bound for v
M = 100  # Number of trajectories (batch size)
BATCH_LIMIT = 1500  # Maximum epochs
ERROR_THRESHOLD = 0.001  # Stopping criterion

# Define initial condition X_0 = (100, ..., 100)
x_0 = np.ones(d) * 100
X_0 = torch.tensor(x_0, dtype=torch.float32).view(1, d)

# Terminal condition (updated to handle 1D and 2D inputs)
def g(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)  # Reshape to 2D if input is 1D
    return np.min(x, axis=1, keepdims=True)  # Min over all dimensions

# Monte Carlo estimate of Y_0 = E[g(X_T)]
def monte_carlo_exact_solution(num_simulations=10000):
    X_T = np.zeros((num_simulations, d))
    for i in range(num_simulations):
        W = np.random.randn(N, d) * np.sqrt(dt)
        X = x_0.copy()
        for n in range(N):
            dX = mu_bar * X * dt + sigma_bar * X * W[n, :]
            X = X + dX
        X_T[i, :] = X
    return np.mean(g(X_T))

# Non-linear term f
def f(t, x, y, z):
    gamma = torch.minimum(
        torch.tensor(gamma_h, dtype=torch.float32),
        torch.maximum(
            torch.tensor(gamma_l, dtype=torch.float32),
            (gamma_h - gamma_l) / (v_h - v_l) * (y - v_h) + gamma_h
        )
    )
    return -(1 - delta) * gamma * y - R * y

# Neural network
n_neurons = int(2/3*(d + 1))
#n_neurons = 256;

class Net(nn.Module):
    def __init__(self, d):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(d + 1, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, 1)
        self.relu = nn.ReLU()
    
    def forward(self, t, x):
        tx = torch.cat((t, x), dim=1)
        x = self.relu(self.fc1(tx))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

'''
class Net(nn.Module):
    def __init__(self, d):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(d + 1, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)  # Batch normalization after first layer
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.bn2 = nn.BatchNorm1d(n_neurons)  # Batch normalization after second layer
        self.fc3 = nn.Linear(n_neurons, n_neurons)
        self.bn3 = nn.BatchNorm1d(n_neurons)  # Batch normalization after third layer
        self.fc4 = nn.Linear(n_neurons, 1)
        self.relu = nn.ReLU()
        
    def forward(self, t, x):
        tx = torch.cat((t, x), dim=1)
        x = self.relu(self.bn1(self.fc1(tx)))  # Apply batch normalization after fc1
        x = self.relu(self.bn2(self.fc2(x)))  # Apply batch normalization after fc2
        x = self.relu(self.bn3(self.fc3(x)))  # Apply batch normalization after fc3
        return self.fc4(x)
'''

# Training model
net = Net(d)

# Optimizers
lr = 0.015
optimizer = torch.optim.Adam(net.parameters(), lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=200, min_lr=1e-3)

losses, Y_0_vals, Y_T_vals = [], [], []

# Monte Carlo estimate of Y_0 (exact solution at t=0)
Y_0_exact = monte_carlo_exact_solution()
#Y_0_exact = 57.300
Y_0_exact = torch.tensor(Y_0_exact, dtype=torch.float32, requires_grad=True)

print(f"Monte Carlo estimate of Y_0 (exact solution): {Y_0_exact:.6f}")

start_time = time.time()

for epoch in range(BATCH_LIMIT):
    optimizer .zero_grad()
    W = torch.randn(M, N, d) * np.sqrt(dt)
    X = X_0.repeat(M, 1)
    Y = net(torch.zeros((M, 1)), X)
    
    for n in range(N - 1):
        t = (n * dt) * torch.ones((M, 1))
        Xt = X.clone()
        Xt.requires_grad_(True)
        Yt = net(t, Xt)
        Zt = torch.autograd.grad(Yt.sum(), Xt, create_graph=True)[0]
        
        # Drift and diffusion terms
        dX = mu_bar * Xt * dt + sigma_bar * Xt * W[:, n, :]  # Element-wise multiplication
        X = X + dX
        
        # Update Y using the dynamics
        Y = Y + f(t, Xt, Yt, Zt) * dt + sigma_bar * (Zt * dX).sum(dim=1, keepdims=True)
    
    # Loss function (terminal condition)
    g_X = torch.tensor(g(X.detach().numpy()), dtype=torch.float32)


    # Compute predicted Y_0 (mean over batch)
    Y_0_pred_tensor = net(torch.zeros((M, 1)), X_0.repeat(M, 1))
    Y_0_pred_mean = Y_0_pred_tensor.mean()

    # Compute loss on terminal condition
    loss = torch.mean((Y - g_X) ** 2)
    

    
    
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    losses.append(loss.item())
    
    # Predictions
    Y_0_pred = net(torch.zeros((M, 1)), X).mean().item()  # Average over the batch
    Y_T_pred = Y.mean().item()
    Y_0_vals.append(Y_0_pred)
    Y_T_vals.append(Y_T_pred)
    
    # Exact solution at t=T
    X_T = X[-1].detach().numpy()  # Final state at t=T
    X_T = X_T.reshape(1, -1)  # Reshape to 2D array
    Y_T_exact = g(X_T).item()  # Exact solution at t=T
    
    # Relative errors
    rel_error_0 = abs(Y_0_pred - Y_0_exact.item()) / abs(Y_0_exact.item())
    rel_error_T = abs(Y_T_pred - Y_T_exact) / abs(Y_T_exact)
    
    # Elapsed time
    elapsed_time = time.time() - start_time
    
    # Print progress
    if epoch % 25 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}, Y_0 = {Y_0_pred:.6f}, Y_T = {Y_T_pred:.6f}, Elapsed Time = {elapsed_time:.2f} seconds")
        print(f"Rel Error Y_0: {rel_error_0:.6f}")
        print(f"Rel Error Y_T: {rel_error_T:.6f}")
        print("-" * 50)
    
    # Stopping criterion
    if rel_error_0 < ERROR_THRESHOLD and rel_error_T < ERROR_THRESHOLD:
        print("Stopping criterion met.")
        break

print("Training complete.")

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(losses, label='Training Loss', color='blue', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot relative errors
plt.figure(figsize=(8, 5))
plt.bar(['Y_0', 'Y_T'], [rel_error_0, rel_error_T], color=['blue', 'orange'])
plt.ylabel('Relative Error')
plt.title('Relative Error in Y_0 and Y_T')
plt.grid(True)
plt.show()

# Plot predicted vs exact values
plt.figure(figsize=(8, 5))
plt.plot(Y_0_vals, label='Y_0 (Predicted)', color='blue', linewidth=2)
plt.axhline(Y_0_exact.item(), color='red', linestyle='--', linewidth=2, label='Y_0 (Exact)')
plt.plot(Y_T_vals, label='Y_T (Predicted)', color='green', linewidth=2)
plt.axhline(Y_T_exact, color='orange', linestyle='--', linewidth=2, label='Y_T (Exact)')
plt.xlabel('Epoch')
plt.ylabel('Values')
plt.legend(loc='upper right', fontsize=12)
plt.title('Y_0 and Y_T Over Training')
plt.grid(True)
plt.show()


