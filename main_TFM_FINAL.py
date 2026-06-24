#%% PDE SOLVER STOCHASTIC TFM
"""
Created on Thu Feb 27 2025 

@author: Alberto Gámez González
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time  # Import time module

# ================================================================
# HAMILTON-JACOBI-BELLMAN (HJB) EQUATION
# ================================================================

# Make some changes in terminal condition function

import seaborn as sns
sns.set_theme(style="whitegrid", font_scale=1.2)



# Constants for HJB
T_HJB = 1  # Time horizon
N_HJB = 80  # Number of time steps
dt_HJB = T_HJB / N_HJB  # Time step
d_HJB = 100  # Dimension of X
sigma_HJB = np.sqrt(2)  # Volatility
M_HJB = 100  # Number of trajectories
BATCH_LIMIT_HJB = 3000  # Maximum epochs
ERROR_THRESHOLD_HJB = 1e-8  # Stopping criterion

# Define initial condition for HJB (all zeros)
#X_0_HJB = torch.zeros((1, d_HJB), dtype=torch.float32)
X_0_HJB = torch.ones((1, d_HJB), dtype=torch.float32)
# Terminal condition for HJB
g_hjb = lambda x: np.log(0.5 * (1 + np.linalg.norm(x, axis=1)**2))
#g_hjb = lambda x: -0.5 * (2 + np.linalg.norm(x, axis=1)**2)
# Exact solution for HJB (using Wiener paths from training)
def exact_solution_hjb(t, x, W_T):
    # Compute the exact solution using the Wiener paths generated during training
    x_plus_W = x + np.sqrt(2) * W_T  # Final state
    g_values = g_hjb(x_plus_W)
    expectation = np.mean(np.exp(-g_values))
    return -np.log(expectation)


# === Loss Functions ===
huber_criterion = torch.nn.SmoothL1Loss()

def log_cosh_loss(pred, target):
    return torch.mean(torch.log(torch.cosh(pred - target + 1e-12)))  # Add small epsilon for numerical stability

def quantile_loss(pred, target, quantile=0.9):
    error = target - pred
    return torch.mean(torch.max((quantile - 1) * error, quantile * error))


# Neural network
n_neurons = int(2/3*d_HJB)+1
#n_neurons = 110;
class NetHJB(nn.Module):
    def __init__(self, d):
        super(NetHJB, self).__init__()
        self.fc1 = nn.Linear(d + 1, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, n_neurons)
        self.fc4 = nn.Linear(n_neurons, 1)
        self.relu = nn.ReLU()
    
    def forward(self, t, x):
        tx = torch.cat((t, x), dim=1)
        x = self.relu(self.fc1(tx))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

# Train HJB equation model
lr = 0.015
net_hjb = NetHJB(d_HJB)
optimizer_hjb = torch.optim.Adam(net_hjb.parameters(), lr)
#scheduler_hjb = torch.optim.lr_scheduler.StepLR(optimizer_hjb, step_size=200, gamma=0.7)
scheduler_hjb = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_hjb, 'min',patience=100, min_lr=1e-5)

print("Training Hamilton-Jacobi-Bellman Equation")
losses_hjb, Y_0_vals_hjb, Y_T_vals_hjb = [], [], []

start_time_hjb = time.time()  # Start time for HJB

for epoch in range(BATCH_LIMIT_HJB):
    optimizer_hjb.zero_grad()
    W = torch.randn(M_HJB, N_HJB, d_HJB) * np.sqrt(dt_HJB)
    X = X_0_HJB.repeat(M_HJB, 1)
    Y = net_hjb(torch.zeros((M_HJB, 1)), X)
    
    for n in range(N_HJB - 1):
        t = (n * dt_HJB) * torch.ones((M_HJB, 1))
        Xt = X.clone()
        Xt.requires_grad_(True)
        Yt = net_hjb(t, Xt)
        Zt = torch.autograd.grad(Yt.sum(), Xt, create_graph=True)[0]
        dX = sigma_HJB * W[:, n, :]
        X = X + dX
        # Default Initial
        Y = Y + (torch.norm(Zt, dim=1, keepdim=True)**2 * dt_HJB + sigma_HJB * (Zt * W[:, n, :]).sum(dim=1, keepdim=True))

    # Loss function for HJB (terminal condition)
    Y_T_pred = Y
    X_np = X.detach().cpu().numpy()
    g_values = torch.tensor(g_hjb(X_np), dtype=torch.float32).unsqueeze(1)
    
    # === Choose one loss function ===
    loss = torch.mean((Y_T_pred - g_values) ** 2)               # MSE Loss
    #loss = huber_criterion(Y_T_pred, g_values)                 # Huber Loss
    #loss = log_cosh_loss(Y_T_pred, g_values)                   # Log-Cosh Loss
    #loss = quantile_loss(Y_T_pred, g_values, quantile=0.9)     # Quantile Loss


    loss.backward()
    optimizer_hjb.step()
    scheduler_hjb.step(loss)
    losses_hjb.append(loss.item())
    
    Y_0_pred = net_hjb(torch.zeros((1, 1)), X_0_HJB).item()
    Y_T_pred = Y.mean().item()
    Y_0_vals_hjb.append(Y_0_pred)
    Y_T_vals_hjb.append(Y_T_pred)
    
    # Exact solution for HJB (using Wiener paths from training)
    W_T = torch.sum(W, dim=1)  # Sum Wiener paths over time
    Y_T_exact = exact_solution_hjb(T_HJB, X_0_HJB.detach().cpu().numpy(), W_T.detach().cpu().numpy())
    
    # Relative errors
    Y_0_exact = exact_solution_hjb(0, X_0_HJB.detach().cpu().numpy(), W_T.detach().cpu().numpy())
    rel_error_0 = abs(Y_0_pred - Y_0_exact) / abs(Y_0_exact)
    rel_error_T = abs(Y_T_pred - Y_T_exact) / abs(Y_T_exact)
    
    # Elapsed time
    elapsed_time_hjb = time.time() - start_time_hjb
    
    if epoch % 25 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}, Y_0 = {Y_0_pred:.6f}, Y_T = {Y_T_pred:.6f}, Elapsed Time = {elapsed_time_hjb:.2f} seconds")
        print(f"Rel Error Y_0: {rel_error_0:.6f}, Rel Error Y_T: {rel_error_T:.6f}")
        print("-" * 50)
    
    # Stopping criterion
    if loss.item() < ERROR_THRESHOLD_HJB:
        print("Stopping criterion met.")
        break

print("Training complete for HJB.")
# ======= Final Summary for HJB =======
final_loss_hjb = losses_hjb[-1]
final_Y0_hjb = Y_0_vals_hjb[-1]
final_YT_hjb = Y_T_vals_hjb[-1]
total_time_hjb = time.time() - start_time_hjb

print("\n" + "="*60)
print("Final Training Summary: Hamilton-Jacobi-Bellman Equation")
print("="*60)
print(f"Final Loss:            {final_loss_hjb:.6f}")
print(f"Predicted Y_0:         {final_Y0_hjb:.6f}")
print(f"Exact Y_0:             {Y_0_exact:.6f}")
print(f"Relative Error Y_0:    {rel_error_0:.6f}")
print("-" * 40)
print(f"Predicted Y_T:         {final_YT_hjb:.6f}")
print(f"Exact Y_T:             {Y_T_exact:.6f}")
print(f"Relative Error Y_T:    {rel_error_T:.6f}")
print("-" * 40)
print(f"Total Training Time:   {total_time_hjb:.2f} seconds")
print("="*60)

# === HJB: Seaborn Visualization ===
plt.figure(figsize=(10, 6))
sns.lineplot(x=range(len(losses_hjb)), y=losses_hjb, label='Hamilton-Jacobi-Bellman Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Hamilton Jacobi Bellman Loss Over Time')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(x=range(len(Y_0_vals_hjb)), y=Y_0_vals_hjb, label='Y_0 (Predicted)', color='blue', linewidth=2)
#sns.lineplot(x=range(len(Y_T_vals_hjb)), y=Y_T_vals_hjb, label='Y_T (Predicted)', color='green', linewidth=2)
#plt.axhline(Y_0_exact, color='red', linestyle='--', linewidth=2, label='Terminal Value Y0')
plt.axhline(Y_T_exact, color='orange', linestyle='--', linewidth=2, label='Terminal Value Y0')
plt.xlabel('Epoch')
plt.ylabel('Value')
#plt.axis([0,epoch,Y_0_exact-1/2,Y_0_exact+1/2])
plt.title('Y_0 and Y_T Over Training (Hamilton Jacobi Bellman)')
plt.legend()
plt.tight_layout()
plt.show()

#%% ALLEN-CAHN EQUATION SOLVER
"""
Allen-Cahn equation PDE
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time

import seaborn as sns
sns.set_theme(style="whitegrid", font_scale=1.2)
# ==================================================================
# ALLEN-CAHN EQUATION
# ==================================================================

# Constants for Allen-Cahn
T_AC = 1 # Time horizon
N_AC = 80  # Number of time steps
dt_AC = T_AC / N_AC  # Time step
d_AC = 20  # Dimension of X (as specified in the problem)
sigma_AC = 1.0  # Volatility (since dX_t = dW_t)
M_AC = 100  # Number of trajectories
BATCH_LIMIT_AC = 3000  # Maximum epochs
ERROR_THRESHOLD_AC = 1e-8  # Stopping criterion

# Define initial condition for Allen-Cahn (all zeros)
X_0_AC = torch.zeros((1, d_AC), dtype=torch.float32)

# Terminal condition for Allen-Cahn
g_ac = lambda x: 1 / (2 + 0.4 * np.linalg.norm(x, axis=1)**2)

# Neural network for Allen-Cahn
n_neurons_ac = int(2/3 * d_AC) + 1  # Slightly larger network


reference_value = 0.30879  # From paper (valid only for d=20)

# === Loss Functions ===
huber_criterion = torch.nn.SmoothL1Loss()

def log_cosh_loss(pred, target):
    return torch.mean(torch.log(torch.cosh(pred - target + 1e-12)))  # Add small epsilon for numerical stability

def quantile_loss(pred, target, quantile=0.9):
    error = target - pred
    return torch.mean(torch.max((quantile - 1) * error, quantile * error))


class NetAC(nn.Module):
    def __init__(self, d):
        super(NetAC, self).__init__()
        self.fc1 = nn.Linear(d + 1, n_neurons_ac)
        self.fc2 = nn.Linear(n_neurons_ac, n_neurons_ac)
        self.fc3 = nn.Linear(n_neurons_ac, n_neurons_ac)
        self.fc4 = nn.Linear(n_neurons_ac, 1)
        self.relu = nn.ReLU()
    
    def forward(self, t, x):
        tx = torch.cat((t, x), dim=1)
        x = self.relu(self.fc1(tx))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

# Train Allen-Cahn equation model
lr_ac = 0.015
net_ac = NetAC(d_AC)
optimizer_ac = torch.optim.AdamW(net_ac.parameters(), lr_ac)
#scheduler_ac = torch.optim.lr_scheduler.StepLR(optimizer_ac, step_size=100, gamma=0.95)
scheduler_ac = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ac, 'min',patience=300, min_lr=1e-4) # Patience
print("Training Allen-Cahn Equation")
losses_ac, Y_0_vals_ac, Y_T_vals_ac = [], [], []

start_time_ac = time.time()

for epoch in range(BATCH_LIMIT_AC):
    optimizer_ac.zero_grad()
    W = torch.randn(M_AC, N_AC, d_AC) * np.sqrt(dt_AC)
    X = X_0_AC.repeat(M_AC, 1)
    Y = net_ac(torch.zeros((M_AC, 1)), X)
    
    for n in range(N_AC - 1):
        t = (n * dt_AC) * torch.ones((M_AC, 1))
        Xt = X.clone()
        Xt.requires_grad_(True)
        Yt = net_ac(t, Xt)
        Zt = torch.autograd.grad(Yt.sum(), Xt, create_graph=True)[0]
        
        # Allen-Cahn dynamics: dY_t = (-Y_t + Y_t^3)dt + Z_t'dW_t
        dY = (-Yt + Yt**3) * dt_AC + (Zt * W[:, n, :]).sum(dim=1, keepdim=True)
        
        # Update X and Y
        
        X = X + W[:, n, :]  # Since dX_t = dW_t
        Y = Y + dY
    
    # Loss function for Allen-Cahn (terminal condition)
    Y_T_pred = Y
    X_np = X.detach().cpu().numpy()
    g_values = torch.tensor(g_ac(X_np), dtype=torch.float32).unsqueeze(1)
    
    # === Choose one loss function ===
    loss = torch.mean((Y_T_pred - g_values) ** 2)               # MSE Loss
    #loss = huber_criterion(Y_T_pred, g_values)                 # Huber Loss
    #loss = log_cosh_loss(Y_T_pred, g_values)                   # Log-Cosh Loss
    #loss = quantile_loss(Y_T_pred, g_values, quantile=0.9)     # Quantile Loss

    loss.backward()
    optimizer_ac.step()
    scheduler_ac.step(loss)
    losses_ac.append(loss.item())
    
    # Track Y_0 and Y_T values
    Y_0_pred = net_ac(torch.zeros((1, 1)), X_0_AC).item()
    Y_T_pred = Y.mean().item()
    Y_0_vals_ac.append(Y_0_pred)
    Y_T_vals_ac.append(Y_T_pred)
    
    # Compute terminal condition values for reference
    terminal_values = g_ac(X_np).mean()
    
    # Relative error for terminal condition
    rel_error_T = abs(Y_T_pred - terminal_values) / abs(terminal_values)
    
    elapsed_time_ac = time.time() - start_time_ac
    
    if epoch % 25 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}, Y_0 = {Y_0_pred:.6f}, Y_T = {Y_T_pred:.6f}")
        print(f"Terminal Value = {terminal_values:.6f}, Rel Error Y_T = {rel_error_T:.6f}")
        print(f"Elapsed Time = {elapsed_time_ac:.2f} seconds")
        print("-" * 50)
    
    
    # Stopping criterion
    if loss.item() < ERROR_THRESHOLD_AC:
        print("Stopping criterion met.")
        break

print("Training complete for Allen-Cahn.")
# Final Summary for Allen-Cahn
final_loss = losses_ac[-1]
final_Y0 = Y_0_vals_ac[-1]
final_YT = Y_T_vals_ac[-1]
final_rel_error_T = abs(final_YT - terminal_values) / abs(terminal_values)
total_time_ac = time.time() - start_time_ac
rel_error_Y0_reference = abs(final_Y0 - reference_value) / abs(reference_value)


print("\n" + "="*60)
print("Final Training Summary: Allen-Cahn Equation")
print("="*60)
print(f"Final Loss:           {final_loss:.6f}")
print(f"Predicted Y_0:        {final_Y0:.6f}")
print(f"Predicted Y_T:        {final_YT:.6f}")
print(f"Mean Terminal Value:  {terminal_values:.6f}")
print(f"Relative Error (Y_T): {final_rel_error_T:.6f}")
print(f"Relative Error Reference (Y_0): {rel_error_Y0_reference:.6f}")
print(f"Total Training Time:  {total_time_ac:.2f} seconds")
print("="*60)


# === Allen-Cahn: Seaborn Visualization ===
plt.figure(figsize=(10, 6))
sns.lineplot(x=range(len(losses_ac)), y=losses_ac, label='Allen-Cahn Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Allen-Cahn Loss Over Time')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(x=range(len(Y_0_vals_ac)), y=Y_0_vals_ac, label='Y_0 (Predicted)', color='blue', linewidth=2)
sns.lineplot(x=range(len(Y_T_vals_ac)), y=Y_T_vals_ac, label='Y_T (Predicted)', color='green', linewidth=2)
plt.axhline(terminal_values, color='orange', linestyle='--', linewidth=2, label='Terminal Value')
plt.xlabel('Epoch')
plt.ylabel('Value')
#plt.axis([0,epoch,-0.05,0.1])
plt.title('Y_0 and Y_T Over Training (Allen-Cahn)')
plt.legend()
plt.tight_layout()
plt.show()

# %% Merton Problem Solver 
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

sns.set_theme(style="whitegrid", font_scale=1.2)

# ======= Problem Parameters =======
d = 1        # Number of assets
T = 1.0          # Time horizon
N = 80          # Time steps
dt = T / N
x_0 = torch.ones(d)
mu_bar = 0.02 # Original 0.02
sigma_bar = 0.4 # Original 0.4
gamma = 0.4      # Risk aversion
eta = 0.5    
mu=0.6          # Risk premium norm
M=200           # Number trajectories
# ======= Generator and Terminal Conditions =======
def f(t, x, y, z):
    return -0.5 * torch.sum((mu**2) * (z**2), dim=1) / gamma


def utility(x):
    return -torch.exp(-eta * x)

def u_exact(t, x):
    time_factor = torch.exp(torch.tensor(-(T - t) * mu**2 / 2))
    return time_factor * utility(torch.norm(x, dim=-1))

# ======= Monte Carlo Reference for Y0 (Optional) =======
def monte_carlo_exact_solution(num_simulations=1000):
    X_T = np.zeros((num_simulations, d))
    for i in range(num_simulations):
        W = np.random.randn(N, d) * np.sqrt(dt)
        X = np.ones(d)
        for n in range(N):
            dX = mu_bar * X * dt +  sigma_bar*X * W[n, :]
            X += dX
        X_T[i, :] = X
    return torch.mean(utility(torch.mean(torch.tensor(X_T))))



# LOSES FUNCTIONS
huber_criterion = torch.nn.SmoothL1Loss()

def log_cosh_loss(pred, target):
    return torch.mean(torch.log(torch.cosh(pred - target + 1e-12)))  # Add small epsilon for numerical stability


def quantile_loss(pred, target, quantile=0.9):
    error = target - pred
    return torch.mean(torch.max((quantile - 1) * error, quantile * error))


# ======= Network =======
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# ======= Initialize Y0 =======
use_monte_carlo_Y0 = True
Y0_reference = torch.tensor(2.0)

exact_Y0 = u_exact(0,x_0)
#exact_Y0 = monte_carlo_exact_solution()
exact_Y0 = exact_Y0.item()

Y0_reference = Y0_reference.item()
Y0_init = monte_carlo_exact_solution() if use_monte_carlo_Y0 else Y0_reference
Y0 = nn.Parameter(torch.tensor([Y0_init], dtype=torch.float32, requires_grad=True))

# ======= Solver Setup =======
lr = 0.015
epochs = 3000
n_neurons = int(2/3 * d) + 1
#n_neurons = 128;
Z_net = FeedForwardNN(d, n_neurons, d)
optimizer = torch.optim.AdamW(list(Z_net.parameters()) + [Y0], lr)
# Optional Scheduler: min_lr=10e-3 or e-4. patience = 10
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10,min_lr=10**-4)


# ======= Training Loop =======
error_metrics = {
    'Loss': [], 'Y0_pred': [], 'Y0_exact': [], 'Y0_rel_error': [],
    'MSE': [], 'MAE': [], 'LearningRate': [], 'Time': [], 'TotalTime': []
}

start_total = time.time()

for epoch in range(epochs):
    start_epoch = time.time()

    X = x_0.clone()
    Y = Y0
    Z = Z_net(X)

    for n in range(M):
        dW = torch.randn(d) * np.sqrt(dt)
        f_val = f(n * dt, X.unsqueeze(0), Y.unsqueeze(0), Z.unsqueeze(0))
        Y = Y - f_val.squeeze() * dt + torch.sum(Z * dW)
        X = X + mu * dt + dW 
        Z = Z_net(X)

    #Y_terminal = u_exact(T, X)  # Correct terminal condition
    Y_terminal = utility(X)
    loss = (Y - Y_terminal) ** 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    # ======= Tracking =======
    with torch.no_grad():
        mse = (Y - Y_terminal).pow(2).mean().item()
        mae = (Y - Y_terminal).abs().mean().item()
        rel_error = abs(Y0.item() - exact_Y0) / abs(exact_Y0)

    error_metrics['Loss'].append(loss.item())
    error_metrics['Y0_pred'].append(Y0.item())
    error_metrics['Y0_exact'].append(exact_Y0)
    error_metrics['Y0_rel_error'].append(rel_error)
    error_metrics['MSE'].append(mse)
    error_metrics['MAE'].append(mae)
    error_metrics['LearningRate'].append(optimizer.param_groups[0]['lr'])
    error_metrics['Time'].append(time.time() - start_epoch)
    error_metrics['TotalTime'].append(time.time() - start_total)

    if epoch % 10 == 0:
        print(f"\nEpoch {epoch}:")
        print(f"  Loss = {loss.item():.6f}")
        print(f"  Y0 = {Y0.item():.6f} (Exact = {exact_Y0:.6f})")
        print(f"  Y0 Relative Error = {rel_error:.6f}")
        print(f"  MSE = {mse:.6f}, MAE = {mae:.6f}")
        print(f"  LR = {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Time = {error_metrics['Time'][-1]:.2f}s, Total = {error_metrics['TotalTime'][-1]:.2f}s")
        print("-" * 60)

# ======= Final Summary =======
print("\nTraining Summary:")
print(f"Final Y0 Prediction: {error_metrics['Y0_pred'][-1]:.6f}")
print(f"Final Y0 Exact: {error_metrics['Y0_exact'][-1]:.6f}")
print(f"Final Y0 Relative Error: {error_metrics['Y0_rel_error'][-1]:.6f}")
print(f"Final Loss: {error_metrics['Loss'][-1]:.6f}")
print(f"Final MSE: {error_metrics['MSE'][-1]:.6f}")
print(f"Final MAE: {error_metrics['MAE'][-1]:.6f}")
print(f"Total Training Time: {error_metrics['TotalTime'][-1]:.2f} seconds")

# ======= Visualization with Seaborn =======
metrics = ['Y0_pred', 'Y0_rel_error', 'Loss', 'MSE', 'MAE', 'LearningRate']
titles = ['Y0 Convergence', 'Y0 Relative Error', 'Training Loss',
          'Mean Squared Error', 'Mean Absolute Error', 'Learning Rate']
y_labels = ['Y0', 'Relative Error', 'Loss', 'MSE', 'MAE', 'Learning Rate']

for metric, title, y_label in zip(metrics, titles, y_labels):
    y_vals = [float(v) for v in error_metrics[metric]]
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(len(y_vals)), y=y_vals, label=metric, linewidth=2)
    # Uncomment this for d=1 and comment for d not equal t 1!!
    if metric == 'Y0_pred':
        plt.axhline(y=exact_Y0, linestyle='--', label='Exact Y0', color='orange')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    if metric in ['Y0_rel_error', 'Loss', 'MSE', 'MAE']:
        plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()
