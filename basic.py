# Stochastic funcitons
# Here if we want to add some function
#%% MAIN
import numpy as np
import matplotlib.pyplot as plt

T = 1.0 # End time
N = 500 # Number of steps
dt = T / N
t = np.linspace(0, T, N)

W = np.zeros(N)
dW = np.zeros(N)

dW[0] = np.random.randn()
W[0] = dW[0]

# This defines the Wiener process, or Brownian motion
for j in range(1, N):
    dW[j] = np.sqrt(dt) * np.random.randn()
    W[j] = W[j-1] + dW[j]

plt.plot(t, W)
plt.xlabel(r'$t$')
plt.ylabel(r'$W(t)$')
plt.show()

# %%
T = 1.0 # End time
N = 500 # Number of steps
dt = T / N
t = np.linspace(0, T, N)

M = 10^3 # Will construct M paths simultaneously

dW = np.sqrt(dt) * np.random.randn(M, N) # Now construct the increments for all points and paths
W = np.cumsum(dW, 1) # Take the cumulative sum for each path separately

U = np.exp(t + 0.5*W) # Construct a function of the Brownian path
Umean = np.mean(U, 0) # Take the mean across all paths

# Compare against the expected average solution
print("Average error is {}".format(np.linalg.norm((Umean - np.exp(9*t/8)), np.inf)))

plt.plot(t, Umean, 'b-')
for i in range(5):
    plt.plot(t, U[i, :], 'r--')
plt.xlabel(r'$t$')
plt.ylabel(r'$U(t)$')
plt.legend(('Mean of all paths', 'Some individual paths'), loc = 'upper left')

plt.show()

# %% STOCHASTIC INTEGRALS
T = 1.0
N = 500
dt = T / N

dW = np.sqrt(dt) * np.random.randn(N)
W = np.cumsum(dW)

shiftedW = np.zeros(N)
shiftedW[1:] = W[:-1]

# The two different integrals. 
# The Ito integral is roughly the Riemann integral evaluated at the left edge of the subinterval
ito = np.sum(shiftedW*dW)
# The Stratonovich integral is roughly the Riemann integral evaluated at the centre of the subinterval
stratonovich = np.sum((0.5*(shiftedW+W) + 0.5*np.sqrt(dt)*np.random.randn(N))*dW)

# Note that the exact solutions are different - markedly so!
print("The Ito integral is {} with error {}".format(ito, np.abs(ito - 0.5*(W[-1]**2-T))))
print("The Stratonovich integral is {} with error {}".format(stratonovich, np.abs(stratonovich - 0.5*W[-1]**2)))


# %% euler maruyama
lam = 2.0
mu  = 1.0
X0  = 1.0
T   = 1.0

N = 2**8
dt = T / N
t = np.linspace(0, T, N)

dW = np.sqrt(dt) * np.random.randn(N)
W = np.cumsum(dW)

# The "true" (average, or expected) solution
Xtrue = X0 * np.exp((lam - 0.5*mu**2)*t + mu*W)

# Downsample the number of points.
# Note that we average the increments.
R = 4
Dt = R * dt
L = int(N / R)
tt = np.linspace(0, T, L)

# Initial data
Xem = np.zeros(L)
Xem[0] = X0

# Euler-Maruyama applied to this SDE
Xtemp = X0
for j in range(L):
    Winc = sum(dW[R*j:R*(j+1)])
    Xtemp += (Dt * lam + mu * Winc) * Xtemp
    Xem[j] = Xtemp
# Note that this is resetting, or overwriting, the initial data point.
# I find this slightly confusing.

print("Error at the endpoint is {}".format(np.abs(Xem[-1] - Xtrue[-1])))

plt.plot(t, Xtrue, 'm-')
plt.plot(tt, Xem, 'r--*')
plt.xlabel(r'$t$')
plt.ylabel(r'$X$')
plt.legend(('Expected solution', 'Realized solution'), loc = 'upper left')

plt.show()