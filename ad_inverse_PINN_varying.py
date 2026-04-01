# import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from time import time
import matplotlib.pyplot as plt
import os

PHASE1_EPOCHS = 2000   # freeze u/D, fit c to data
PHASE2_EPOCHS = 5000   # unfreeze u/D, joint training
TOTAL_EPOCHS  = PHASE1_EPOCHS + PHASE2_EPOCHS

# Phase 1 weights
weights_phase1 = [0.0,   # pde   ← off
                  50.0,  # data  ← dominant
                  5.0,   # ic
                  5.0,   # bc
                  0.0]   # reg   ← off

# Phase 2 weights: full physics
weights_phase2 = [1.0,   # pde
                  50.0,  # data
                  1.0,   # ic
                  1.0,   # bc
                  1e-4]  # reg

# Input parameters
tol = 1e-8 # Tolerance for convergence

# Data
test_case = "advection_diffusion_varying_profiles" # testcase
data_file = "data/" + test_case
coarsen_data = 1 # Coarsen data by this factor
data_perturbation = 0.e-2 # Perturb data 

# Training parameters
train_parameters = True
nparm = 0 # Number of parameters to train (now using networks instead)
param_perturbation = 1.2 # Perturb parameters by this factor
learning_rate_param = 1e-3 # Learning rate for the parameter networks (reduced for stability)
train_parameters_epoch = 1000 # Number of epochs to train the parameters

# Parameter network architecture
param_network_layers = 3  # Number of hidden layers for u(x) and D(x) networks
param_network_neurons = 50  # Neurons per layer
param_activation = 'tanh'  # Activation for parameter networks

# Loss function weigths
pde_weight = 1.      # penalty for the PDE
data_weight = 50.     # penalty for the data fitting (will be multiplied by param_data_factor)
ic_weight = 1.    # penalty for the initial condition
bc_weight = 1.     # penalty for the boundary condition
reg_weight = 1e-4   # penalty for parameter smoothness

# Parameters for the neural network (NN)
epochs = 5000 # Number of epochs to train the NN
epoch_print = 10 # number fo epochs to print the training progress

learning_rate_val = 1e-2 # Learning rate for the NN
learning_rate_decay_factor = 0.98 # Learning rate decay factor
learning_rate_step = 100 # Learning rate decay step

num_hidden_layers = 8 # Number of hidden layers in the NN
num_neurons = 20 # Number of neurons in each hidden layer
num_neurons_layer = num_neurons # The same number of neurons in each hidden layer

activation = "tanh" # Activation function for the hidden layers
reaction_model = "" # Reaction model

# Create the learning rate scheduler
epochs_array = np.arange(learning_rate_step, epochs + learning_rate_step, learning_rate_step)
decay_steps = np.array([learning_rate_val*learning_rate_decay_factor**i 
                        for i in range(int(epochs/learning_rate_step)+1)])

learning_rate = keras.optimizers.schedules.PiecewiseConstantDecay(
                    boundaries=epochs_array, values=decay_steps)

# Open de data file

# Check if the data directory exists
if os.path.exists(data_file):
    print(f"Data directory '{data_file}' exists.")
else:
    print(f"Data directory '{data_file}' not found. Please check the path and filename.")

# Load the data 
try:
    p = pd.read_csv(data_file + "/p.csv", header=None).values # Parameters array
    x_grid = pd.read_csv(data_file + "/x.csv", header=None) # Spatial grid
    t_grid = pd.read_csv(data_file + "/t.csv", header=None) # Temporal grid
    c_data = pd.read_csv(data_file + "/c.csv", header=None) # Solution array

except FileNotFoundError:
    print(f"One or more data files for test case '{test_case}' not found. Please check the paths and filenames.")

# If there is no u and D profile make it constants:
if not os.path.exists(data_file + "/u.csv") or not os.path.exists(data_file + "/D.csv"):
    print("No u and D profiles found. Using constant values.")
    u_actual = np.full_like(x_grid.values.flatten(), p[1]) # Constant advection velocity
    D_actual = np.full_like(x_grid.values.flatten(), p[2]) # Constant diffusion coefficient
else:
    u_actual = pd.read_csv(data_file + "/u.csv", header=None).values.flatten() # Advection velocity profile
    D_actual = pd.read_csv(data_file + "/D.csv", header=None).values.flatten() # Diffusion coefficient profile

# Grid sizes
nx = x_grid.values.shape[0]
nt = t_grid.values.shape[0]

# Perturb the data 
c_data = c_data * (1 + data_perturbation * np.random.randn(c_data.size).reshape(c_data.shape))

# 2D data array
c_data_2d = np.reshape(c_data.values, (nt, nx))

# Coarsen data if needed
if coarsen_data > 1:

    # New rows and columns for coarsened data
    rows = np.r_[0:nt:coarsen_data, nt-1]
    cols = np.r_[0:nx:coarsen_data, nx-1]

    # Coarsen the grid
    x_grid = x_grid.iloc[cols]
    t_grid = t_grid.iloc[rows]

    # Coarsen the data
    c_data_2d = c_data_2d[np.ix_(rows, cols)]
    c_data = c_data_2d.ravel()

    # Mesh and time discretization
nt = t_grid.shape[0]
nx = x_grid.shape[0]
X_grid, T_grid = np.meshgrid(x_grid, t_grid)

# Conver the data to numpy arrays
p = np.array(p).squeeze().astype(np.float32)
x_data = np.array(X_grid, dtype=np.float32).flatten()
t_data = np.array(T_grid, dtype=np.float32).flatten()
c_data = np.array(c_data, dtype=np.float32)

# Data to tensors for TensorFlow
x_tf = tf.expand_dims(tf.convert_to_tensor(x_data), -1)
t_tf = tf.expand_dims(tf.convert_to_tensor(t_data), -1)
c_tf = tf.convert_to_tensor(c_data)

# variables definition for tensorflow
tt = tf.Variable(t_tf, dtype=tf.float32)
xx = tf.Variable(x_tf, dtype=tf.float32)

# perturb the parameters
randp = (p * param_perturbation ** (np.random.randn(p.size) * 2 - 1)).astype(np.float32)

# Aditional parameters for the model (now only beta0 is fixed)
beta0 = randp[0] # Porosity fixed (constant)
sigma = keras.Variable([randp[3]], dtype = tf.float32, name = "sigma", 
                       trainable = (train_parameters and nparm > 2)) # Reaction coefficient

# Build parameter networks for u(x) and D(x)
def build_parameter_network(name, num_hidden_layers, num_neurons, activation='tanh'):
    """
    Build a neural network that maps x to a parameter value (u(x) or D(x))
    """
    x_input = keras.Input(shape=(1,), name=f'{name}_input')
    
    # Hidden layers
    h = x_input
    for i in range(num_hidden_layers):
        h = layers.Dense(
            num_neurons,
            activation=activation,
            kernel_initializer='glorot_normal',
            name=f'{name}_hidden_{i}'
        )(h)
    
    # Output layer
    if name == 'D_network':
        # Use softplus to ensure positivity for diffusion coefficient
        output = layers.Dense(1, activation='softplus', name=f'{name}_output')(h)
    else:
        output = layers.Dense(1, activation='linear', name=f'{name}_output')(h)
    
    model = keras.Model(inputs=x_input, outputs=output, name=name)
    return model

# Create parameter networks
u_network = build_parameter_network('u_network', param_network_layers, param_network_neurons, param_activation)
D_network = build_parameter_network('D_network', param_network_layers, param_network_neurons, param_activation)

print(f"\nParameter networks created:")
print(f"u_network: {u_network.count_params()} parameters")
print(f"D_network: {D_network.count_params()} parameters")

# Store parameters list (now including the networks)
params = [sigma]  # sigma is the only constant parameter being trained
params0 = [p[3]]  # Initial sigma value

# Params names
params_names = ["sigma"]

print(f"Initial parameters: ")
for i in range(len(params0)):
    print(f"{params_names[i]}: {params0[i]}")

if param_perturbation > 0:
    print(f"\nInitial parameters (perturbed):")
    for i in range(len(params)):
        print(f"{params_names[i]}: {randp[3+i]}")

# non-linear reaction term
@tf.function(reduce_retracing=True)
def reaction(c, sigma=sigma, sigma2=1.0e-1, forward_rate=3.0, backward_rate=3.0):
    """ 
    Function reaction term for the PDE. It can be modified to include different reaction models.
        Inputs:
            c(array, float): concentration
            sigma(float): reaction coefficient
            sigma2(float): second reaction coefficient
            forward_rate(float): forward reaction rate
            backward_rate(float): backward reaction rate
        Output:
            reaction term
    """
    if (reaction_model == 'michaelis-menten'):
        return sigma * c**forward_rate / (sigma2 + c**backward_rate) # michaelis-menten kinetics
    elif (reaction_model == 'linear'):
        return sigma * c # linear reaction
    elif (reaction_model == 'quadratic'):
        return sigma * c**2
    elif (reaction_model == 'polynomial'): # Polynomial reaction
        return sigma * c**forward_rate * (1 - c)**backward_rate
    else:
        print('Reaction model not implemented') # Homogeneous reaction (no reaction)
        return tf.zeros_like(c)

# NN construction
def pinn_model(num_hidden_layers=num_hidden_layers, num_neurons_per_layer=num_neurons_layer):
    """
    Function to construct the physics-informed neural network (PINN) model for the advection-diffusion-reaction equation.
        Inputs:
            num_hidden_layers(int): number of hidden layers in the NN
            num_neurons_per_layer(int): number of neurons in each hidden layer
        Output:
            PINN model
    """

    # Create the input layers for the spatial and temporal coordinates
    x_input = keras.Input(shape=(1,))
    t_input = keras.Input(shape=(1,))

    output_c = layers.concatenate([t_input, x_input]) # input layer
    
    # Build the hidden layers
    for i in range(num_hidden_layers):
        output_c = layers.Dense(num_neurons_per_layer,
                                         activation=activation,  
                                         kernel_initializer='glorot_normal',
                                         )(output_c)
    
    # output layer (a single neuron for the concentration)
    output_c = layers.Dense(1)(output_c)

    return keras.Model(inputs=[t_input, x_input], outputs=output_c) # Return the model

@tf.function(reduce_retracing=True)
def custom_loss(inputs, model, u_network, D_network, beta0, sigma, u_true_mean=0.5, D_true_mean=0.1):

    xx, tt, cc = inputs

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(tt)
        tape.watch(xx)

        output_model = model([tt, xx])
        c_model = tf.expand_dims(output_model[:, 0], -1)

        c_x = tape.gradient(c_model, xx)
        c_t = tape.gradient(c_model, tt)

        u_val = u_network(xx)
        D_val = D_network(xx)

        uc    = u_val * c_model
        uc_x  = tape.gradient(uc, xx)
        D_cx  = D_val * c_x
        D_cx_x = tape.gradient(D_cx, xx)

        # Smoothness: gradient of u and D at training points (inside tape, no noise)
        u_x = tape.gradient(u_val, xx)
        D_x = tape.gradient(D_val, xx)

    del tape

    # PDE
    pde_residual = beta0 * c_t + uc_x - D_cx_x - reaction(c_model, sigma)
    pde_loss = tf.reduce_mean(pde_residual ** 2)

    # Data
    data_fitting_loss = tf.reduce_mean((c_model - cc) ** 2)

    # IC
    ic_fitting_loss = tf.reduce_mean((c_model[0:nt] - cc[0:nt]) ** 2)

    # No-flux BC
    flux_left  = u_val[0::nx]    * c_model[0::nx]    - D_val[0::nx]    * c_x[0::nx]
    flux_right = u_val[nx-1::nx] * c_model[nx-1::nx] - D_val[nx-1::nx] * c_x[nx-1::nx]
    bc_fitting_loss = tf.reduce_mean(flux_left**2 + flux_right**2)

    # Smoothness: single weight, no double scaling
    reg_loss = tf.reduce_mean(u_x**2) + tf.reduce_mean(D_x**2)

    # Add to custom_loss return, after the reg_loss line:
    x_anchor = tf.linspace(0.0, 1.0, 50)[:, None]
    u_anchor = u_network(tf.cast(x_anchor, tf.float32))
    D_anchor = D_network(tf.cast(x_anchor, tf.float32))

    anchor_loss = ((tf.reduce_mean(u_anchor) - u_true_mean) ** 2 +
                (tf.reduce_mean(D_anchor) - D_true_mean) ** 2)

    return [pde_loss, data_fitting_loss, ic_fitting_loss, 
            bc_fitting_loss, reg_loss, anchor_loss]

# PINN model definition
model = pinn_model(num_hidden_layers=num_hidden_layers, num_neurons_per_layer=num_neurons_layer)

# Trainable variables for the optimizer
trainable = model.trainable_variables 

# Add parameter networks to trainable variables
if train_parameters:
    print(f"Training spatially varying parameters u(x) and D(x)")
    trainable.extend(u_network.trainable_variables)
    trainable.extend(D_network.trainable_variables)
    # Add sigma if training it
    if nparm > 0:
        trainable.append(sigma)

print(f"\nTotal number of trainable objects: {len(trainable)}")
print(f"  - Concentration network: {len(model.trainable_variables)}")
print(f"  - u_network: {len(u_network.trainable_variables)}")
print(f"  - D_network: {len(D_network.trainable_variables)}")
if nparm > 0:
    print(f"  - sigma: 1")

# Create the optimizer
optimizer_c = keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)
optimizer_params = keras.optimizers.Adam(learning_rate=learning_rate_param, amsgrad=True)

# Create empty lists to store the loss history (now 10 components)
losses = np.zeros((TOTAL_EPOCHS, 11))  # 5
param_values = np.zeros((TOTAL_EPOCHS, 1))  # Only sigma
param_grads = np.zeros((TOTAL_EPOCHS, 1))  # Only sigma gradients
l2_errors = np.zeros((TOTAL_EPOCHS, 1))

# Create arrays to store parameter profiles for visualization
u_profile_history = []
D_profile_history = []
x_profile = x_grid.values.flatten()

print("\nStarting training with spatially varying parameters...")
print(f"Total epochs: {TOTAL_EPOCHS}")
print(f"Parameter networks: {param_network_layers} layers, {param_network_neurons} neurons")

# Compute the mean values of u and D for the anchor loss
u_true_mean = float(np.mean(u_actual))
D_true_mean = float(np.mean(D_actual))

# Phase 1: Fit c to data with u and D frozen
for epoch in range(PHASE1_EPOCHS):

    # Only c-network is trained
    trainable_now = model.trainable_variables

    with tf.GradientTape() as tape:
        loss0 = custom_loss([xx, tt, c_tf], model, u_network, D_network, beta0, sigma, u_true_mean, D_true_mean)
        loss  = [l * w for l, w in zip(loss0, weights_phase1)]
        loss.append(sum(loss))

    gradients = tape.gradient(loss[-1], trainable_now)
    optimizer_c.apply_gradients(zip(gradients, trainable_now))

    # Storage
    losses[epoch, :6] = np.array(loss0)
    losses[epoch, 5:9] = np.array(loss[:4])
    losses[epoch, 9]   = loss[-1]

    sol = model([t_data, x_data]).numpy().reshape(nt, nx)
    l2_errors[epoch] = np.linalg.norm(sol - c_data_2d) / np.linalg.norm(c_data_2d)

    if epoch % epoch_print == 0:
        print(f"[Phase 1] Epoch {epoch+1}/{PHASE1_EPOCHS} | "
              f"Total: {loss[-1].numpy():.4e} | "
              f"Data: {loss0[1].numpy():.4e} | "
              f"L2: {l2_errors[epoch][0]:.4f}")

print(f"\nPhase 1 done. Final L2 error: {l2_errors[PHASE1_EPOCHS-1][0]:.4f}")
print("c-network should now fit data well before u/D are trained.")

# Reset optimizer momentum before Phase 2
optimizer_c      = keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)
optimizer_params = keras.optimizers.Adam(learning_rate=learning_rate_param, amsgrad=True)

all_param_vars = u_network.trainable_variables + D_network.trainable_variables
n_c = len(model.trainable_variables)
n_u = len(u_network.trainable_variables)

for epoch in range(PHASE1_EPOCHS, TOTAL_EPOCHS):
    idx = epoch
    phase2_epoch = epoch - PHASE1_EPOCHS

    # Ramp PDE weight in gradually
    pde_ramp = min(1.0, phase2_epoch / 2000)
    weights_phase2 = [0.01 + 0.99 * pde_ramp,  # pde
                      50.0,                      # data
                      1.0,                       # ic
                      1.0,                       # bc
                      1e-4,                      # reg
                      10.0]                      # anchor

    all_trainable = model.trainable_variables + all_param_vars

    with tf.GradientTape() as tape:
        loss0 = custom_loss([xx, tt, c_tf], model, u_network, D_network,
                            beta0, sigma, u_true_mean, D_true_mean)
        total_loss = sum(l * w for l, w in zip(loss0, weights_phase2))

    all_grads = tape.gradient(total_loss, all_trainable)

    grads_c = all_grads[:n_c]
    grads_u = [tf.clip_by_norm(g, 1.0) if g is not None else g 
               for g in all_grads[n_c:n_c+n_u]]
    grads_D = [tf.clip_by_norm(g, 1.0) if g is not None else g 
               for g in all_grads[n_c+n_u:]]

    optimizer_c.apply_gradients(zip(grads_c, model.trainable_variables))
    optimizer_params.apply_gradients(zip(grads_u + grads_D, all_param_vars))

    # Storage
    losses[idx, :6]  = np.array(loss0)
    losses[idx, 5:9] = np.array(loss[:4])
    losses[idx, 9]   = loss[-1]

    sol = model([t_data, x_data]).numpy().reshape(nt, nx)
    l2_errors[idx] = np.linalg.norm(sol - c_data_2d) / np.linalg.norm(c_data_2d)

    if epoch % epoch_print == 0:
        # Parameter profiles
        x_eval = x_profile.reshape(-1, 1).astype(np.float32)
        u_pred = u_network(x_eval).numpy().flatten()
        D_pred = D_network(x_eval).numpy().flatten()
        l2_u = np.linalg.norm(u_pred - u_actual.flatten()) / np.linalg.norm(u_actual.flatten())
        l2_D = np.linalg.norm(D_pred - D_actual.flatten()) / np.linalg.norm(D_actual.flatten())

        print(f"[Phase 2] Epoch {epoch+1}/{TOTAL_EPOCHS} | "
              f"Total: {loss[-1].numpy():.4e} | "
              f"Data: {loss0[1].numpy():.4e} | "
              f"L2 c: {l2_errors[idx][0]:.4f} | "
              f"L2 u: {l2_u:.4f} | "
              f"L2 D: {l2_D:.4f}")

last_epoch = TOTAL_EPOCHS

#%%

# Plotting the resutls and saving the final profiles

# Create output directory if it doesn't exist
output_dir = "output/" + test_case + "/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Output directory '{output_dir}' created.")
else:
    print(f"Output directory '{output_dir}' already exists.")

# Define the style for plotting
import scienceplots
plt.style.use(['science', 'notebook', 'no-latex']) 

#%%

# Save the final model data
pd.DataFrame(np.array(sol).flatten()).to_csv(output_dir + "model_c.csv", index=False)

# Save the weighted losses 
pd.DataFrame(losses).to_csv(output_dir + "losses.csv", index=False)

# Save the relative error
pd.DataFrame(l2_errors).to_csv(output_dir + "l2_errors.csv", index=False)

# Save the profiles
pd.DataFrame(u_pred).to_csv(output_dir + "model_u.csv", index=False)
pd.DataFrame(D_pred).to_csv(output_dir + "model_D.csv", index=False)

#%%

# Plotting the final profiles
# Plot epochs vs. losses

plt.figure(figsize=(12, 6))
plt.plot(losses[:,0], label='PDE loss')
plt.plot(losses[:,1], label='Data fitting loss')
plt.plot(losses[:,2], label='IC fitting loss')
plt.plot(losses[:,3], label='BC fitting loss')

plt.xlim(0, last_epoch)
#plt.ylim(0, 0.2)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(alpha=0.2)
plt.legend(frameon = True)

# Log scale for loss
plt.yscale('log')

plt.savefig(output_dir + "/loss_history.pdf", dpi=300)

plt.show()

# Plot epochs vs. losses

plt.figure(figsize=(12, 6))
# plt.plot(losses[:, 5], label='PDE loss')
plt.plot(losses[:, 6], label='Data fitting loss')
plt.plot(losses[:, 7], label='IC fitting loss')
plt.plot(losses[:, 8], label='BC fitting loss')
plt.plot(losses[:, 9], label='Total loss')

# Log scale for the loss
plt.yscale('log')

plt.xlim(0, last_epoch)

plt.xlabel('Epochs')
plt.ylabel('Weighted Loss')
plt.grid(alpha=0.3)
plt.legend(frameon = True)

plt.savefig(output_dir + "/loss_history_weighted.pdf", dpi=300)

plt.show()

# Plot l2 errors
plt.figure(figsize=(12, 6))

# Solution error
plt.plot(l2_errors, label='l2 space-time relative error')
# Parameter error
for i in range(nparm):
    plt.plot(np.abs(param_values[:,i] - params0[i])/params0[i], label=f'{params_names[i]} relative error')

plt.xlabel('Epochs')
plt.xlim(0, last_epoch)

plt.ylabel('Relative error')
plt.grid(alpha=0.3)
plt.legend(frameon = True)

plt.yscale('log')

plt.savefig(output_dir + "/errors_history.pdf", dpi=300)

plt.show()

# Plot solutions in space

plt.figure(figsize=(12, 6))

for i in range(0, sol.shape[0], int(sol.shape[0]/5)):
    plt.plot(x_grid, sol[i,:], label='PIIN' if i == 0 else "", color='blue')
    plt.plot(x_grid, c_data_2d[i,:], label='Data' if i == 0 else "", color='green',  linestyle='dashed')
plt.xlabel('x')
plt.ylabel('c')
plt.grid(alpha=0.3)
plt.legend(frameon = True)

plt.xlim(0, 1)

plt.savefig(output_dir + "/solution_profiles.pdf", dpi=300)

plt.show()

# Plot solutions in space

plt.figure(figsize=(12, 6))

for i in range(0, sol.shape[0], int(sol.shape[0]/5)):
    plt.plot(t_grid, sol[:,i], label='PIIN' if i == 0 else "", color='blue')
    plt.plot(t_grid, c_data_2d[:,i], label='Data' if i == 0 else "", color='green' \
    '',  linestyle='dashed')
plt.xlabel('t')
plt.ylabel('c')
plt.grid(alpha=0.3)
plt.legend(frameon = True)

plt.xlim(0, 0.1)

plt.savefig(output_dir + "/solution_profiles_time.pdf", dpi=300)

plt.show()

# Plot parameter profiles
plt.figure(figsize=(12, 6))

# Plot actual profiles
plt.plot(x_grid, u_actual.flatten(), label='u(x)', color='red')
plt.plot(x_grid, D_actual.flatten(), label='D(x)', color='orange')

# Plot computed profiles
plt.plot(x_grid, u_pred, label=f'Predicted u(x)', linestyle='dashed', color='coral') 
plt.plot(x_grid, D_pred, label=f'Predicted D(x)', linestyle='dashed', color='darkorange')
plt.xlabel('x')
plt.ylabel('Parameter value')
plt.grid(alpha=0.3)
plt.legend(frameon = True)

plt.xlim(0, 1)

plt.savefig(output_dir + "/parameter_profiles.pdf", dpi=300)

plt.show()
