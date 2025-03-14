import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Assuming all models are imported from their respective classes
from models import DNN, CGLEPINN_TimeMu, NOPINN, NPINN_PRO_MAX, NPINN_PRO_MAX_CONTINUOUSMU, NPINN_PRO_MAX_TIMEBLOCK_V2

# Preprocessing (run once)
print("Loading data...")
state = np.load("data/states_processed_cropped_finalized.npy")
myu_full = np.load("data/myus_binarized_processed_cropped_finalized.npy")

A_r_data = state.real
A_i_data = state.imag

Nt, Nx, Ny = state.shape
dt, dx, dy = 0.05, 0.3, 0.3
Nx_down, Ny_down = 10, 10
degrade_x = Nx // Nx_down
degrade_y = Ny // Ny_down
degrade_t = 46

# Data points for training
n_data = 20000
idx_t = np.random.randint(0, Nt, size=n_data)
idx_x = np.random.randint(0, Nx, size=n_data)
idx_y = np.random.randint(0, Ny, size=n_data)

t_vals = np.arange(Nt) * dt
x_vals = np.arange(Nx) * dx
y_vals = np.arange(Ny) * dy

t_data_np = t_vals[idx_t]
x_data_np = x_vals[idx_x]
y_data_np = y_vals[idx_y]

Ar_data_np = A_r_data[idx_t, idx_x, idx_y]
Ai_data_np = A_i_data[idx_t, idx_x, idx_y]

device = "cuda" if torch.cuda.is_available() else "cpu"

# Convert to PyTorch tensors
x_data_t = torch.tensor(x_data_np, dtype=torch.float32, device=device).view(-1, 1)
y_data_t = torch.tensor(y_data_np, dtype=torch.float32, device=device).view(-1, 1)
t_data_t = torch.tensor(t_data_np, dtype=torch.float32, device=device).view(-1, 1)
Ar_data_t = torch.tensor(Ar_data_np, dtype=torch.float32, device=device).view(-1, 1)
Ai_data_t = torch.tensor(Ai_data_np, dtype=torch.float32, device=device).view(-1, 1)

# Collocation points for PDE loss
n_coll = 20000
t_eqs_np = np.random.uniform(0, t_vals[-1], size=n_coll)
x_eqs_np = np.random.uniform(0, x_vals[-1], size=n_coll)
y_eqs_np = np.random.uniform(0, y_vals[-1], size=n_coll)

x_eqs_t = torch.tensor(x_eqs_np, dtype=torch.float32, device=device, requires_grad=True).view(-1, 1)
y_eqs_t = torch.tensor(y_eqs_np, dtype=torch.float32, device=device, requires_grad=True).view(-1, 1)
t_eqs_t = torch.tensor(t_eqs_np, dtype=torch.float32, device=device, requires_grad=True).view(-1, 1)

# Function to train each model
def run_model(model, model_name, n_epochs=2000, batch_size=2048, lr=1e-3, video_freq=500):
    print(f"Training {model_name}...")
    model.to(device)
    model.train_model(
        x_data=x_data_t,
        y_data=y_data_t,
        t_data=t_data_t,
        A_r_data=Ar_data_t,
        A_i_data=Ai_data_t,
        x_eqs=x_eqs_t,
        y_eqs=y_eqs_t,
        t_eqs=t_eqs_t,
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        model_name=model_name,
        output_dir="./results",
        video_freq=video_freq,
        state_exp=state,
        myu_full_exp=myu_full,
        x_vals=x_vals,
        y_vals=y_vals,
        t_vals=t_vals,
        device=device
    )
    print(f"Finished training {model_name}.\n")

# Sequentially run all models
if __name__ == "__main__":
    # Model 1: CGLEPINN_TimeMu
    model_1 = CGLEPINN_TimeMu(
        layers=[3, 64, 64, 2],
        Nt=Nt, Nx_down=Nx_down, Ny_down=Ny_down,
        dt=dt, dx=dx, dy=dy,
        degrade_x=degrade_x, degrade_y=degrade_y,
        delta=1.0, myu_init=myu_full, weight_pde=1.0, device=device
    )
    run_model(model_1, "CGLEPINN_TimeMu")

    # Model 2: NOPINN
    model_2 = NOPINN(
        layers=[3, 64, 64, 2],
        Nt=Nt, Nx_down=Nx_down, Ny_down=Ny_down,
        dt=dt, dx=dx, dy=dy,
        degrade_x=degrade_x, degrade_y=degrade_y,
        delta=1.0, myu_init=myu_full, weight_pde=1.0, dropout_rate=0.1, device=device
    )
    run_model(model_2, "NOPINN")

    # Model 3: NPINN_PRO_MAX
    model_3 = NPINN_PRO_MAX(
        layers=[3, 128, 128, 2],
        Nt=Nt, Nx=Nx, Ny=Ny,
        Nx_down=Nx_down, Ny_down=Ny_down,
        dt=dt, dx=dx, dy=dy,
        degrade_x=degrade_x, degrade_y=degrade_y,
        delta=0.01, weight_pde=1.0, device=device
    )
    run_model(model_3, "NPINN_PRO_MAX")

    # Model 4: NPINN_PRO_MAX_CONTINUOUSMU
    model_4 = NPINN_PRO_MAX_CONTINUOUSMU(
        layers=[3, 128, 128, 2],
        Nt=Nt, Nx=Nx, Ny=Ny,
        Nx_down=Nx_down, Ny_down=Ny_down,
        dt=dt, dx=dx, dy=dy,
        degrade_x=degrade_x, degrade_y=degrade_y,
        delta=0.01, weight_pde=1.0, device=device
    )
    run_model(model_4, "NPINN_PRO_MAX_CONTINUOUSMU")

    # Model 5: NPINN_PRO_MAX_TIMEBLOCK_V2
    model_5 = NPINN_PRO_MAX_TIMEBLOCK_V2(
        layers=[3, 128, 256, 256, 128, 2],
        Nt=Nt, Nx=Nx, Ny=Ny,
        Nx_down=Nx_down, Ny_down=Ny_down,
        dt=dt, dx=dx, dy=dy,
        degrade_x=degrade_x, degrade_y=degrade_y,
        degrade_t=degrade_t,
        delta=0.01, weight_pde=1.0, device=device
    )
    run_model(model_5, "NPINN_PRO_MAX_TIMEBLOCK_V2")
