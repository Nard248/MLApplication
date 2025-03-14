import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from final.utils import *
import torch
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.layers(x))

class ImprovedDNN(nn.Module):
    def __init__(self, layers, n_res_blocks=3):
        super().__init__()

        modules = []
        # Input projection
        modules.append(nn.Linear(layers[0], layers[1]))
        modules.append(nn.BatchNorm1d(layers[1]))
        modules.append(nn.GELU())

        # Middle layers with residual blocks
        mid_dim = layers[1]
        for _ in range(n_res_blocks):
            modules.append(ResBlock(mid_dim))

        # Additional dense layers with increasing width
        for i in range(1, len(layers)-2):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            modules.append(nn.BatchNorm1d(layers[i+1]))
            modules.append(nn.GELU())

        # Output projection
        modules.append(nn.Linear(layers[-2], layers[-1]))

        self.net = nn.Sequential(*modules)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

class NPINN_PRO_MAX_TIMEBLOCK_V2(nn.Module):
    def __init__(
        self,
        layers,
        Nt, Nx, Ny,
        Nx_down, Ny_down,
        dt, dx, dy,
        degrade_x, degrade_y,
        degrade_t,
        delta=0.01,
        weight_pde=1.0,
        device='cpu'
    ):
        super().__init__()
        self.device = device
        self.delta = delta
        self.weight_pde = weight_pde

        # Domain parameters (same as original)
        self.Nt, self.Nx, self.Ny = Nt, Nx, Ny
        self.Nx_down, self.Ny_down = Nx_down, Ny_down
        self.dt, self.dx, self.dy = dt, dx, dy
        self.degrade_x, self.degrade_y = degrade_x, degrade_y
        self.degrade_t = degrade_t
        self.Nt_down = Nt // degrade_t

        # Improved neural network for A(x,y,t)
        self.dnn = ImprovedDNN(layers, n_res_blocks=3).to(device)

        # The trainable mu_small (same as original)
        init = 0.3 * torch.randn(self.Nt_down, Nx_down, Ny_down)
        self.mu_small_raw = nn.Parameter(init.to(device))

        # Spatial frequency encodings
        self.register_buffer('freq_x', torch.linspace(0, 10, layers[0]))
        self.register_buffer('freq_y', torch.linspace(0, 10, layers[0]))
        self.register_buffer('freq_t', torch.linspace(0, 10, layers[0]))

    def positional_encoding(self, x, y, t):
        """Add spatial frequency encodings to the input"""
        enc_x = torch.sin(x * self.freq_x[None, :])
        enc_y = torch.sin(y * self.freq_y[None, :])
        enc_t = torch.sin(t * self.freq_t[None, :])
        return (enc_x + enc_y + enc_t) / 3.0

    def net_A(self, x, y, t):
        # Concatenate inputs and add positional encoding
        inp_raw = torch.cat([x, y, t], dim=1)
        pos_enc = self.positional_encoding(x, y, t)
        inp = inp_raw + pos_enc

        out = self.dnn(inp)
        return out[:, 0:1], out[:, 1:2]

    def forward(self, x, y, t):
        return self.net_A(x, y, t)

    def binarize_mu_small(self):
        """
        Hard threshold the entire mu_small_raw -> 0 or 1 in place.
        This is optional and breaks gradient flow.
        """
        with torch.no_grad():
            self.mu_small_raw.data = (self.mu_small_raw.data > 0.0).float()

    def get_myu_collocation(self, x, y, t):
        """
        (x,y,t) -> integer indices (i, j_down, k_down).
        But for time, we do i = floor(t/dt), then i_down = floor(i/degrade_t).
        Then threshold to 0/1.
        """
        # Convert t-> i in [0..Nt-1]
        i = (t[:,0] / self.dt).round().long().clamp(0, self.Nt-1)
        # Then the coarse time index
        i_down = (i // self.degrade_t).clamp(0, self.Nt_down-1)

        j_down = (x[:,0] / (self.dx*self.degrade_x)).floor().long()
        k_down = (y[:,0] / (self.dy*self.degrade_y)).floor().long()

        j_down = j_down.clamp(0, self.Nx_down-1)
        k_down = k_down.clamp(0, self.Ny_down-1)

        mu_vals_raw = self.mu_small_raw[i_down, j_down, k_down]
        # Binarize for PDE
        mu_bin = (mu_vals_raw > 0.0).float()  # shape (batch,)
        return mu_bin.view(-1,1)

    def pde_residual(self, x, y, t):
        A_r, A_i = self.net_A(x,y,t)
        mu_vals = self.get_myu_collocation(x,y,t)

        A_r_t = torch.autograd.grad(A_r, t,
            grad_outputs=torch.ones_like(A_r),
            create_graph=True, retain_graph=True)[0]
        A_i_t = torch.autograd.grad(A_i, t,
            grad_outputs=torch.ones_like(A_i),
            create_graph=True, retain_graph=True)[0]

        # wrt x
        A_r_x = torch.autograd.grad(A_r, x,
            grad_outputs=torch.ones_like(A_r),
            create_graph=True, retain_graph=True)[0]
        A_i_x = torch.autograd.grad(A_i, x,
            grad_outputs=torch.ones_like(A_i),
            create_graph=True, retain_graph=True)[0]

        # wrt y
        A_r_y = torch.autograd.grad(A_r, y,
            grad_outputs=torch.ones_like(A_r),
            create_graph=True, retain_graph=True)[0]
        A_i_y = torch.autograd.grad(A_i, y,
            grad_outputs=torch.ones_like(A_i),
            create_graph=True, retain_graph=True)[0]

        # second derivatives
        A_r_xx = torch.autograd.grad(A_r_x, x,
            grad_outputs=torch.ones_like(A_r_x),
            create_graph=True, retain_graph=True)[0]
        A_r_yy = torch.autograd.grad(A_r_y, y,
            grad_outputs=torch.ones_like(A_r_y),
            create_graph=True, retain_graph=True)[0]

        A_i_xx = torch.autograd.grad(A_i_x, x,
            grad_outputs=torch.ones_like(A_i_x),
            create_graph=True, retain_graph=True)[0]
        A_i_yy = torch.autograd.grad(A_i_y, y,
            grad_outputs=torch.ones_like(A_i_y),
            create_graph=True, retain_graph=True)[0]

        lapA_r = A_r_xx + A_r_yy
        lapA_i = A_i_xx + A_i_yy

        A_abs2 = A_r**2 + A_i**2

        f_r = A_r_t - mu_vals*A_r - self.delta*lapA_r + A_abs2*A_r
        f_i = A_i_t - mu_vals*A_i - self.delta*lapA_i + A_abs2*A_i
        return f_r, f_i

    def loss_pde(self, x_eqs, y_eqs, t_eqs):
        f_r, f_i = self.pde_residual(x_eqs, y_eqs, t_eqs)
        return torch.mean(f_r**2 + f_i**2)

    def gradient_penalty(self, x, y, t):
        """Additional regularization for derivatives"""
        A_r, A_i = self.net_A(x, y, t)

        gradients_r = torch.autograd.grad(
            A_r.sum(), x, create_graph=True, retain_graph=True)[0]
        gradients_i = torch.autograd.grad(
            A_i.sum(), x, create_graph=True, retain_graph=True)[0]

        return (gradients_r.pow(2).sum() + gradients_i.pow(2).sum()) / x.shape[0]

    def loss_data(self, x_data, y_data, t_data, A_r_data, A_i_data):
        A_r_pred, A_i_pred = self.net_A(x_data, y_data, t_data)

        # L2 loss
        l2_loss = torch.mean((A_r_pred - A_r_data)**2 + (A_i_pred - A_i_data)**2)

        # Add L1 loss for better stability
        l1_loss = torch.mean(torch.abs(A_r_pred - A_r_data) + torch.abs(A_i_pred - A_i_data))

        return l2_loss + 0.1 * l1_loss

    def train_model(
        self,
        x_data, y_data, t_data, A_r_data, A_i_data,
        x_eqs, y_eqs, t_eqs,
        n_epochs=200000,
        lr=1e-3,
        batch_size=1024,
        model_name="MyModel",
        output_dir="./results",
        video_freq=10000,
        state_exp=None,
        myu_full_exp=None,
        x_vals=None,
        y_vals=None,
        t_vals=None,
        device="cpu"
    ):
        import os
        from datetime import datetime
        from torch.utils.data import TensorDataset, DataLoader

        # Create dataloaders
        train_dataset = TensorDataset(x_data, y_data, t_data, A_r_data, A_i_data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Collocation points dataloader
        coll_dataset = TensorDataset(x_eqs, y_eqs, t_eqs)
        coll_loader = DataLoader(coll_dataset, batch_size=batch_size, shuffle=True)

        # Optimizer and scheduler
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=n_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            div_factor=25.0
        )

        # Gradient clipping
        max_grad_norm = 1.0

        model_folder = os.path.join(output_dir, model_name)
        os.makedirs(model_folder, exist_ok=True)

        best_loss = float('inf')
        patience_counter = 0
        patience = 10  # epochs for early stopping
        loss_dict_list = []

        for epoch in range(n_epochs):
            total_loss = 0
            total_data_loss = 0
            total_pde_loss = 0

            for (x_d, y_d, t_d, ar_d, ai_d), (x_e, y_e, t_e) in zip(train_loader, coll_loader):
                optimizer.zero_grad()

                # Data loss
                data_loss = self.loss_data(x_d, y_d, t_d, ar_d, ai_d)

                # PDE loss
                pde_loss = self.loss_pde(x_e, y_e, t_e)

                # Gradient penalty
                grad_penalty = self.gradient_penalty(x_e, y_e, t_e)

                # Total loss
                loss = data_loss + self.weight_pde * pde_loss + 0.01 * grad_penalty

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                total_data_loss += data_loss.item()
                total_pde_loss += pde_loss.item()

            avg_loss = total_loss / len(train_loader)
            avg_data_loss = total_data_loss / len(train_loader)
            avg_pde_loss = total_pde_loss / len(train_loader)

            temp_loss_dict = {
                "loss": avg_loss,
                "data_loss": avg_data_loss,
                "pde_loss": avg_pde_loss
            }

            loss_dict_list.append(temp_loss_dict)

            if epoch % 500 == 0:
                print(f"Epoch={epoch}, total={avg_loss:.4e}, data={avg_data_loss:.4e}, PDE={avg_pde_loss:.4e}")
                # Early stopping check
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.state_dict(), os.path.join(model_folder, f"{model_name}_best.pt"))
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered")
                        break

            # Rest of the training loop (checkpointing, video generation) remains the same
            if (epoch % video_freq == 0 and epoch > 0):
                ckpt_path = os.path.join(model_folder, f"{model_name}_epoch_{epoch}.pt")
                torch.save(self.state_dict(), ckpt_path)
                print(f"Checkpoint saved at {ckpt_path}")

                if all(x is not None for x in [state_exp, myu_full_exp, x_vals, y_vals, t_vals]):
                    vid_name = f"{model_name}_epoch_{epoch}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    video_folder = os.path.join(model_folder, "videos")
                    os.makedirs(video_folder, exist_ok=True)
                    video_path = os.path.join(video_folder, vid_name)
                    generate_video(state_exp, myu_full_exp, self, x_vals, y_vals, t_vals, device=device, output_path=video_path)

        loss_df = pd.DataFrame(loss_dict_list)
        loss_csv_path = os.path.join(model_folder, f"{model_name}_losses.csv")
        loss_df.to_csv(loss_csv_path, index=False)
        final_ckpt = os.path.join(model_folder, f"{model_name}_final_{n_epochs}.pt")
        torch.save(self.state_dict(), final_ckpt)
        print(f"Final checkpoint saved at {final_ckpt}\nTraining done.\n")



    def expand_myu_full(self, do_binarize=True, scale_255=False):
        """
        Expand mu_small_raw shape = (Nt_down, Nx_down, Ny_down)
        to full shape (Nt, Nx, Ny) by:
         1) repeat_interleave along time dim by degrade_t
         2) repeat_interleave along x,y dims by degrade_x, degrade_y
        """
        with torch.no_grad():
            mu_raw = self.mu_small_raw.detach()  # shape (Nt_down, Nx_down, Ny_down)

            if do_binarize:
                mu_bin = (mu_raw>0.0).float()
            else:
                mu_bin = mu_raw

            # time expansion
            mu_time = mu_bin.repeat_interleave(self.degrade_t, dim=0)
            # shape => (Nt_down*degrade_t, Nx_down, Ny_down) = (Nt, Nx_down, Ny_down)

            # expand in x,y
            mu_full_x = mu_time.repeat_interleave(self.degrade_x, dim=1)
            mu_full_xy = mu_full_x.repeat_interleave(self.degrade_y, dim=2)

            if scale_255:
                mu_full_xy = mu_full_xy * 255.0

            return mu_full_xy.cpu().numpy()

    def predict(self, x, y, t):
        self.eval()
        with torch.no_grad():
            A_r, A_i = self.net_A(x, y, t)
        return A_r.cpu().numpy(), A_i.cpu().numpy()

state = np.load("data/validation/data_10.npy")
myu_full = np.load("data/validation/myu_10.npy")

print("State shape:", state.shape, state.dtype)  # (350,530,880), complex128
print("Myu shape:  ", myu_full.shape, myu_full.dtype)

A_r_data = state.real
A_i_data = state.imag

Nt, Nx, Ny = state.shape
dt, dx, dy = 0.05, 0.3, 0.3
Nx_down, Ny_down = 18, 20
degrade_x = Nx // Nx_down
degrade_y = Ny // Ny_down

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

device = 'cuda'

x_data_t = torch.tensor(x_data_np, dtype=torch.float32, device=device).view(-1, 1)
y_data_t = torch.tensor(y_data_np, dtype=torch.float32, device=device).view(-1, 1)
t_data_t = torch.tensor(t_data_np, dtype=torch.float32, device=device).view(-1, 1)
Ar_data_t = torch.tensor(Ar_data_np, dtype=torch.float32, device=device).view(-1, 1)
Ai_data_t = torch.tensor(Ai_data_np, dtype=torch.float32, device=device).view(-1, 1)

n_coll = 20000
t_eqs_np = np.random.uniform(0, t_vals[-1], size=n_coll)
x_eqs_np = np.random.uniform(0, x_vals[-1], size=n_coll)
y_eqs_np = np.random.uniform(0, y_vals[-1], size=n_coll)

x_eqs_t = torch.tensor(x_eqs_np, dtype=torch.float32, device=device, requires_grad=True).view(-1, 1)
y_eqs_t = torch.tensor(y_eqs_np, dtype=torch.float32, device=device, requires_grad=True).view(-1, 1)
t_eqs_t = torch.tensor(t_eqs_np, dtype=torch.float32, device=device, requires_grad=True).view(-1, 1)


# First initialize the improved model
model_5 = NPINN_PRO_MAX_TIMEBLOCK_V2(
    layers=[3, 128, 256, 256, 128, 2],  # Deeper and wider architecture
    Nt=Nt, Nx=Nx, Ny=Ny,
    Nx_down=Nx_down, Ny_down=Ny_down,
    dt=dt, dx=dx, dy=dy,
    degrade_x=degrade_x, degrade_y=degrade_y,
    delta=0.01,
    weight_pde=0.1,
    device='cuda',
    degrade_t=121
).to('cuda')

model_5.train_model(
    x_data=x_data_t,
    y_data=y_data_t,
    t_data=t_data_t,
    A_r_data=Ar_data_t,
    A_i_data=Ai_data_t,
    x_eqs=x_eqs_t,
    y_eqs=y_eqs_t,
    t_eqs=t_eqs_t,
    n_epochs=50,
    lr=1e-3,
    batch_size=2048,
    model_name="TimeBlockerV2_Test",
    output_dir="./results",
    video_freq=2,
    state_exp=state,
    myu_full_exp=myu_full,
    x_vals=x_vals,
    y_vals=y_vals,
    t_vals=t_vals,
    device='cuda'
)