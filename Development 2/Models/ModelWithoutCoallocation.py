import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime
import os


class SimplifiedResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.layers(x))


class CompactDNN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, n_res_blocks=2):
        super().__init__()

        modules = []
        modules.append(nn.Linear(input_dim, hidden_dim))
        modules.append(nn.GELU())

        for _ in range(n_res_blocks):
            modules.append(SimplifiedResBlock(hidden_dim))

        modules.append(nn.Linear(hidden_dim, 2))  # 2 for A_r, A_i

        self.net = nn.Sequential(*modules)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class AdaptedPINN(nn.Module):
    def __init__(
            self,
            Nt=750,
            Nx=64,
            Ny=64,
            dt=0.05,
            dx=0.3,
            dy=0.3,
            delta=0.01,
            weight_pde=0.1,
            device='cuda'
    ):
        super().__init__()
        self.device = device
        self.delta = delta
        self.weight_pde = weight_pde

        self.Nt, self.Nx, self.Ny = Nt, Nx, Ny
        self.dt, self.dx, self.dy = dt, dx, dy

        self.degrade_t = 15
        self.Nt_unique = Nt // self.degrade_t

        self.Nx_down = Nx
        self.Ny_down = Ny
        self.degrade_x = 1
        self.degrade_y = 1

        self.dnn = CompactDNN(input_dim=3, hidden_dim=64, n_res_blocks=2).to(device)

        init = torch.zeros(self.Nt_unique, self.Nx_down, self.Ny_down)
        self.mu_small = nn.Parameter(init.to(device))

    def forward(self, x, y, t):
        return self.net_A(x, y, t)

    def net_A(self, x, y, t):
        inp = torch.cat([x, y, t], dim=1)
        out = self.dnn(inp)
        return out[:, 0:1], out[:, 1:2]

    def get_myu_collocation(self, x, y, t):
        i = (t[:, 0] / self.dt).round().long().clamp(0, self.Nt - 1)
        i_down = (i // self.degrade_t).clamp(0, self.Nt_unique - 1)

        j = (x[:, 0] / self.dx).floor().long().clamp(0, self.Nx - 1)
        k = (y[:, 0] / self.dy).floor().long().clamp(0, self.Ny - 1)

        mu_vals_raw = self.mu_small[i_down, j, k]
        mu_binary = (mu_vals_raw > 0.0).float()

        return mu_binary.view(-1, 1) * 255.0

    def pde_residual(self, x, y, t):
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)

        A_r, A_i = self.net_A(x, y, t)
        mu_vals = self.get_myu_collocation(x, y, t)

        mu_vals = mu_vals / 255.0

        A_r_t = torch.autograd.grad(A_r, t, torch.ones_like(A_r), create_graph=True)[0]
        A_i_t = torch.autograd.grad(A_i, t, torch.ones_like(A_i), create_graph=True)[0]

        A_r_x = torch.autograd.grad(A_r, x, torch.ones_like(A_r), create_graph=True)[0]
        A_i_x = torch.autograd.grad(A_i, x, torch.ones_like(A_i), create_graph=True)[0]

        A_r_y = torch.autograd.grad(A_r, y, torch.ones_like(A_r), create_graph=True)[0]
        A_i_y = torch.autograd.grad(A_i, y, torch.ones_like(A_i), create_graph=True)[0]

        A_r_xx = torch.autograd.grad(A_r_x, x, torch.ones_like(A_r_x), create_graph=True)[0]
        A_r_yy = torch.autograd.grad(A_r_y, y, torch.ones_like(A_r_y), create_graph=True)[0]

        A_i_xx = torch.autograd.grad(A_i_x, x, torch.ones_like(A_i_x), create_graph=True)[0]
        A_i_yy = torch.autograd.grad(A_i_y, y, torch.ones_like(A_i_y), create_graph=True)[0]

        lapA_r = A_r_xx + A_r_yy
        lapA_i = A_i_xx + A_i_yy

        A_abs2 = A_r ** 2 + A_i ** 2

        f_r = A_r_t - mu_vals * A_r - self.delta * lapA_r + A_abs2 * A_r
        f_i = A_i_t - mu_vals * A_i - self.delta * lapA_i + A_abs2 * A_i

        return f_r, f_i

    def loss_pde(self, x_eqs, y_eqs, t_eqs):
        f_r, f_i = self.pde_residual(x_eqs, y_eqs, t_eqs)
        return torch.mean(f_r ** 2 + f_i ** 2)

    def loss_data(self, x_data, y_data, t_data, A_r_data, A_i_data):
        A_r_pred, A_i_pred = self.net_A(x_data, y_data, t_data)
        return torch.mean((A_r_pred - A_r_data) ** 2 + (A_i_pred - A_i_data) ** 2)

    def train_model(
            self,
            x_data, y_data, t_data, A_r_data, A_i_data,
            n_epochs=10000,
            lr=1e-3,
            batch_size=1024,
            model_name="AdaptedPINN",
            output_dir="./results",
            validation_split=0.2
    ):
        from torch.utils.data import TensorDataset, DataLoader, random_split

        full_dataset = TensorDataset(x_data, y_data, t_data, A_r_data, A_i_data)
        val_size = int(len(full_dataset) * validation_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=100)

        best_loss = float('inf')

        for epoch in range(n_epochs):
            self.train()
            train_loss = 0
            for batch_data in train_loader:
                optimizer.zero_grad()

                x_d, y_d, t_d, ar_d, ai_d = [b.to(self.device) for b in batch_data]

                data_loss = self.loss_data(x_d, y_d, t_d, ar_d, ai_d)

                x_d.requires_grad_(True)
                y_d.requires_grad_(True)
                t_d.requires_grad_(True)

#                 print(x_d.requires_grad, y_d.requires_grad, t_d.requires_grad)  # Should all be True
#                 print(x_d.shape, y_d.shape, t_d.shape)  # Should be [batch_size, 1]

                pde_loss = self.loss_pde(x_d, y_d, t_d)

                loss = data_loss + self.weight_pde * pde_loss

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            self.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    x_d, y_d, t_d, ar_d, ai_d = [b.to(self.device) for b in batch]
                    data_loss = self.loss_data(x_d, y_d, t_d, ar_d, ai_d)
                    val_loss += data_loss.item()

            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4e}, Val Loss = {val_loss:.4e}")

            if val_loss < best_loss:
                best_loss = val_loss
                os.makedirs(output_dir, exist_ok=True)
                torch.save(self.state_dict(), f"{output_dir}/{model_name}_best.pt")

    def expand_myu_full(self):
        with torch.no_grad():
            mu_binary = (self.mu_small > 0.0).float()
            mu_time_expanded = mu_binary.repeat_interleave(self.degrade_t, dim=0)
            return mu_time_expanded.cpu().numpy() * 255.0


def prepare_data(state, myu):
    Nt, Nx, Ny = state.shape
    dt, dx, dy = 0.05, 0.3, 0.3

    n_data = 10000
    idx_t = np.random.randint(0, Nt, size=n_data)
    idx_x = np.random.randint(0, Nx, size=n_data)
    idx_y = np.random.randint(0, Ny, size=n_data)

    t_vals = np.arange(Nt) * dt
    x_vals = np.arange(Nx) * dx
    y_vals = np.arange(Ny) * dy

    t_data = t_vals[idx_t]
    x_data = x_vals[idx_x]
    y_data = y_vals[idx_y]

    Ar_data = state.real[idx_t, idx_x, idx_y]
    Ai_data = state.imag[idx_t, idx_x, idx_y]

    n_coll = 10000
    t_eqs = np.random.uniform(0, t_vals[-1], size=n_coll)
    x_eqs = np.random.uniform(0, x_vals[-1], size=n_coll)
    y_eqs = np.random.uniform(0, y_vals[-1], size=n_coll)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tensors = {}
    for name, data in [
        ('x_data', x_data), ('y_data', y_data), ('t_data', t_data),
        ('Ar_data', Ar_data), ('Ai_data', Ai_data),
        ('x_eqs', x_eqs), ('y_eqs', y_eqs), ('t_eqs', t_eqs)
    ]:
        tensor = torch.tensor(data, dtype=torch.float32, device=device).view(-1, 1)
        if 'eqs' in name:
            tensor.requires_grad = True
        tensors[name] = tensor

    return tensors