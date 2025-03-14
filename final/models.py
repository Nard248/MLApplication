import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from datetime import datetime 
from utils import *
import torch
import numpy as np

################################################################### MODEL 1 ###################################################################
# class DNN(nn.Module):
#     """Simple MLP that maps (x,y,t)->(A_r, A_i)."""
#     def __init__(self, layers):
#         super().__init__()
#         net = []
#         for i in range(len(layers)-1):
#             net.append(nn.Linear(layers[i], layers[i+1]))
#             if i < len(layers)-2:
#                 net.append(nn.Tanh())
#         self.net = nn.Sequential(*net)
#
#     def forward(self, x):
#         return self.net(x)
#
#
# class CGLEPINN_TimeMu(nn.Module):
#     """
#     PINN for Complex Ginzburg-Landau with a time-dependent, downsampled mu-field:
#       mu_small shape = (Nt, Nx_down, Ny_down).
#     """
#     def __init__(
#         self,
#         layers,             # e.g. [3,64,64,2] for A(x,y,t)
#         Nt, Nx_down, Ny_down,
#         dt, dx, dy,
#         degrade_x, degrade_y,
#         delta=1.0,
#         myu_init=None,      # shape (Nt, Nx_down, Ny_down)
#         weight_pde=1.0,
#         device='cpu'
#     ):
#         super().__init__()
#         self.device = device
#         self.delta = delta
#         self.weight_pde = weight_pde
#
#         # The net that predicts A(x,y,t)
#         self.dnn = DNN(layers).to(device)
#
#         # Store shapes, spacing
#         self.Nt = Nt
#         self.Nx_down = Nx_down
#         self.Ny_down = Ny_down
#         self.dt = dt
#         self.dx = dx
#         self.dy = dy
#         self.degrade_x = degrade_x
#         self.degrade_y = degrade_y
#
#         # Create a trainable param for mu_small
#         if myu_init is None:
#             # random init
#             init = torch.zeros(Nt, Nx_down, Ny_down)  # or normal, etc.
#         else:
#             # pass in the array you downsampled
#             init = torch.tensor(myu_init, dtype=torch.float32)
#
#         self.mu_small = nn.Parameter(init.to(device))
#
#     def net_A(self, x, y, t):
#         """Forward pass for (A_r, A_i)."""
#         inp = torch.cat([x, y, t], dim=1)  # shape (batch,3)
#         out = self.dnn(inp)               # shape (batch,2)
#         A_r = out[:,0:1]
#         A_i = out[:,1:2]
#         return A_r, A_i
#
#     def get_mu(self, x, y, t):
#         """
#         Convert (x,y,t) -> integer indices [i, j, k], and return mu_small[i,j,k].
#         """
#         # shape of x,y,t is (batch,1).
#         i = (t[:,0]/self.dt).round().long().clamp(0, self.Nt-1)  # time index
#         j = (x[:,0]/(self.dx*self.degrade_x)).floor().long()
#         k = (y[:,0]/(self.dy*self.degrade_y)).floor().long()
#
#         j = j.clamp(0, self.Nx_down-1)
#         k = k.clamp(0, self.Ny_down-1)
#
#         mu_vals = self.mu_small[i, j, k]  # shape (batch,)
#         return mu_vals.view(-1,1)
#
#     def pde_residual(self, x, y, t):
#         """A_t = mu*A + delta Lap(A) - |A|^2 A."""
#         A_r, A_i = self.net_A(x, y, t)
#         mu_vals = self.get_mu(x, y, t)
#
#         # partial derivatives wrt t:
#         A_r_t = torch.autograd.grad(
#             A_r, t,
#             grad_outputs=torch.ones_like(A_r),
#             create_graph=True, retain_graph=True
#         )[0]
#         A_i_t = torch.autograd.grad(
#             A_i, t,
#             grad_outputs=torch.ones_like(A_i),
#             create_graph=True, retain_graph=True
#         )[0]
#
#         # wrt x:
#         A_r_x = torch.autograd.grad(
#             A_r, x,
#             grad_outputs=torch.ones_like(A_r),
#             create_graph=True, retain_graph=True
#         )[0]
#         A_i_x = torch.autograd.grad(
#             A_i, x,
#             grad_outputs=torch.ones_like(A_i),
#             create_graph=True, retain_graph=True
#         )[0]
#
#         # wrt y:
#         A_r_y = torch.autograd.grad(A_r, y,
#             grad_outputs=torch.ones_like(A_r),
#             create_graph=True, retain_graph=True
#         )[0]
#         A_i_y = torch.autograd.grad(A_i, y,
#             grad_outputs=torch.ones_like(A_i),
#             create_graph=True, retain_graph=True
#         )[0]
#
#         # second derivatives -> Laplacian
#         A_r_xx = torch.autograd.grad(A_r_x, x,
#             grad_outputs=torch.ones_like(A_r_x),
#             create_graph=True, retain_graph=True
#         )[0]
#         A_r_yy = torch.autograd.grad(A_r_y, y,
#             grad_outputs=torch.ones_like(A_r_y),
#             create_graph=True, retain_graph=True
#         )[0]
#
#         A_i_xx = torch.autograd.grad(A_i_x, x,
#             grad_outputs=torch.ones_like(A_i_x),
#             create_graph=True, retain_graph=True
#         )[0]
#         A_i_yy = torch.autograd.grad(A_i_y, y,
#             grad_outputs=torch.ones_like(A_i_y),
#             create_graph=True, retain_graph=True
#         )[0]
#
#         lapA_r = A_r_xx + A_r_yy
#         lapA_i = A_i_xx + A_i_yy
#
#         A_abs2 = A_r**2 + A_i**2
#
#         f_r = A_r_t - mu_vals*A_r - self.delta*lapA_r + A_abs2*A_r
#         f_i = A_i_t - mu_vals*A_i - self.delta*lapA_i + A_abs2*A_i
#
#         return f_r, f_i
#
#     def loss_pde(self, x_eqs, y_eqs, t_eqs):
#         f_r, f_i = self.pde_residual(x_eqs, y_eqs, t_eqs)
#         return torch.mean(f_r**2 + f_i**2)
#
#     def loss_data(self, x_data, y_data, t_data, A_r_data, A_i_data):
#         A_r_pred, A_i_pred = self.net_A(x_data, y_data, t_data)
#         return torch.mean((A_r_pred - A_r_data)**2 + (A_i_pred - A_i_data)**2)
#
#     def train_model(
#         self,
#         x_data, y_data, t_data, A_r_data, A_i_data,
#         x_eqs, y_eqs, t_eqs,
#         n_epochs=3000,
#         lr=1e-3,
#         video_freq=4000,
#         state_exp=None,
#         myu_full_exp=None,
#         x_vals=None,
#         y_vals=None,
#         t_vals=None,
#         model_name="MyModel"
#     ):
#         class_name = self.__class__.__name__
#         model_folder = f"./final_results/model_1_{class_name}"
#         os.makedirs(model_folder, exist_ok=True)
#
#         losses_file = os.path.join(model_folder, "losses.csv")
#         checkpoint_dir = model_folder
#         video_folder = os.path.join(model_folder, "videos")
#         os.makedirs(video_folder, exist_ok=True)
#
#         optimizer = optim.Adam(self.parameters(), lr=lr)
#         loss_data = []
#
#         for epoch in range(n_epochs):
#             optimizer.zero_grad()
#             l_pde = self.loss_pde(x_eqs, y_eqs, t_eqs)
#             l_data = self.loss_data(x_data, y_data, t_data, A_r_data, A_i_data)
#             loss = l_data + self.weight_pde * l_pde
#             loss.backward()
#             optimizer.step()
#
#             loss_data.append({
#                 "epoch": (epoch + 1),
#                 "l_data_loss": l_data.item(),
#                 "l_pde_loss": l_pde.item(),
#                 "loss_total": loss.item()
#             })
#
#             if epoch % 500 == 0:
#                 print(f"Epoch {epoch}: total={loss.item():.4e}, data={l_data.item():.4e}, pde={l_pde.item():.4e}")
#
#             if (epoch % video_freq == 0) or (epoch == n_epochs - 1):
#                 vid_name = f"{model_name}_epoch_{epoch}_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
#                 video_path = os.path.join(video_folder, vid_name)
#
#             if (epoch % 1501 == 0 and epoch > 0) or epoch == n_epochs - 1:
#                 checkpoint_path = os.path.join(checkpoint_dir, f"model_real_data_checkpoint_epoch_{epoch}.pt")
#                 torch.save(self.state_dict(), checkpoint_path)
#                 print(f"Checkpoint saved at {checkpoint_path}")
#
#         df = pd.DataFrame(loss_data)
#         df.to_csv(losses_file, index=False)
#         print(f"Training losses saved to {losses_file}")
#
#     def predict(self, x_star, y_star, t_star):
#         """Returns (A_r, A_i) in numpy arrays."""
#         self.eval()
#         with torch.no_grad():
#             A_r, A_i = self.net_A(x_star, y_star, t_star)
#         return A_r.cpu().numpy(), A_i.cpu().numpy()
#
#     def predict_myu_small(self):
#         """Return the current mu_small array as numpy."""
#         return self.mu_small.detach().cpu().numpy()
#
#     def expand_myu_full(self, do_binarize=True, scale_255=False):
#         """
#         Expand mu_small to full shape (Nt, Nx, Ny) by repeating each cell degrade_x, degrade_y times.
#         This can help with final visualization or 'export'.
#
#         do_binarize: whether to threshold to 0/1
#         scale_255:  whether to multiply 1->255.
#         """
#         with torch.no_grad():
#             mu_raw = self.mu_small.detach()  # shape (Nt, Nx_down, Ny_down)
#             if do_binarize:
#                 mu_bin = (mu_raw > 0.0).float()  # (Nt, Nx_down, Ny_down)
#             else:
#                 mu_bin = mu_raw
#
#             if scale_255:
#                 mu_bin = mu_bin*255.0
#
#             # Now expand in x,y by degrade_x, degrade_y
#             # shape after repeat_interleave => (Nt, Nx_down*degrade_x, Ny_down*degrade_y) = (Nt, Nx, Ny)
#             mu_full_x = mu_bin.repeat_interleave(self.degrade_x, dim=1)
#             mu_full_xy = mu_full_x.repeat_interleave(self.degrade_y, dim=2)
#
#             return mu_full_xy.cpu().numpy()  # shape (Nt, Nx, Ny)
# ################################################################### MODEL 2 ###################################################################
#
# class NOPINN(nn.Module):
#     def __init__(
#         self,
#         layers,  # [3, 64, 64, 2]
#         Nt, Nx_down, Ny_down,
#         dt, dx, dy,
#         degrade_x, degrade_y,
#         delta=1.0,
#         myu_init=None,  # shape: (Nt, Nx_down, Ny_down)
#         weight_pde=1.0,
#         dropout_rate=0.1,
#         device='cuda'
#     ):
#         super(NOPINN, self).__init__()
#         self.device = device
#         self.delta = delta
#         self.weight_pde = weight_pde
#
#         self.dnn = self._build_network(layers, dropout_rate).to(device)
#         self.Nt, self.Nx_down, self.Ny_down = Nt, Nx_down, Ny_down
#         self.dt, self.dx, self.dy = dt, dx, dy
#         self.degrade_x, self.degrade_y = degrade_x, degrade_y
#
#         if myu_init is None:
#             init = torch.zeros(Nt, Nx_down, Ny_down)
#         else:
#             init = torch.tensor(myu_init, dtype=torch.float32)
#         self.mu_small = nn.Parameter(init.to(device))
#
#     def _build_network(self, layers, dropout_rate):
#         net = []
#         for i in range(len(layers) - 1):
#             net.append(nn.Linear(layers[i], layers[i + 1]))
#             if i < len(layers) - 2:
#                 net.append(nn.SiLU())  # Swish activation
#                 net.append(nn.Dropout(dropout_rate))  # Dropout
#         return nn.Sequential(*net)
#
#     def net_A(self, x, y, t):
#         inp = torch.cat([x, y, t], dim=1)
#         out = self.dnn(inp)
#         return out[:, 0:1], out[:, 1:2]
#
#     def get_mu(self, x, y, t):
#         i = (t[:, 0] / self.dt).round().long().clamp(0, self.Nt - 1)
#         j = (x[:, 0] / (self.dx * self.degrade_x)).floor().long().clamp(0, self.Nx_down - 1)
#         k = (y[:, 0] / (self.dy * self.degrade_y)).floor().long().clamp(0, self.Ny_down - 1)
#         return self.mu_small[i, j, k].view(-1, 1)
#
#     def pde_residual(self, x, y, t):
#         A_r, A_i = self.net_A(x, y, t)
#         mu_vals = self.get_mu(x, y, t)
#
#         A_r_t = torch.autograd.grad(A_r, t, torch.ones_like(A_r), create_graph=True, retain_graph=True)[0]
#         A_i_t = torch.autograd.grad(A_i, t, torch.ones_like(A_i), create_graph=True, retain_graph=True)[0]
#
#         A_r_x = torch.autograd.grad(A_r, x, torch.ones_like(A_r), create_graph=True, retain_graph=True)[0]
#         A_i_x = torch.autograd.grad(A_i, x, torch.ones_like(A_i), create_graph=True, retain_graph=True)[0]
#
#         A_r_y = torch.autograd.grad(A_r, y, torch.ones_like(A_r), create_graph=True, retain_graph=True)[0]
#         A_i_y = torch.autograd.grad(A_i, y, torch.ones_like(A_i), create_graph=True, retain_graph=True)[0]
#
#         A_r_xx = torch.autograd.grad(A_r_x, x, torch.ones_like(A_r_x), create_graph=True, retain_graph=True)[0]
#         A_r_yy = torch.autograd.grad(A_r_y, y, torch.ones_like(A_r_y), create_graph=True, retain_graph=True)[0]
#
#         A_i_xx = torch.autograd.grad(A_i_x, x, torch.ones_like(A_i_x), create_graph=True, retain_graph=True)[0]
#         A_i_yy = torch.autograd.grad(A_i_y, y, torch.ones_like(A_i_y), create_graph=True, retain_graph=True)[0]
#
#         lapA_r = A_r_xx + A_r_yy
#         lapA_i = A_i_xx + A_i_yy
#
#         A_abs2 = A_r**2 + A_i**2
#         f_r = A_r_t - mu_vals * A_r - self.delta * lapA_r + A_abs2 * A_r
#         f_i = A_i_t - mu_vals * A_i - self.delta * lapA_i + A_abs2 * A_i
#
#         return f_r, f_i
#
#     def loss_pde(self, x_eqs, y_eqs, t_eqs):
#         f_r, f_i = self.pde_residual(x_eqs, y_eqs, t_eqs)
#         return torch.mean(f_r**2 + f_i**2)
#
#     def loss_data(self, x_data, y_data, t_data, A_r_data, A_i_data):
#         A_r_pred, A_i_pred = self.net_A(x_data, y_data, t_data)
#         return torch.mean((A_r_pred - A_r_data)**2 + (A_i_pred - A_i_data)**2)
#
#     def train_model(
#         self,
#         x_data, y_data, t_data, A_r_data, A_i_data,
#         x_eqs, y_eqs, t_eqs,
#         n_epochs=3000, lr=1e-3, weight_decay=1e-4, clip_value=1.0
#     ):
#         optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=200, min_lr=1e-6)
#         for epoch in range(n_epochs):
#             optimizer.zero_grad()
#             l_pde = self.loss_pde(x_eqs, y_eqs, t_eqs)
#             l_data = self.loss_data(x_data, y_data, t_data, A_r_data, A_i_data)
#             loss = l_data + self.weight_pde * l_pde
#             loss.backward()
#             nn.utils.clip_grad_norm_(self.parameters(), clip_value)
#             optimizer.step()
#
#             scheduler.step(loss)
#
#             if epoch % 500 == 0:
#                 print(f"Epoch {epoch}: total={loss.item():.4e}, data={l_data.item():.4e}, pde={l_pde.item():.4e}")
#
#     def predict(self, x_star, y_star, t_star):
#         self.eval()
#         with torch.no_grad():
#             A_r, A_i = self.net_A(x_star, y_star, t_star)
#         return A_r.cpu().numpy(), A_i.cpu().numpy()
#
#     def predict_myu_small(self):
#         return self.mu_small.detach().cpu().numpy()
#
# ################################################################### MODEL 3 ###################################################################
#
# class DNN3(nn.Module):
#     """Simple MLP that maps (x,y,t)->(A_r, A_i)."""
#     def __init__(self, layers):
#         super().__init__()
#         net = []
#         for i in range(len(layers)-1):
#             net.append(nn.Linear(layers[i], layers[i+1]))
#             if i < len(layers)-2:
#                 net.append(nn.Tanh())
#         self.net = nn.Sequential(*net)
#
#     def forward(self, x):
#         return self.net(x)
#
# class NPINN_PRO_MAX(nn.Module):
#     """
#     A 'PINN' for the Complex Ginzburg-Landau equation, where:
#       - A(x,y,t) is predicted by a small neural net.
#       - mu_small is a time-dependent 2D field in shape (Nt, Nx_down, Ny_down).
#       - We interpret each mu_small[i,j,k] as {0 or 1}, i.e. a binary value.
#       - We do 'collocation' lookups: (x,y,t) -> (i_down,j_down) -> mu_small[i, j_down, k_down].
#     """
#
#     def __init__(
#         self,
#         layers,               # e.g. [3,64,64,2] for A_r, A_i
#         Nt, Nx, Ny,           # full domain sizes
#         Nx_down, Ny_down,     # smaller, downsampled domain for mu
#         dt, dx, dy,
#         degrade_x, degrade_y,
#         delta=0.01,
#         weight_pde=1.0,
#         device='cpu'
#     ):
#         super().__init__()
#         self.device = device
#         self.delta = delta
#         self.weight_pde = weight_pde
#
#         # Basic domain sizes
#         self.Nt, self.Nx, self.Ny = Nt, Nx, Ny
#         self.Nx_down, self.Ny_down = Nx_down, Ny_down
#         self.dt, self.dx, self.dy = dt, dx, dy
#         self.degrade_x, self.degrade_y = degrade_x, degrade_y
#
#         # 1) The neural net for A(x,y,t)
#         self.dnn = DNN3(layers).to(device)
#
#         # 2) The trainable mu_small: shape (Nt, Nx_down, Ny_down)
#         #    We'll init randomly near 0..1, or zeros, etc.
#         init = 0.3*torch.randn(Nt, Nx_down, Ny_down)  # Some small random
#         self.mu_small_raw = nn.Parameter(init.to(device))
#         # We'll interpret this as a "logit," or just clamp each iteration, or do a step function in get_myu.
#
#     def forward(self, x, y, t):
#         """Alias so we can do model(x,y,t) -> (A_r, A_i)."""
#         return self.net_A(x, y, t)
#
#     def net_A(self, x, y, t):
#         """Forward pass for A_r, A_i."""
#         inp = torch.cat([x,y,t], dim=1)   # shape (batch,3)
#         out = self.dnn(inp)              # shape (batch,2)
#         A_r = out[:, 0:1]
#         A_i = out[:, 1:2]
#         return A_r, A_i
#
#     def binarize_mu_small(self):
#         """
#         In practice, you might do this once per epoch, or only after training,
#         because a hard step breaks gradient flow.
#         This forcibly thresholds mu_small_raw -> 0 or 1 in place.
#         """
#         with torch.no_grad():
#             self.mu_small_raw.data = (self.mu_small_raw.data > 0.0).float()
#
#     def get_myu_collocation(self, x, y, t):
#         """
#         For PDE collocation: (x, y, t) -> integer indices (i, j, k).
#         Then interpret mu_small_raw[i,j,k] as binary => 0 or 1,
#         and if 1 => 255 if you want. Let's do 1 => 1 for PDE,
#         you can scale if needed.
#         """
#         # 1) Convert time to index i
#         i = (t[:,0] / self.dt).round().long().clamp(0, self.Nt-1)
#
#         # 2) Convert (x,y) to (j_down, k_down)
#         j_down = (x[:,0] / (self.dx*self.degrade_x)).floor().long()
#         k_down = (y[:,0] / (self.dy*self.degrade_y)).floor().long()
#
#         j_down = j_down.clamp(0, self.Nx_down-1)
#         k_down = k_down.clamp(0, self.Ny_down-1)
#
#         # 3) Access raw param -> threshold to 0/1
#         #    We'll do a step function in the forward pass:
#         mu_vals_raw = self.mu_small_raw[i, j_down, k_down]  # shape (batch,)
#
#         # step function => 0 or 1
#         mu_bin = (mu_vals_raw > 0.0).float()
#
#         return mu_bin.view(-1,1)
#
#     def pde_residual(self, x, y, t):
#         """
#         PDE: A_t = mu*A + delta Lap(A) - |A|^2 A
#         where mu = get_myu_collocation(x,y,t).
#         """
#         A_r, A_i = self.net_A(x,y,t)
#         mu_vals = self.get_myu_collocation(x,y,t)  # shape (batch,1)
#
#         # partial derivatives wrt t
#         A_r_t = torch.autograd.grad(A_r, t,
#             grad_outputs=torch.ones_like(A_r),
#             create_graph=True, retain_graph=True)[0]
#         A_i_t = torch.autograd.grad(A_i, t,
#             grad_outputs=torch.ones_like(A_i),
#             create_graph=True, retain_graph=True)[0]
#
#         # wrt x
#         A_r_x = torch.autograd.grad(A_r, x,
#             grad_outputs=torch.ones_like(A_r),
#             create_graph=True, retain_graph=True)[0]
#         A_i_x = torch.autograd.grad(A_i, x,
#             grad_outputs=torch.ones_like(A_i),
#             create_graph=True, retain_graph=True)[0]
#
#         # wrt y
#         A_r_y = torch.autograd.grad(A_r, y,
#             grad_outputs=torch.ones_like(A_r),
#             create_graph=True, retain_graph=True)[0]
#         A_i_y = torch.autograd.grad(A_i, y,
#             grad_outputs=torch.ones_like(A_i),
#             create_graph=True, retain_graph=True)[0]
#
#         # second derivatives -> laplacian
#         A_r_xx = torch.autograd.grad(A_r_x, x,
#             grad_outputs=torch.ones_like(A_r_x),
#             create_graph=True, retain_graph=True)[0]
#         A_r_yy = torch.autograd.grad(A_r_y, y,
#             grad_outputs=torch.ones_like(A_r_y),
#             create_graph=True, retain_graph=True)[0]
#
#         A_i_xx = torch.autograd.grad(A_i_x, x,
#             grad_outputs=torch.ones_like(A_i_x),
#             create_graph=True, retain_graph=True)[0]
#         A_i_yy = torch.autograd.grad(A_i_y, y,
#             grad_outputs=torch.ones_like(A_i_y),
#             create_graph=True, retain_graph=True)[0]
#
#         lapA_r = A_r_xx + A_r_yy
#         lapA_i = A_i_xx + A_i_yy
#
#         A_abs2 = A_r**2 + A_i**2
#
#         # Residual
#         f_r = A_r_t - mu_vals*A_r - self.delta*lapA_r + A_abs2*A_r
#         f_i = A_i_t - mu_vals*A_i - self.delta*lapA_i + A_abs2*A_i
#         return f_r, f_i
#
#     def loss_pde(self, x_eqs, y_eqs, t_eqs):
#         f_r, f_i = self.pde_residual(x_eqs, y_eqs, t_eqs)
#         return torch.mean(f_r**2 + f_i**2)
#
#     def loss_data(self, x_data, y_data, t_data, A_r_data, A_i_data):
#         A_r_pred, A_i_pred = self.net_A(x_data, y_data, t_data)
#         return torch.mean((A_r_pred - A_r_data)**2 + (A_i_pred - A_i_data)**2)
#
#     def train_model(
#         self,
#         x_data, y_data, t_data, A_r_data, A_i_data,
#         x_eqs, y_eqs, t_eqs,
#         n_epochs=3000,
#         lr=1e-3,
#         video_freq=2500,
#         state_exp=None,
#         myu_full_exp=None,
#         x_vals=None,
#         y_vals=None,
#         t_vals=None,
#         model_name="MyModel",
#         clip_value=1.0
#     ):
#         class_name = self.__class__.__name__
#         model_folder = f"./final_results/model_2_{class_name}"
#         os.makedirs(model_folder, exist_ok=True)
#
#         losses_file = os.path.join(model_folder, "losses.csv")
#         checkpoint_dir = model_folder
#         video_folder = os.path.join(model_folder, "videos")
#         os.makedirs(video_folder, exist_ok=True)
#
#         optimizer = optim.Adam(self.parameters(), lr=lr)
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=200, min_lr=1e-6)
#         loss_data = []
#
#         for epoch in range(n_epochs):
#             optimizer.zero_grad()
#             l_pde = self.loss_pde(x_eqs, y_eqs, t_eqs)
#             l_data = self.loss_data(x_data, y_data, t_data, A_r_data, A_i_data)
#             loss = l_data + self.weight_pde * l_pde
#             loss.backward()
#             nn.utils.clip_grad_norm_(self.parameters(), clip_value)
#             optimizer.step()
#             scheduler.step(loss)
#
#             loss_data.append({
#                 "epoch": (epoch + 1),
#                 "l_data_loss": l_data.item(),
#                 "l_pde_loss": l_pde.item(),
#                 "loss_total": loss.item()
#             })
#
#             if epoch % 500 == 0:
#                 print(f"Epoch {epoch}: total={loss.item():.4e}, data={l_data.item():.4e}, pde={l_pde.item():.4e}")
#
#             if (epoch % video_freq == 0) or (epoch == n_epochs - 1):
#                 vid_name = f"{model_name}_epoch_{epoch}_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
#                 video_path = os.path.join(video_folder, vid_name)
#                 generate_video(
#                     state=state_exp,
#                     mu_full=myu_full_exp,
#                     model=self,
#                     x_vals=x_vals,
#                     y_vals=y_vals,
#                     t_vals=t_vals,
#                     device=self.device,
#                     output_path=video_path,
#                 )
#
#             if (epoch % 1501 == 0 and epoch > 0) or epoch == n_epochs - 1:
#                 checkpoint_path = os.path.join(checkpoint_dir, f"model_real_data_checkpoint_epoch_{epoch}.pt")
#                 torch.save(self.state_dict(), checkpoint_path)
#                 print(f"Checkpoint saved at {checkpoint_path}")
#
#         df = pd.DataFrame(loss_data)
#         df.to_csv(losses_file, index=False)
#         print(f"Training losses saved to {losses_file}")
#
#     def expand_myu_full(self, do_binarize=True, scale_255=False):
#         """
#         Expand mu_small to full shape (Nt, Nx, Ny) by repeating each cell degrade_x, degrade_y times.
#         This can help with final visualization or 'export'.
#
#         do_binarize: whether to threshold to 0/1
#         scale_255:  whether to multiply 1->255.
#         """
#         with torch.no_grad():
#             mu_raw = self.mu_small_raw.detach()  # shape (Nt, Nx_down, Ny_down)
#             if do_binarize:
#                 mu_bin = (mu_raw > 0.0).float()  # (Nt, Nx_down, Ny_down)
#             else:
#                 mu_bin = mu_raw
#
#             if scale_255:
#                 mu_bin = mu_bin*255.0
#
#             # Now expand in x,y by degrade_x, degrade_y
#             # shape after repeat_interleave => (Nt, Nx_down*degrade_x, Ny_down*degrade_y) = (Nt, Nx, Ny)
#             mu_full_x = mu_bin.repeat_interleave(self.degrade_x, dim=1)
#             mu_full_xy = mu_full_x.repeat_interleave(self.degrade_y, dim=2)
#
#             return mu_full_xy.cpu().numpy()  # shape (Nt, Nx, Ny)
#
#     def predict(self, x, y, t):
#         """
#         Evaluate the neural net for A(x,y,t) -> (A_r, A_i) in NumPy form.
#         """
#         self.eval()
#         with torch.no_grad():
#             A_r, A_i = self.net_A(x, y, t)  # Tensors
#         return A_r.cpu().numpy(), A_i.cpu().numpy()
#
# ################################################################### MODEL 4 ###################################################################
#
# class DNN4(nn.Module):
#     """
#     Simple MLP that maps (x,y,t)->(A_r, A_i).
#     This is the same as before; you can incorporate different activation
#     or layer structures as needed.
#     """
#     def __init__(self, layers):
#         super().__init__()
#         modules = []
#         for i in range(len(layers)-1):
#             modules.append(nn.Linear(layers[i], layers[i+1]))
#             if i < len(layers)-2:
#                 modules.append(nn.Tanh())
#         self.net = nn.Sequential(*modules)
#
#     def forward(self, x):
#         return self.net(x)
#
#
# class NPINN_PRO_MAX_CONTINUOUSMU(nn.Module):
#     """
#     A 'PINN' for the Complex Ginzburg-Landau equation, where:
#       - A(x,y,t) is predicted by a small neural net (self.dnn).
#       - mu_small_raw is a time-dependent 2D field in shape (Nt, Nx_down, Ny_down),
#         but we do NOT binarize mu. It's fully continuous.
#
#     PDE:
#        A_t = mu A + delta Lap(A) - |A|^2 A
#
#     The shape is:
#       mu_small_raw: (Nt, Nx_down, Ny_down)
#     We do a cell-based lookup:
#       i = round(t/dt), j=(x/(dx*degrade_x)), k=(y/(dy*degrade_y))
#     Then PDE sees that continuous value.
#
#     The function expand_myu_full(...) can still be used to upsample
#     the mu_small to (Nt, Nx, Ny) for visualization or direct comparison.
#     """
#
#     def __init__(
#         self,
#         layers,               # e.g. [3,64,64,2] for A_r, A_i
#         Nt, Nx, Ny,           # full domain sizes
#         Nx_down, Ny_down,     # smaller, downsampled domain for mu
#         dt, dx, dy,
#         degrade_x, degrade_y,
#         delta=0.01,
#         weight_pde=1.0,
#         device='cpu'
#     ):
#         super().__init__()
#         self.device = device
#         self.delta = delta
#         self.weight_pde = weight_pde
#
#         # Basic domain sizes
#         self.Nt, self.Nx, self.Ny = Nt, Nx, Ny
#         self.Nx_down, self.Ny_down = Nx_down, Ny_down
#         self.dt, self.dx, self.dy = dt, dx, dy
#         self.degrade_x, self.degrade_y = degrade_x, degrade_y
#
#         # 1) The neural net for A(x,y,t)
#         if layers is not None:
#             self.dnn = DNN4(layers).to(device)
#         else:
#             self.dnn = None  # to be overridden externally if needed
#
#         # 2) The trainable mu_small: shape (Nt, Nx_down, Ny_down)
#         init = 0.3 * torch.randn(Nt, Nx_down, Ny_down)
#         self.mu_small_raw = nn.Parameter(init.to(device))
#
#     def forward(self, x, y, t):
#         """
#         For convenience, you can do model(x,y,t)->(A_r,A_i).
#         """
#         return self.net_A(x, y, t)
#
#     def net_A(self, x, y, t):
#         """Forward pass for A_r, A_i."""
#         if self.dnn is None:
#             raise ValueError("No DNN assigned! You must define self.dnn or pass layers != None.")
#         inp = torch.cat([x, y, t], dim=1)  # shape (batch,3)
#         out = self.dnn(inp)               # shape (batch,2)
#         A_r = out[:, 0:1]
#         A_i = out[:, 1:2]
#         return A_r, A_i
#
#     def get_myu_collocation(self, x, y, t):
#         """
#         For PDE collocation: (x, y, t) -> integer indices (i, j_down, k_down).
#         Return mu_small_raw[i,j_down,k_down] as a continuous scalar (no binarize).
#         """
#         i = (t[:,0] / self.dt).round().long().clamp(0, self.Nt-1)
#         j_down = (x[:,0] / (self.dx*self.degrade_x)).floor().long()
#         k_down = (y[:,0] / (self.dy*self.degrade_y)).floor().long()
#
#         j_down = j_down.clamp(0, self.Nx_down-1)
#         k_down = k_down.clamp(0, self.Ny_down-1)
#
#         mu_vals_raw = self.mu_small_raw[i, j_down, k_down]
#         # No threshold, purely continuous
#         return mu_vals_raw.view(-1,1)
#
#     def pde_residual(self, x, y, t):
#         """
#         PDE: A_t = mu*A + delta Lap(A) - |A|^2 A
#         where mu = get_myu_collocation(x,y,t) (continuous).
#         """
#         A_r, A_i = self.net_A(x, y, t)
#         mu_vals = self.get_myu_collocation(x, y, t)  # shape (batch,1)
#
#         # partial derivatives wrt t
#         A_r_t = torch.autograd.grad(A_r, t,
#             grad_outputs=torch.ones_like(A_r),
#             create_graph=True, retain_graph=True)[0]
#         A_i_t = torch.autograd.grad(A_i, t,
#             grad_outputs=torch.ones_like(A_i),
#             create_graph=True, retain_graph=True)[0]
#
#         # wrt x
#         A_r_x = torch.autograd.grad(A_r, x,
#             grad_outputs=torch.ones_like(A_r),
#             create_graph=True, retain_graph=True)[0]
#         A_i_x = torch.autograd.grad(A_i, x,
#             grad_outputs=torch.ones_like(A_i),
#             create_graph=True, retain_graph=True)[0]
#
#         # wrt y
#         A_r_y = torch.autograd.grad(A_r, y,
#             grad_outputs=torch.ones_like(A_r),
#             create_graph=True, retain_graph=True)[0]
#         A_i_y = torch.autograd.grad(A_i, y,
#             grad_outputs=torch.ones_like(A_i),
#             create_graph=True, retain_graph=True)[0]
#
#         # second derivatives -> Laplacian
#         A_r_xx = torch.autograd.grad(A_r_x, x,
#             grad_outputs=torch.ones_like(A_r_x),
#             create_graph=True, retain_graph=True)[0]
#         A_r_yy = torch.autograd.grad(A_r_y, y,
#             grad_outputs=torch.ones_like(A_r_y),
#             create_graph=True, retain_graph=True)[0]
#
#         A_i_xx = torch.autograd.grad(A_i_x, x,
#             grad_outputs=torch.ones_like(A_i_x),
#             create_graph=True, retain_graph=True)[0]
#         A_i_yy = torch.autograd.grad(A_i_y, y,
#             grad_outputs=torch.ones_like(A_i_y),
#             create_graph=True, retain_graph=True)[0]
#
#         lapA_r = A_r_xx + A_r_yy
#         lapA_i = A_i_xx + A_i_yy
#
#         A_abs2 = A_r**2 + A_i**2
#
#         f_r = A_r_t - mu_vals*A_r - self.delta*lapA_r + A_abs2*A_r
#         f_i = A_i_t - mu_vals*A_i - self.delta*lapA_i + A_abs2*A_i
#         return f_r, f_i
#
#     def loss_pde(self, x_eqs, y_eqs, t_eqs):
#         f_r, f_i = self.pde_residual(x_eqs, y_eqs, t_eqs)
#         return torch.mean(f_r**2 + f_i**2)
#
#     def loss_data(self, x_data, y_data, t_data, A_r_data, A_i_data):
#         A_r_pred, A_i_pred = self.net_A(x_data, y_data, t_data)
#         return torch.mean((A_r_pred - A_r_data)**2 + (A_i_pred - A_i_data)**2)
#
#     def train_model(
#         self,
#         x_data, y_data, t_data, A_r_data, A_i_data,
#         x_eqs, y_eqs, t_eqs,
#         n_epochs=3000, lr=1e-3,
#         model_name="model",
#         output_dir="./results",
#         video_freq=10000,
#         state_exp=None,
#         myu_full_exp=None,
#         x_vals=None,
#         y_vals=None,
#         t_vals=None,
#         device='cpu'
#     ):
#
#         optimizer = torch.optim.Adam(self.parameters(), lr=lr)
#
#         # Make a folder for logs
#         model_folder = os.path.join(output_dir, model_name)
#         os.makedirs(model_folder, exist_ok=True)
#
#         for epoch in range(n_epochs):
#             optimizer.zero_grad()
#             l_pde  = self.loss_pde(x_eqs, y_eqs, t_eqs)
#             l_data = self.loss_data(x_data, y_data, t_data, A_r_data, A_i_data)
#             loss   = l_data + self.weight_pde*l_pde
#
#             loss.backward()
#             optimizer.step()
#
#             if epoch % 500 == 0:
#                 print(f"Epoch={epoch}, total={loss.item():.4e}, data={l_data.item():.4e}, PDE={l_pde.item():.4e}")
#
#             # checkpoint + video if needed
#             if (epoch % video_freq == 0 and epoch>0):
#                 ckpt_path = os.path.join(model_folder, f"{model_name}_epoch_{epoch}.pt")
#                 torch.save(self.state_dict(), ckpt_path)
#                 print(f"Checkpoint saved at {ckpt_path}")
#
#                 # If you want to generate a video comparing the model to real data
#                 if (state_exp is not None) and (myu_full_exp is not None) and (x_vals is not None) \
#                    and (y_vals is not None) and (t_vals is not None):
#                     # function generate_video(...) if you want
#                     vid_name = f"{model_name}_epoch_{epoch}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
#                     video_folder = os.path.join(model_folder, "videos")
#                     os.makedirs(video_folder, exist_ok=True)
#
#                     video_path = os.path.join(video_folder, vid_name)
#
#                     generate_video(
#                         state_exp,
#                         myu_full_exp,
#                         self,  # pass the model
#                         x_vals, y_vals, t_vals,
#                         device=device,
#                         output_path=video_path
#                     )
#
#         # final checkpoint
#         final_ckpt = os.path.join(model_folder, f"{model_name}_final_{n_epochs}.pt")
#         torch.save(self.state_dict(), final_ckpt)
#         print(f"Final checkpoint saved at {final_ckpt}\nTraining done.\n")
#
#
#     def predict(self, x, y, t):
#         """
#         Evaluate the net for A(x,y,t) in eval mode, returning (A_r, A_i) as np arrays.
#         """
#         self.eval()
#         with torch.no_grad():
#             A_r, A_i = self.net_A(x, y, t)
#         return A_r.cpu().numpy(), A_i.cpu().numpy()
#
#     def expand_myu_full(self, clamp_range=None, scale_255=False):
#         """
#         Expand mu_small_raw to shape (Nt, Nx, Ny) by repeating each cell degrade_x, degrade_y times.
#         If clamp_range is (low, high), we can clamp mu values between them.
#         If scale_255, multiply by 255 after clamp or raw usage.
#
#         Return shape = (Nt, Nx, Ny).
#         """
#         with torch.no_grad():
#             mu_raw = self.mu_small_raw.detach().clone()  # shape (Nt, Nx_down, Ny_down)
#
#             if clamp_range is not None:
#                 low, high = clamp_range
#                 mu_raw = torch.clamp(mu_raw, min=low, max=high)
#
#             # Expand x, y dims
#             mu_full_x = mu_raw.repeat_interleave(self.degrade_x, dim=1)
#             mu_full_xy = mu_full_x.repeat_interleave(self.degrade_y, dim=2)
#
#             if scale_255:
#                 mu_full_xy = mu_full_xy * 255.0
#
#             return mu_full_xy.cpu().numpy()
#
# ################################################################### MODEL 5 ###################################################################
# class DNN5(nn.Module):
#     def __init__(self, layers):
#         super().__init__()
#         modules = []
#         for i in range(len(layers)-1):
#             modules.append(nn.Linear(layers[i], layers[i+1]))
#             if i < len(layers)-2:
#                 modules.append(nn.Tanh())
#         self.net = nn.Sequential(*modules)
#
#     def forward(self, x):
#         return self.net(x)
#
# class NPINN_PRO_MAX_TIMEBLOCK(nn.Module):
#     """
#     A 'PINN' for the Complex Ginzburg-Landau equation, with both
#     - time downsampling: degrade_t
#     - x,y downsampling:  degrade_x, degrade_y
#     - We interpret each mu_small_raw in shape (Nt_down, Nx_down, Ny_down).
#
#     PDE: A_t = mu A + delta Lap(A) - |A|^2 A
#     where mu is a BINARY field => 0 or 1, but stored as raw -> threshold in get_myu_collocation.
#
#     'Time blocks': each coarse time index covers degrade_t frames in the full domain.
#     """
#
#     def __init__(
#         self,
#         layers,               # e.g. [3,64,64,2] for A_r, A_i
#         Nt, Nx, Ny,           # full domain sizes
#         Nx_down, Ny_down,     # smaller, downsampled domain for mu in x,y
#         dt, dx, dy,
#         degrade_x, degrade_y,
#         degrade_t,            # <--- NEW: factor for time downsampling
#         delta=0.01,
#         weight_pde=1.0,
#         device='cpu'
#     ):
#         super().__init__()
#         self.device = device
#         self.delta  = delta
#         self.weight_pde = weight_pde
#
#         # Full domain
#         self.Nt, self.Nx, self.Ny = Nt, Nx, Ny
#         self.Nx_down, self.Ny_down = Nx_down, Ny_down
#         self.dt, self.dx, self.dy = dt, dx, dy
#         self.degrade_x, self.degrade_y = degrade_x, degrade_y
#         self.degrade_t = degrade_t
#
#         # The reduced domain size in time
#         # we assume Nt is divisible by degrade_t for simplicity
#         self.Nt_down = Nt // degrade_t
#
#         # 1) The neural net for A(x,y,t)
#         self.dnn = DNN5(layers).to(device)
#
#         # 2) The trainable mu_small: shape (Nt_down, Nx_down, Ny_down).
#         init = 0.3 * torch.randn(self.Nt_down, Nx_down, Ny_down)
#         self.mu_small_raw = nn.Parameter(init.to(device))
#
#     def forward(self, x, y, t):
#         return self.net_A(x, y, t)
#
#     def net_A(self, x, y, t):
#         inp = torch.cat([x,y,t], dim=1)
#         out = self.dnn(inp)
#         A_r = out[:,0:1]
#         A_i = out[:,1:2]
#         return A_r, A_i
#
#     def binarize_mu_small(self):
#         """
#         Hard threshold the entire mu_small_raw -> 0 or 1 in place.
#         This is optional and breaks gradient flow.
#         """
#         with torch.no_grad():
#             self.mu_small_raw.data = (self.mu_small_raw.data > 0.0).float()
#
#     def get_myu_collocation(self, x, y, t):
#         """
#         (x,y,t) -> integer indices (i, j_down, k_down).
#         But for time, we do i = floor(t/dt), then i_down = floor(i/degrade_t).
#         Then threshold to 0/1.
#         """
#         # Convert t-> i in [0..Nt-1]
#         i = (t[:,0] / self.dt).round().long().clamp(0, self.Nt-1)
#         # Then the coarse time index
#         i_down = (i // self.degrade_t).clamp(0, self.Nt_down-1)
#
#         j_down = (x[:,0] / (self.dx*self.degrade_x)).floor().long()
#         k_down = (y[:,0] / (self.dy*self.degrade_y)).floor().long()
#
#         j_down = j_down.clamp(0, self.Nx_down-1)
#         k_down = k_down.clamp(0, self.Ny_down-1)
#
#         mu_vals_raw = self.mu_small_raw[i_down, j_down, k_down]
#         # Binarize for PDE
#         mu_bin = (mu_vals_raw > 0.0).float()  # shape (batch,)
#         return mu_bin.view(-1,1)
#
#     def pde_residual(self, x, y, t):
#         A_r, A_i = self.net_A(x,y,t)
#         mu_vals = self.get_myu_collocation(x,y,t)
#
#         A_r_t = torch.autograd.grad(A_r, t,
#             grad_outputs=torch.ones_like(A_r),
#             create_graph=True, retain_graph=True)[0]
#         A_i_t = torch.autograd.grad(A_i, t,
#             grad_outputs=torch.ones_like(A_i),
#             create_graph=True, retain_graph=True)[0]
#
#         # wrt x
#         A_r_x = torch.autograd.grad(A_r, x,
#             grad_outputs=torch.ones_like(A_r),
#             create_graph=True, retain_graph=True)[0]
#         A_i_x = torch.autograd.grad(A_i, x,
#             grad_outputs=torch.ones_like(A_i),
#             create_graph=True, retain_graph=True)[0]
#
#         # wrt y
#         A_r_y = torch.autograd.grad(A_r, y,
#             grad_outputs=torch.ones_like(A_r),
#             create_graph=True, retain_graph=True)[0]
#         A_i_y = torch.autograd.grad(A_i, y,
#             grad_outputs=torch.ones_like(A_i),
#             create_graph=True, retain_graph=True)[0]
#
#         # second derivatives
#         A_r_xx = torch.autograd.grad(A_r_x, x,
#             grad_outputs=torch.ones_like(A_r_x),
#             create_graph=True, retain_graph=True)[0]
#         A_r_yy = torch.autograd.grad(A_r_y, y,
#             grad_outputs=torch.ones_like(A_r_y),
#             create_graph=True, retain_graph=True)[0]
#
#         A_i_xx = torch.autograd.grad(A_i_x, x,
#             grad_outputs=torch.ones_like(A_i_x),
#             create_graph=True, retain_graph=True)[0]
#         A_i_yy = torch.autograd.grad(A_i_y, y,
#             grad_outputs=torch.ones_like(A_i_y),
#             create_graph=True, retain_graph=True)[0]
#
#         lapA_r = A_r_xx + A_r_yy
#         lapA_i = A_i_xx + A_i_yy
#
#         A_abs2 = A_r**2 + A_i**2
#
#         f_r = A_r_t - mu_vals*A_r - self.delta*lapA_r + A_abs2*A_r
#         f_i = A_i_t - mu_vals*A_i - self.delta*lapA_i + A_abs2*A_i
#         return f_r, f_i
#
#     def loss_pde(self, x_eqs, y_eqs, t_eqs):
#         f_r, f_i = self.pde_residual(x_eqs, y_eqs, t_eqs)
#         return torch.mean(f_r**2 + f_i**2)
#
#     def loss_data(self, x_data, y_data, t_data, A_r_data, A_i_data):
#         A_r_pred, A_i_pred = self.net_A(x_data, y_data, t_data)
#         return torch.mean((A_r_pred - A_r_data)**2 + (A_i_pred - A_i_data)**2)
#
#     def train_model(
#         self,
#         x_data, y_data, t_data, A_r_data, A_i_data,
#         x_eqs, y_eqs, t_eqs,
#         n_epochs=200000,
#         lr=1e-3,
#         video_freq=10000,
#         state_exp=None,
#         myu_full_exp=None,
#         x_vals=None,
#         y_vals=None,
#         t_vals=None,
#         device="cpu"
#     ):
#
#         class_name = self.__class__.__name__
#         model_folder = f"./final_results/model_1_{class_name}"
#         os.makedirs(model_folder, exist_ok=True)
#
#         losses_file = os.path.join(model_folder, "losses.csv")
#         checkpoint_dir = model_folder
#         video_folder = os.path.join(model_folder, "videos")
#         os.makedirs(video_folder, exist_ok=True)
#
#         optimizer = optim.Adam(self.parameters(), lr=lr)
#         loss_data = []
#
#         for epoch in range(n_epochs):
#             optimizer.zero_grad()
#             l_pde = self.loss_pde(x_eqs, y_eqs, t_eqs)
#             l_data = self.loss_data(x_data, y_data, t_data, A_r_data, A_i_data)
#             loss = l_data + self.weight_pde * l_pde
#             loss.backward()
#             optimizer.step()
#
#             loss_data.append({
#                 "epoch": (epoch + 1),
#                 "l_data_loss": l_data.item(),
#                 "l_pde_loss": l_pde.item(),
#                 "loss_total": loss.item()
#             })
#
#             if epoch % 500 == 0:
#                 print(f"Epoch={epoch}, total={loss.item():.4e}, data={l_data.item():.4e}, PDE={l_pde.item():.4e}")
#
#             if (epoch % video_freq == 0 and epoch > 0) or (epoch == n_epochs - 1):
#                 vid_name = f"{class_name}_epoch_{epoch}_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
#                 video_path = os.path.join(video_folder, vid_name)
#                 generate_video(
#                     state_exp,
#                     myu_full_exp,
#                     self,
#                     x_vals, y_vals, t_vals,
#                     device=device,
#                     output_path=video_path
#                 )
#
#             if (epoch % video_freq == 0 and epoch > 0) or epoch == n_epochs - 1:
#                 checkpoint_path = os.path.join(checkpoint_dir, f"model_real_data_checkpoint_epoch_{epoch}.pt")
#                 torch.save(self.state_dict(), checkpoint_path)
#                 print(f"Checkpoint saved at {checkpoint_path}")
#
#         df = pd.DataFrame(loss_data)
#         df.to_csv(losses_file, index=False)
#         print(f"Training losses saved to {losses_file}\nFinal checkpoint saved.\nTraining done.")
#
#     def expand_myu_full(self, do_binarize=True, scale_255=False):
#         """
#         Expand mu_small_raw shape = (Nt_down, Nx_down, Ny_down)
#         to full shape (Nt, Nx, Ny) by:
#          1) repeat_interleave along time dim by degrade_t
#          2) repeat_interleave along x,y dims by degrade_x, degrade_y
#         """
#         with torch.no_grad():
#             mu_raw = self.mu_small_raw.detach()  # shape (Nt_down, Nx_down, Ny_down)
#
#             if do_binarize:
#                 mu_bin = (mu_raw>0.0).float()
#             else:
#                 mu_bin = mu_raw
#
#             # time expansion
#             mu_time = mu_bin.repeat_interleave(self.degrade_t, dim=0)
#             # shape => (Nt_down*degrade_t, Nx_down, Ny_down) = (Nt, Nx_down, Ny_down)
#
#             # expand in x,y
#             mu_full_x = mu_time.repeat_interleave(self.degrade_x, dim=1)
#             mu_full_xy = mu_full_x.repeat_interleave(self.degrade_y, dim=2)
#
#             if scale_255:
#                 mu_full_xy = mu_full_xy * 255.0
#
#             return mu_full_xy.cpu().numpy()
#
#     def predict(self, x, y, t):
#         self.eval()
#         with torch.no_grad():
#             A_r, A_i = self.net_A(x, y, t)
#         return A_r.cpu().numpy(), A_i.cpu().numpy()


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
            device="cpu",
            validation_split=0.2,  # 20% of data used for validation
            val_freq=100  # Validate every 100 epochs
    ):
        import os
        from datetime import datetime
        from torch.utils.data import TensorDataset, DataLoader, random_split

        # Create complete dataset for training data
        full_dataset = TensorDataset(x_data, y_data, t_data, A_r_data, A_i_data)

        # Split into training and validation sets
        val_size = int(len(full_dataset) * validation_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Handle collocation points (PDE equation points)
        # We need to handle these differently since they need requires_grad=True

        # Create tensors without requires_grad for storage
        x_eqs_np = x_eqs.detach().cpu().numpy()
        y_eqs_np = y_eqs.detach().cpu().numpy()
        t_eqs_np = t_eqs.detach().cpu().numpy()

        # Split into train/val indices
        n_coll = len(x_eqs_np)
        indices = np.arange(n_coll)
        np.random.shuffle(indices)

        val_coll_size = int(n_coll * validation_split)
        train_coll_size = n_coll - val_coll_size

        train_indices = indices[:train_coll_size]
        val_indices = indices[train_coll_size:]

        # Prepare index dataloaders
        train_coll_loader = DataLoader(train_indices, batch_size=batch_size, shuffle=True)
        val_coll_loader = DataLoader(val_indices, batch_size=batch_size, shuffle=False)

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

        # Create folders for output
        model_folder = os.path.join(output_dir, model_name)
        os.makedirs(model_folder, exist_ok=True)

        # Initialize tracking variables
        best_loss = float('inf')
        patience_counter = 0
        patience = 10  # epochs for early stopping

        # Dictionary to store all loss metrics
        metrics = {
            'epoch': [],
            'train_total_loss': [],
            'train_data_loss': [],
            'train_pde_loss': [],
            'val_total_loss': [],
            'val_data_loss': [],
            'val_pde_loss': []
        }

        print(f"Starting training with {train_size} training samples and {val_size} validation samples")

        for epoch in range(n_epochs):
            # Training phase
            self.train()
            train_total_loss = 0
            train_data_loss = 0
            train_pde_loss = 0
            n_train_batches = 0

            for (x_d, y_d, t_d, ar_d, ai_d), coll_indices in zip(train_loader, train_coll_loader):
                n_train_batches += 1
                optimizer.zero_grad()

                # Move data tensors to device
                x_d, y_d, t_d = x_d.to(device), y_d.to(device), t_d.to(device)
                ar_d, ai_d = ar_d.to(device), ai_d.to(device)

                # Create collocation tensors with requires_grad=True
                x_e = torch.tensor(x_eqs_np[coll_indices], dtype=torch.float32, device=device, requires_grad=True)
                y_e = torch.tensor(y_eqs_np[coll_indices], dtype=torch.float32, device=device, requires_grad=True)
                t_e = torch.tensor(t_eqs_np[coll_indices], dtype=torch.float32, device=device, requires_grad=True)

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

                train_total_loss += loss.item()
                train_data_loss += data_loss.item()
                train_pde_loss += pde_loss.item()

            # Calculate average training losses
            train_total_loss /= n_train_batches
            train_data_loss /= n_train_batches
            train_pde_loss /= n_train_batches

            # Validation phase (every val_freq epochs)
            if epoch % val_freq == 0 or epoch == n_epochs - 1:
                self.eval()
                val_total_loss = 0
                val_data_loss = 0
                val_pde_loss = 0
                n_val_batches = 0

                with torch.no_grad():
                    for (x_d, y_d, t_d, ar_d, ai_d), coll_indices in zip(val_loader, val_coll_loader):
                        n_val_batches += 1

                        # Move data tensors to device
                        x_d, y_d, t_d = x_d.to(device), y_d.to(device), t_d.to(device)
                        ar_d, ai_d = ar_d.to(device), ai_d.to(device)

                        # Data loss (doesn't need gradients)
                        data_loss = self.loss_data(x_d, y_d, t_d, ar_d, ai_d)

                        # For PDE loss, we need to temporarily exit no_grad context
                        # and create tensors with requires_grad=True
                        with torch.enable_grad():
                            # Create collocation tensors with requires_grad=True
                            x_e = torch.tensor(x_eqs_np[coll_indices], dtype=torch.float32, device=device,
                                               requires_grad=True)
                            y_e = torch.tensor(y_eqs_np[coll_indices], dtype=torch.float32, device=device,
                                               requires_grad=True)
                            t_e = torch.tensor(t_eqs_np[coll_indices], dtype=torch.float32, device=device,
                                               requires_grad=True)

                            # PDE loss
                            pde_loss = self.loss_pde(x_e, y_e, t_e)

                        # Total loss
                        loss = data_loss + self.weight_pde * pde_loss

                        val_total_loss += loss.item()
                        val_data_loss += data_loss.item()
                        val_pde_loss += pde_loss.item()

                # Calculate average validation losses
                val_total_loss /= n_val_batches
                val_data_loss /= n_val_batches
                val_pde_loss /= n_val_batches

                # Update metrics dictionary
                metrics['epoch'].append(epoch)
                metrics['train_total_loss'].append(train_total_loss)
                metrics['train_data_loss'].append(train_data_loss)
                metrics['train_pde_loss'].append(train_pde_loss)
                metrics['val_total_loss'].append(val_total_loss)
                metrics['val_data_loss'].append(val_data_loss)
                metrics['val_pde_loss'].append(val_pde_loss)

                # Early stopping check
                if val_total_loss < best_loss:
                    best_loss = val_total_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.state_dict(), os.path.join(model_folder, f"{model_name}_best.pt"))
                    print(f"Epoch {epoch}: New best model saved (val_loss={val_total_loss:.4e})")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered after {epoch} epochs")
                        break

                # Print progress
                print(f"Epoch {epoch}: "
                      f"Train [total={train_total_loss:.4e}, data={train_data_loss:.4e}, PDE={train_pde_loss:.4e}] | "
                      f"Val [total={val_total_loss:.4e}, data={val_data_loss:.4e}, PDE={val_pde_loss:.4e}]")

            # Regular checkpointing and video generation
            if (epoch % video_freq == 0 and epoch > 0) or epoch == n_epochs - 1:
                ckpt_path = os.path.join(model_folder, f"{model_name}_epoch_{epoch}.pt")
                torch.save(self.state_dict(), ckpt_path)
                print(f"Checkpoint saved at {ckpt_path}")

                # Video generation if data is provided
                if all(x is not None for x in [state_exp, myu_full_exp, x_vals, y_vals, t_vals]):
                    vid_name = f"{model_name}_epoch_{epoch}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    video_folder = os.path.join(model_folder, "videos")
                    os.makedirs(video_folder, exist_ok=True)
                    video_path = os.path.join(video_folder, vid_name)
                    try:
                        generate_video(state_exp, myu_full_exp, self, x_vals, y_vals, t_vals, device=device,
                                       output_path=video_path)
                    except Exception as e:
                        print(f"Warning: Video generation failed with error: {e}")

        # Save final model and metrics
        final_ckpt = os.path.join(model_folder, f"{model_name}_final_{n_epochs}.pt")
        torch.save(self.state_dict(), final_ckpt)

        # Save loss metrics to CSV
        import pandas as pd
        loss_df = pd.DataFrame(metrics)
        loss_csv_path = os.path.join(model_folder, f"{model_name}_losses.csv")
        loss_df.to_csv(loss_csv_path, index=False)

        # Generate validation error plots
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(16, 10))

            # Plot training and validation losses
            plt.subplot(2, 1, 1)
            plt.semilogy(metrics['epoch'], metrics['train_total_loss'], 'b-', label='Training Loss')
            plt.semilogy(metrics['epoch'], metrics['val_total_loss'], 'r-', label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Total Loss (log scale)')
            plt.legend()
            plt.grid(True)

            # Plot component losses
            plt.subplot(2, 1, 2)
            plt.semilogy(metrics['epoch'], metrics['train_data_loss'], 'b--', label='Train Data Loss')
            plt.semilogy(metrics['epoch'], metrics['train_pde_loss'], 'b:', label='Train PDE Loss')
            plt.semilogy(metrics['epoch'], metrics['val_data_loss'], 'r--', label='Val Data Loss')
            plt.semilogy(metrics['epoch'], metrics['val_pde_loss'], 'r:', label='Val PDE Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Component Losses (log scale)')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(model_folder, f"{model_name}_loss_curves.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Could not generate loss plots: {e}")

        print(f"Final checkpoint saved at {final_ckpt}")
        print(f"Training metrics saved to {loss_csv_path}")
        print("Training completed successfully!")



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