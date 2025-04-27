import torch.nn as nn
import torch.optim as optim
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from datetime import datetime
from tqdm import tqdm


def create_combined_frame(pinn_prod, original_prod, abs_diff, mu_pred, mu_full):
    """
    Create a combined frame for visualization with improved layout.
    Uses the layout structure from the second version but maintains
    the 5-panel format from the first version.
    """
    # Get min/max for consistent color scaling
    vmin = np.min(original_prod)
    vmax = np.max(original_prod)

    # Create figure with 2x3 grid layout
    fig = plt.figure(figsize=(18, 12))
    spec = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)

    # Original data
    ax1 = fig.add_subplot(spec[0, 0])
    im1 = ax1.imshow(original_prod, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
    ax1.set_title("Original: Real × Imag", fontsize=14)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # PINN prediction
    ax2 = fig.add_subplot(spec[0, 1])
    im2 = ax2.imshow(pinn_prod, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
    ax2.set_title("PINN: Real × Imag", fontsize=14)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Absolute difference
    ax3 = fig.add_subplot(spec[1, 0])
    im3 = ax3.imshow(abs_diff, cmap="hot", origin="lower")
    ax3.set_title("Absolute Difference", fontsize=14)
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Original μ
    ax4 = fig.add_subplot(spec[1, 1])
    im4 = ax4.imshow(mu_full, cmap="viridis", origin="lower")
    ax4.set_title("Original μ", fontsize=14)
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # Predicted μ
    ax5 = fig.add_subplot(spec[2, 0])
    im5 = ax5.imshow(mu_pred, cmap="viridis", origin="lower")
    ax5.set_title("Predicted μ", fontsize=14)
    ax5.set_xlabel("X")
    ax5.set_ylabel("Y")
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    # Add metrics panel - text summary of current frame
    ax6 = fig.add_subplot(spec[2, 1])
    ax6.axis('off')

    # Calculate metrics
    mse = np.mean((original_prod - pinn_prod) ** 2)
    mae = np.mean(np.abs(original_prod - pinn_prod))
    max_error = np.max(np.abs(original_prod - pinn_prod))

    # Display metrics
    metrics_text = (
        f"Frame Metrics:\n\n"
        f"MSE: {mse:.6f}\n"
        f"MAE: {mae:.6f}\n"
        f"Max Error: {max_error:.6f}\n\n"
        f"Original Range: [{np.min(original_prod):.3f}, {np.max(original_prod):.3f}]\n"
        f"Prediction Range: [{np.min(pinn_prod):.3f}, {np.max(pinn_prod):.3f}]"
    )
    ax6.text(0.1, 0.5, metrics_text, fontsize=12, va='center')

    # Finalize and convert to image
    fig.suptitle("PINNs Visualization Comparison", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for suptitle

    # Convert figure to image
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8').reshape(height, width, 4)
    plt.close(fig)
    return image[:, :, :3]  # Return RGB (drop alpha channel)


def create_video(output_path, pinn_prod_frames, original_prod_frames, abs_diff_frames, mu_pred_frames, mu_full_frames,
                 fps=30, additional_title=""):
    """
    Create a video from the frame arrays.
    Maintains the original function signature from the first version.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    video_path = os.path.join(output_path,
                              f"output_video_{datetime.now().strftime('%Y%m%d%H%M%S')}_{additional_title}.mp4")

    # Create first frame to get dimensions
    first_frame = create_combined_frame(
        pinn_prod_frames[0], original_prod_frames[0], abs_diff_frames[0], mu_pred_frames[0], mu_full_frames[0]
    )
    height, width, _ = first_frame.shape

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    try:
        # Process all frames
        for i in tqdm(range(len(pinn_prod_frames)), desc="Creating video"):
            combined_frame = create_combined_frame(
                pinn_prod_frames[i], original_prod_frames[i], abs_diff_frames[i], mu_pred_frames[i], mu_full_frames[i]
            )
            video_writer.write(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))
        video_writer.release()
    except Exception as e:
        video_writer.release()
        raise RuntimeError(f"Failed to create video: {e}")

    print(f"Video saved at: {video_path}")


def generate_video(state, mu_full, model, x_vals, y_vals, t_vals, device, output_path):
    """
    Generate video comparing model predictions with ground truth.
    Maintains the exact same function signature as the original version.
    """
    # Initialize frame arrays
    pinn_prod_frames = []
    original_state_prod_frames = []
    abs_diff_frames = []
    mu_pred_frames = []
    mu_full_frames = []

    # Get expanded mu from model
    mu_expanded = model.expand_myu_full(do_binarize=True, scale_255=True)

    # Generate frames for each timestep
    for i, t_val in enumerate(tqdm(t_vals, desc="Generating frames")):
        # Create meshgrid for evaluation
        X, Y = np.meshgrid(x_vals, y_vals)
        XX = X.ravel()
        YY = Y.ravel()
        TT = np.full_like(XX, t_val)

        # Convert to tensors
        x_test_t = torch.tensor(XX, dtype=torch.float32, device=device).view(-1, 1)
        y_test_t = torch.tensor(YY, dtype=torch.float32, device=device).view(-1, 1)
        t_test_t = torch.tensor(TT, dtype=torch.float32, device=device).view(-1, 1)

        # Get model predictions
        A_r_pred, A_i_pred = model.predict(x_test_t, y_test_t, t_test_t)
        A_r_pred_2d = A_r_pred.reshape(X.shape)
        A_i_pred_2d = A_i_pred.reshape(X.shape)

        # Calculate products
        pinn_prod = A_r_pred_2d * A_i_pred_2d
        original_prod = state[i].real * state[i].imag

        # Ensure consistent shapes
        if pinn_prod.shape != original_prod.shape:
            pinn_prod = pinn_prod.T

        # Calculate absolute difference
        abs_diff = np.abs(original_prod - pinn_prod)

        # Get mu values
        mu_pred_2d = mu_expanded[i]
        mu_full_2d = mu_full[i]

        # Append to frame arrays
        pinn_prod_frames.append(pinn_prod)
        original_state_prod_frames.append(original_prod)
        abs_diff_frames.append(abs_diff)
        mu_pred_frames.append(mu_pred_2d)
        mu_full_frames.append(mu_full_2d)

    # Convert lists to arrays
    pinn_prod_frames = np.array(pinn_prod_frames)
    original_state_prod_frames = np.array(original_state_prod_frames)
    abs_diff_frames = np.array(abs_diff_frames)
    mu_pred_frames = np.array(mu_pred_frames)
    mu_full_frames = np.array(mu_full_frames)

    # Create video
    create_video(output_path, pinn_prod_frames, original_state_prod_frames, abs_diff_frames, mu_pred_frames,
                 mu_full_frames, fps=30)

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