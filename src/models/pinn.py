import torch
import torch.nn as nn
import torch.autograd as autograd

class PINN(nn.Module):
    def __init__(self, input_dim=3, hidden_layers=[128, 128, 128], output_dim=2):
        super(PINN, self).__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.Tanh())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def physics_loss(self, x, predictions):
        """Compute physics-based constraints."""
        t = x[:, 0:1]
        x_coord = x[:, 1:2]
        y_coord = x[:, 2:3]

        real = predictions[:, 0:1]
        imag = predictions[:, 1:2]

        real_grad = autograd.grad(
            outputs=real, inputs=x,
            grad_outputs=torch.ones_like(real),
            retain_graph=True, create_graph=True
        )[0]
        imag_grad = autograd.grad(
            outputs=imag, inputs=x,
            grad_outputs=torch.ones_like(imag),
            retain_graph=True, create_graph=True
        )[0]

        # Example physics constraints: Laplacian of the state
        laplacian_real = real_grad[:, 1]**2 + real_grad[:, 2]**2
        laplacian_imag = imag_grad[:, 1]**2 + imag_grad[:, 2]**2

        physics_loss = torch.mean((laplacian_real - real)**2 + (laplacian_imag - imag)**2)
        return physics_loss
