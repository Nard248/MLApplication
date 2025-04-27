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