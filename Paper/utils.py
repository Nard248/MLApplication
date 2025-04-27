import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from datetime import datetime


def create_combined_frame(pinn_prod, original_prod, abs_diff_prod, mu_pred, mu_full, abs_diff_myu, mu_pred_original,
                          mu_full_original, abs_diff_myu_original, enable_vmin_vmax=False):
    """
    Create a combined visualization frame with 9 panels.

    Args:
        pinn_prod: PINN predicted real*imag
        original_prod: Original real*imag
        abs_diff_prod: Absolute difference of products
        mu_pred: Predicted μ (binarized)
        mu_full: Original μ (binarized)
        abs_diff_myu: Absolute difference of binarized μ
        mu_pred_original: Predicted μ (non-binarized)
        mu_full_original: Original μ (non-binarized)
        abs_diff_myu_original: Absolute difference of non-binarized μ
        enable_vmin_vmax: Whether to use consistent color scaling

    Returns:
        numpy array: RGB image of the frame
    """
    vmin, vmax = (np.min(original_prod), np.max(original_prod)) if enable_vmin_vmax else (None, None)

    fig = plt.figure(figsize=(20, 14))
    spec = gridspec.GridSpec(ncols=3, nrows=3, figure=fig, wspace=0.2, hspace=0.3)

    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1])
    ax3 = fig.add_subplot(spec[0, 2])
    ax4 = fig.add_subplot(spec[1, 0])
    ax5 = fig.add_subplot(spec[1, 1])
    ax6 = fig.add_subplot(spec[1, 2])
    ax7 = fig.add_subplot(spec[2, 0])
    ax8 = fig.add_subplot(spec[2, 1])
    ax9 = fig.add_subplot(spec[2, 2])

    im1 = ax1.imshow(original_prod, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
    ax1.set_title("Original: Real x Imag")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    im2 = ax2.imshow(pinn_prod, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
    ax2.set_title("PINN: Real x Imag")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    im3 = ax3.imshow(abs_diff_prod, cmap="viridis", origin="lower")
    ax3.set_title("Absolute Difference")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")

    im4 = ax4.imshow(mu_full, cmap="viridis", origin="lower")
    ax4.set_title("Original μ (binarized)")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")

    im5 = ax5.imshow(mu_pred, cmap="viridis", origin="lower")
    ax5.set_title("Predicted μ (binarized)")
    ax5.set_xlabel("X")
    ax5.set_ylabel("Y")

    im6 = ax6.imshow(abs_diff_myu, cmap="viridis", origin="lower")
    ax6.set_title("Absolute Difference (Binarized Case)")
    ax6.set_xlabel("X")
    ax6.set_ylabel("Y")

    im7 = ax7.imshow(mu_full_original, cmap="viridis", origin="lower")
    ax7.set_title("Original μ (non-binarized)")
    ax7.set_xlabel("X")
    ax7.set_ylabel("Y")

    im8 = ax8.imshow(mu_pred_original, cmap="viridis", origin="lower")
    ax8.set_title("Predicted μ (non-binarized)")
    ax8.set_xlabel("X")
    ax8.set_ylabel("Y")

    im9 = ax9.imshow(abs_diff_myu_original, cmap="viridis", origin="lower")
    ax9.set_title("Absolute Difference (Non-Binarized Case)")
    ax9.set_xlabel("X")
    ax9.set_ylabel("Y")

    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    fig.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
    fig.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
    fig.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)

    fig.tight_layout()

    # Fix for compatibility with different matplotlib versions
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()

    # Try multiple methods to get the image data
    try:
        # Method 1: Modern method (buffer_rgba)
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8').reshape(height, width, 4)
        image = image[:, :, :3]  # Remove alpha channel
    except (AttributeError, TypeError):
        try:
            # Method 2: Newer versions
            image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        except (AttributeError, TypeError):
            try:
                # Method 3: Older versions method
                image = np.array(fig.canvas.renderer._renderer)
            except (AttributeError, TypeError):
                # Method 4: Last resort
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                canvas = FigureCanvasAgg(fig)
                canvas.draw()
                image = np.array(canvas.buffer_rgba())[:, :, :3]

    plt.close(fig)
    return image

def create_video(output_path, video_file_name, pinn_prod_frames, original_prod_frames, abs_diff_prod_frames,
                 mu_pred_frames, mu_full_frames, abs_diff_myu_frames, mu_pred_original_frames, mu_full_original_frames,
                 abs_diff_myu_original_frames, fps=30, enable_vmin_vmax=False):
    """
    Create a video from the frame arrays.

    Args:
        output_path: Directory to save the video
        video_file_name: Name of the video file
        pinn_prod_frames: List of PINN predicted real*imag frames
        original_prod_frames: List of original real*imag frames
        abs_diff_prod_frames: List of absolute difference of products frames
        mu_pred_frames: List of predicted μ (binarized) frames
        mu_full_frames: List of original μ (binarized) frames
        abs_diff_myu_frames: List of absolute difference of binarized μ frames
        mu_pred_original_frames: List of predicted μ (non-binarized) frames
        mu_full_original_frames: List of original μ (non-binarized) frames
        abs_diff_myu_original_frames: List of absolute difference of non-binarized μ frames
        fps: Frames per second
        enable_vmin_vmax: Whether to use consistent color scaling
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    video_path = os.path.join(output_path,
                              video_file_name if video_file_name.endswith(".mp4") else f"{video_file_name}.mp4")

    first_frame = create_combined_frame(
        pinn_prod_frames[0], original_prod_frames[0], abs_diff_prod_frames[0],
        mu_pred_frames[0], mu_full_frames[0], abs_diff_myu_frames[0],
        mu_pred_original_frames[0], mu_full_original_frames[0], abs_diff_myu_original_frames[0],
        enable_vmin_vmax
    )
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    try:
        for i in tqdm(range(len(pinn_prod_frames)), desc="Creating video"):
            combined_frame = create_combined_frame(
                pinn_prod_frames[i], original_prod_frames[i], abs_diff_prod_frames[i],
                mu_pred_frames[i], mu_full_frames[i], abs_diff_myu_frames[i],
                mu_pred_original_frames[i], mu_full_original_frames[i], abs_diff_myu_original_frames[i],
                enable_vmin_vmax
            )
            video_writer.write(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))
        video_writer.release()
    except Exception as e:
        video_writer.release()
        raise RuntimeError(f"Failed to create video: {e}")

    print(f"Video saved at: {video_path}")


def generate_video(state, mu_full, mu_full_original, model, x_vals, y_vals, t_vals, device, output_path,
                   video_file_name, enable_vmin_vmax=False):
    """
    Generate video comparing model predictions with ground truth.
    Handles different shapes between mu_full and mu_full_original.

    Args:
        state: Ground truth complex field
        mu_full: Ground truth binarized μ field
        mu_full_original: Ground truth non-binarized μ field (can have different shape)
        model: Trained PINN model
        x_vals, y_vals, t_vals: Coordinate values
        device: Computation device (CPU/GPU)
        output_path: Directory to save the video
        video_file_name: Name of the video file
        enable_vmin_vmax: Whether to use consistent color scaling
    """
    import cv2

    # Initialize frame arrays
    pinn_prod_frames = []
    original_state_prod_frames = []
    abs_diff_prod_frames = []
    mu_pred_frames = []
    mu_full_frames = []
    abs_diff_myu_frames = []
    mu_pred_original_frames = []
    mu_full_original_frames = []
    abs_diff_myu_original_frames = []

    # Get expanded mu from model for binarized version (0 or 1)
    mu_expanded = model.expand_myu_full(do_binarize=True, scale_255=False)

    # Get expanded mu from model for non-binarized version
    mu_expanded_original = model.expand_myu_full(do_binarize=False, scale_255=False)

    # Check shapes and resize if needed for binarized version
    if mu_expanded.shape[1:] != mu_full.shape[1:]:
        print(f"Resizing mu_expanded from {mu_expanded.shape[1:]} to {mu_full.shape[1:]}")
        mu_expanded_resized = []
        for i in range(len(t_vals)):
            if i < mu_expanded.shape[0]:  # Check if we have this time frame
                # Use nearest neighbor interpolation to preserve binary values
                resized = cv2.resize(
                    mu_expanded[i],
                    (mu_full.shape[2], mu_full.shape[1]),
                    interpolation=cv2.INTER_NEAREST
                )
                mu_expanded_resized.append(resized)
            else:
                # If we don't have this time frame, use the last one
                mu_expanded_resized.append(mu_expanded_resized[-1])
        mu_expanded = np.array(mu_expanded_resized)

    # Similarly, check and resize original version to match mu_full_original shape
    if mu_expanded_original.shape[1:] != mu_full_original.shape[1:]:
        print(f"Resizing mu_expanded_original from {mu_expanded_original.shape[1:]} to {mu_full_original.shape[1:]}")
        mu_expanded_original_resized = []
        for i in range(len(t_vals)):
            if i < mu_expanded_original.shape[0]:  # Check if we have this time frame
                # Use bilinear interpolation for non-binary values
                resized = cv2.resize(
                    mu_expanded_original[i],
                    (mu_full_original.shape[2], mu_full_original.shape[1]),
                    interpolation=cv2.INTER_LINEAR
                )
                mu_expanded_original_resized.append(resized)
            else:
                # If we don't have this time frame, use the last one
                mu_expanded_original_resized.append(mu_expanded_original_resized[-1])
        mu_expanded_original = np.array(mu_expanded_original_resized)

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

        # Calculate absolute differences
        abs_diff_prod = np.abs(original_prod - pinn_prod)

        # For binarized mu comparison
        if i < mu_expanded.shape[0] and i < mu_full.shape[0]:
            abs_diff_myu = np.abs(mu_expanded[i] - mu_full[i])
        else:
            # Create an empty frame if we don't have this time index
            abs_diff_myu = np.zeros_like(mu_full[0])

        # For original mu comparison
        if i < mu_expanded_original.shape[0] and i < mu_full_original.shape[0]:
            abs_diff_myu_original = np.abs(mu_expanded_original[i] - mu_full_original[i])
        else:
            # Create an empty frame if we don't have this time index
            abs_diff_myu_original = np.zeros_like(mu_full_original[0])

        # Append to frame arrays
        pinn_prod_frames.append(pinn_prod)
        original_state_prod_frames.append(original_prod)
        abs_diff_prod_frames.append(abs_diff_prod)

        # For binarized mu
        if i < mu_expanded.shape[0]:
            mu_pred_frames.append(mu_expanded[i])
        else:
            mu_pred_frames.append(np.zeros_like(mu_full[0]))

        if i < mu_full.shape[0]:
            mu_full_frames.append(mu_full[i])
        else:
            mu_full_frames.append(np.zeros_like(mu_full[0]))

        abs_diff_myu_frames.append(abs_diff_myu)

        # For original mu
        if i < mu_expanded_original.shape[0]:
            mu_pred_original_frames.append(mu_expanded_original[i])
        else:
            mu_pred_original_frames.append(np.zeros_like(mu_full_original[0]))

        if i < mu_full_original.shape[0]:
            mu_full_original_frames.append(mu_full_original[i])
        else:
            mu_full_original_frames.append(np.zeros_like(mu_full_original[0]))

        abs_diff_myu_original_frames.append(abs_diff_myu_original)

    # Create video
    create_video(
        output_path,
        video_file_name,
        pinn_prod_frames,
        original_state_prod_frames,
        abs_diff_prod_frames,
        mu_pred_frames,
        mu_full_frames,
        abs_diff_myu_frames,
        mu_pred_original_frames,
        mu_full_original_frames,
        abs_diff_myu_original_frames,
        fps=30,
        enable_vmin_vmax=enable_vmin_vmax
    )