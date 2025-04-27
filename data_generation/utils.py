import os
import cv2
import torch
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tqdm import tqdm
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import warnings
warnings.filterwarnings('ignore')


def divide_frame_into_rectangles_with_means_and_binarization(
    data, 
    additional_data, 
    num_rects_x, 
    num_rects_y, 
    show_plot=False,
    threshold=0.5,
    use_mean_of_frame=True,
    use_smallest_frames_mean=True,
    crop=True
):
    if not np.iscomplexobj(additional_data):
        warnings.warn("The additional data is not of a complex type. Results might not be accurate.")

    frames_with_rects = []
    cropped_binarized_frames = []
    cropped_additional_frames = []

    min_height, min_width = float('inf'), float('inf')

    if use_mean_of_frame and use_smallest_frames_mean:
        smallest_mean_threshold = min(np.mean(frame) for frame in data)
    else:
        smallest_mean_threshold = None

    for frame_idx, (frame, additional_frame) in enumerate(zip(data, additional_data)):
        height, width = frame.shape
        rect_width = width // num_rects_x
        rect_height = height // num_rects_y

        binarized_frame = np.zeros_like(frame, dtype=np.uint8)
        annotated_frame = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        if use_mean_of_frame:
            frame_mean = smallest_mean_threshold if use_smallest_frames_mean else np.mean(frame)
        else:
            frame_mean = threshold

        start_col = width
        for row in range(height):
            white_pixels = np.where(frame[row] > np.mean(frame))[0]
            if white_pixels.size > 0:
                start_col = min(start_col, white_pixels[0])
        if start_col >= width:
            start_col = 0

        new_width = width - start_col
        new_num_rects_x = new_width // rect_width

        start_row = 0
        for col in range(width):
            white_pixels = np.where(frame[:, col] > np.mean(frame))[0]
            if white_pixels.size > 0:
                start_row = min(start_row, white_pixels[0])
        if start_row >= height:
            start_row = 0

        new_height = height - start_row
        new_num_rects_y = new_height // rect_height

        for i in range(new_num_rects_y):
            for j in range(new_num_rects_x):
                top_left = (start_col + j * rect_width, start_row + i * rect_height)
                bottom_right = (start_col + (j + 1) * rect_width, start_row + (i + 1) * rect_height)

                rect = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                mean_val = np.mean(rect)

                binarized_value = 1 if mean_val >= frame_mean else 0
                binarized_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = binarized_value

                cv2.rectangle(annotated_frame, top_left, bottom_right, (0, 255, 0), 2)
                text_position = (top_left[0] + 5, top_left[1] + 15)
                cv2.putText(annotated_frame, f"{mean_val:.2f}", text_position, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        if crop and frame.shape != additional_frame.shape:
            cropped_binarized = binarized_frame[start_row:start_row + new_num_rects_y * rect_height, 
                                                start_col:start_col + new_num_rects_x * rect_width]
            cropped_additional = additional_frame[start_row:start_row + new_num_rects_y * rect_height, 
                                                  start_col:start_col + new_num_rects_x * rect_width]
        else:
            cropped_binarized = binarized_frame
            cropped_additional = additional_frame

        min_height = min(min_height, cropped_binarized.shape[0])
        min_width = min(min_width, cropped_binarized.shape[1])

        cropped_binarized_frames.append(cropped_binarized)
        cropped_additional_frames.append(cropped_additional)
        frames_with_rects.append(annotated_frame)

        if show_plot:
            plt.figure(figsize=(30, 8))

            plt.subplot(1, 5, 1)
            plt.title(f"Original Frame {frame_idx + 1}")
            plt.imshow(frame, cmap="viridis")
            plt.axis("off")

            plt.subplot(1, 5, 2)
            plt.title(f"Annotated Frame {frame_idx + 1} (Grid + Means)")
            plt.imshow(annotated_frame)
            plt.axis("off")

            plt.subplot(1, 5, 3)
            plt.title(f"Binarized Frame {frame_idx + 1} (Threshold: {frame_mean:.2f})")
            plt.imshow(binarized_frame, cmap="gray")
            plt.axis("off")

            plt.subplot(1, 5, 4)
            plt.title(f"Cropped Region (Frame {frame_idx + 1})")
            plt.imshow(cropped_binarized, cmap="gray")
            plt.axis("off")

            plt.subplot(1, 5, 5)
            plt.title(f"Additional Data ROI (Frame {frame_idx + 1})")
            plt.imshow((cropped_additional.real * cropped_additional.imag), cmap="viridis")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

    cropped_binarized_frames = np.array([r[:min_height, :min_width] for r in cropped_binarized_frames])
    cropped_additional_frames = np.array([r[:min_height, :min_width] for r in cropped_additional_frames])

    return cropped_binarized_frames, cropped_additional_frames


def hash_frame(frame):
    return hashlib.md5(frame.tobytes()).hexdigest()


def count_unique_frames(array_3d):
    with ThreadPoolExecutor() as executor:
        hashes = list(executor.map(hash_frame, [array_3d[:, :, i] for i in range(array_3d.shape[2])]))
    frame_counts = Counter(hashes)
    return frame_counts


def show_unique_frames(array_3d):
    frame_counts = count_unique_frames(array_3d)
    df = pd.DataFrame.from_dict(frame_counts, orient='index', columns=['count'])
    df.index.name = 'frame_hash'
    df.reset_index(inplace=True)
    df['frame_number'] = range(len(df))
    df.set_index('frame_number', inplace=True)

    df['count'].plot(kind='bar')
    plt.xlabel('Frame Number')
    plt.ylabel('Number of Occurrences')
    plt.title('Histogram of Unique Frames')
    plt.tight_layout()
    plt.show()
    return df


def create_combined_frame(pinn_prod, original_prod, abs_diff_prod, mu_pred, mu_full, abs_diff_myu, mu_pred_original, mu_full_original, abs_diff_myu_original, enable_vmin_vmax=False):
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
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8').reshape(height, width, 4)
    plt.close(fig)
    return image


def create_video(output_path, video_file_name, pinn_prod_frames, original_prod_frames, abs_diff_prod_frames, mu_pred_frames, mu_full_frames, abs_diff_myu_frames, mu_pred_original_frames, mu_full_original_frames, abs_diff_myu_original_frames, fps=30, enable_vmin_vmax=False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    video_path = os.path.join(output_path, video_file_name if video_file_name.endswith(".mp4") else f"{video_file_name}.mp4")

    first_frame = create_combined_frame(
        pinn_prod_frames[0], original_prod_frames[0], abs_diff_prod_frames[0], mu_pred_frames[0], mu_full_frames[0], abs_diff_myu_frames[0], mu_pred_original_frames[0], mu_full_original_frames[0], abs_diff_myu_original_frames[0], enable_vmin_vmax
    )
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    try:
        for i in tqdm(range(len(pinn_prod_frames)), desc="Creating video"):
            combined_frame = create_combined_frame(
                pinn_prod_frames[i], original_prod_frames[i], abs_diff_prod_frames[i], mu_pred_frames[i], mu_full_frames[i], abs_diff_myu_frames[i], mu_pred_original_frames[i], mu_full_original_frames[i], abs_diff_myu_original_frames[i], enable_vmin_vmax
            )
            video_writer.write(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))
        video_writer.release()
    except Exception as e:
        video_writer.release()
        raise RuntimeError(f"Failed to create video: {e}")

    print(f"Video saved at: {video_path}")


def generate_video(
    state, 
    mu_full, 
    mu_full_original, 
    model, 
    x_vals, 
    y_vals, 
    t_vals, 
    device, 
    output_path, 
    video_file_name, 
    enable_vmin_vmax=False
):
    pinn_prod_frames, original_state_prod_frames, abs_diff_prod_frames, mu_pred_frames, mu_full_frames, abs_diff_myu_frames, mu_pred_original_frames, mu_full_original_frames, abs_diff_myu_original_frames = [], [], [], [], [], [], [], [], []

    mu_expanded_binarized = model.expand_myu_full(do_binarize=True, scale_255=False)
    mu_expanded_in_model = model.expand_myu_full(do_binarize=False, scale_255=False)
    
    for i, t_val in enumerate(tqdm(t_vals, desc="Generating frames")):
        X, Y = np.meshgrid(x_vals, y_vals)
        XX = X.ravel()
        YY = Y.ravel()
        TT = np.full_like(XX, t_val)

        x_test_t = torch.tensor(XX, dtype=torch.float32, device=device).view(-1, 1)
        y_test_t = torch.tensor(YY, dtype=torch.float32, device=device).view(-1, 1)
        t_test_t = torch.tensor(TT, dtype=torch.float32, device=device).view(-1, 1)

        A_r_pred, A_i_pred = model.predict(x_test_t, y_test_t, t_test_t)
        A_r_pred_2d = A_r_pred.reshape(X.shape)
        A_i_pred_2d = A_i_pred.reshape(X.shape)

        pinn_prod = A_r_pred_2d * A_i_pred_2d
        original_prod = state[i].real * state[i].imag
        
        if pinn_prod.shape != original_prod.shape:
            pinn_prod = pinn_prod.T
    
        abs_diff_prod = np.abs(original_prod - pinn_prod)
        abs_diff_myu = np.abs(mu_expanded_binarized[i] - mu_full[i])
        abs_diff_myu_original = np.abs(mu_full[i] - mu_expanded_in_model[i])

        pinn_prod_frames.append(pinn_prod)
        original_state_prod_frames.append(original_prod)
        abs_diff_prod_frames.append(abs_diff_prod)
        mu_pred_frames.append(mu_expanded_binarized[i])
        mu_full_frames.append(mu_full[i])
        abs_diff_myu_frames.append(abs_diff_myu)
        mu_pred_original_frames.append(mu_expanded_binarized[i])
        mu_full_original_frames.append(mu_full_original[i])
        abs_diff_myu_original_frames.append(abs_diff_myu_original)

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