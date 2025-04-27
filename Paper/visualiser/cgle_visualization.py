import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.gridspec as gridspec
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import HTML, display
import matplotlib.colors as colors


class CGLEVisualizer:
    """
    A class for visualizing Complex Ginzburg-Landau Equation (CGLE) data,
    including state fields and mu fields (binary and raw).

    This version is designed to handle complex-valued state data.
    """

    def __init__(self, output_dir="./visualizations"):
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_data(self, state_path, mu_binary_path=None, mu_raw_path=None):
        """
        Load data from .npy files with complex state data handling.

        Args:
            state_path: Path to state data (complex-valued)
            mu_binary_path: Path to binarized mu field data
            mu_raw_path: Path to raw mu field data

        Returns:
            self for method chaining
        """
        try:
            self.state_data = np.load(state_path)
            print(f"State data shape: {self.state_data.shape}")
            print(f"State data type: {self.state_data.dtype}")

            # Handle complex data - extract real and imaginary parts
            if np.iscomplexobj(self.state_data):
                print("Detected complex data, extracting real and imaginary parts...")
                self.A_complex = self.state_data
                self.A_r = np.real(self.state_data)
                self.A_i = np.imag(self.state_data)
            else:
                # If not complex, use state data as real part and zeros for imaginary
                self.A_r = self.state_data
                self.A_i = np.zeros_like(self.state_data)
                self.A_complex = self.A_r + 1j * self.A_i

            # Get dimensions
            self.Nt, self.Nx, self.Ny = self.state_data.shape

            print(f"A_r shape: {self.A_r.shape}")
            print(f"A_i shape: {self.A_i.shape}")

            # Compute amplitude and phase
            self.A_amplitude = np.abs(self.A_complex)
            self.A_phase = np.angle(self.A_complex)

            print(f"A_amplitude shape: {self.A_amplitude.shape}")
            print(f"A_phase shape: {self.A_phase.shape}")

        except Exception as e:
            print(f"Error loading state data: {e}")
            raise

        # Load mu fields if provided
        if mu_binary_path:
            try:
                self.mu_binary = np.load(mu_binary_path)
                print(f"Mu binary data shape: {self.mu_binary.shape}")
            except Exception as e:
                print(f"Error loading mu binary data: {e}")
                self.mu_binary = None
        else:
            self.mu_binary = None

        if mu_raw_path:
            try:
                self.mu_raw = np.load(mu_raw_path)
                print(f"Mu raw data shape: {self.mu_raw.shape}")

                # Handle different dimensions between state and mu_raw
                if self.mu_raw.shape != self.state_data.shape:
                    print(f"Note: Mu raw data shape differs from state data shape")
                    # If the dimensions differ in the last dimension, we might need to resize
                    if (self.mu_raw.shape[0] == self.state_data.shape[0] and
                            self.mu_raw.shape[1] == self.state_data.shape[1]):
                        # Different only in the last dimension
                        # Let's resize to match state data for visualization
                        new_mu_raw = np.zeros((self.Nt, self.Nx, self.Ny))
                        # Take the appropriate slice or pad
                        min_Ny = min(self.Ny, self.mu_raw.shape[2])
                        print(f"Resizing mu_raw to match state data shape (taking first {min_Ny} columns)")
                        new_mu_raw[:, :, :min_Ny] = self.mu_raw[:, :, :min_Ny]
                        self.mu_raw = new_mu_raw
            except Exception as e:
                print(f"Error loading mu raw data: {e}")
                self.mu_raw = None
        else:
            self.mu_raw = None

        return self

    def visualize_frame(self, time_idx, fig_size=(16, 10), save_path=None):
        """
        Visualize a specific time frame with multiple subplots.

        Args:
            time_idx: Time index to visualize
            fig_size: Figure size (width, height)
            save_path: If provided, save the figure to this path

        Returns:
            matplotlib figure
        """
        if time_idx >= self.Nt:
            print(f"Error: time_idx {time_idx} is outside the range of available data (0-{self.Nt - 1})")
            return None

        fig = plt.figure(figsize=fig_size)

        # Create grid for subplots
        n_rows = 2
        n_cols = 3

        if self.mu_binary is not None or self.mu_raw is not None:
            n_rows += 1

        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)

        # Create subplots
        ax1 = fig.add_subplot(gs[0, 0])  # A_r
        ax2 = fig.add_subplot(gs[0, 1])  # A_i
        ax3 = fig.add_subplot(gs[0, 2])  # |A|
        ax4 = fig.add_subplot(gs[1, 0])  # phase
        ax5 = fig.add_subplot(gs[1, 1])  # A_r * A_i
        ax6 = fig.add_subplot(gs[1, 2])  # Combined visualization

        if self.mu_binary is not None or self.mu_raw is not None:
            ax7 = fig.add_subplot(gs[2, 0])  # mu_binary
            ax8 = fig.add_subplot(gs[2, 1])  # mu_raw
            if self.mu_raw is not None and self.mu_binary is not None:
                ax9 = fig.add_subplot(gs[2, 2])  # difference

        # Plot A_r (real part)
        vmax = np.max(np.abs(self.A_r[time_idx]))
        im1 = ax1.imshow(self.A_r[time_idx], cmap='RdBu', vmin=-vmax, vmax=vmax)
        ax1.set_title(f'Real part A_r (t={time_idx})')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax)

        # Plot A_i (imaginary part)
        vmax = np.max(np.abs(self.A_i[time_idx]))
        im2 = ax2.imshow(self.A_i[time_idx], cmap='RdBu', vmin=-vmax, vmax=vmax)
        ax2.set_title(f'Imaginary part A_i (t={time_idx})')
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax)

        # Plot amplitude |A|
        im3 = ax3.imshow(self.A_amplitude[time_idx], cmap='viridis')
        ax3.set_title(f'Amplitude |A| (t={time_idx})')
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im3, cax=cax)

        # Plot phase arg(A)
        im4 = ax4.imshow(self.A_phase[time_idx], cmap='hsv', vmin=-np.pi, vmax=np.pi)
        ax4.set_title(f'Phase arg(A) (t={time_idx})')
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im4, cax=cax)

        # Plot heatmap of Re(A) * Im(A)
        product = self.A_r[time_idx] * self.A_i[time_idx]
        vmax = np.max(np.abs(product))
        im5 = ax5.imshow(product, cmap='RdBu', vmin=-vmax, vmax=vmax)
        ax5.set_title(f'Re(A) * Im(A) (t={time_idx})')
        divider = make_axes_locatable(ax5)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im5, cax=cax)

        # Plot combined visualization: amplitude as brightness, phase as color
        # Create an HSV representation where:
        # - H (hue) is the phase
        # - S (saturation) is set to 1
        # - V (value/brightness) is the normalized amplitude
        hsv_data = np.zeros((self.Nx, self.Ny, 3))
        normalized_phase = (self.A_phase[time_idx] + np.pi) / (2 * np.pi)  # Map from [-π, π] to [0, 1]
        normalized_amplitude = self.A_amplitude[time_idx] / (np.max(self.A_amplitude[time_idx]) + 1e-10)

        hsv_data[:, :, 0] = normalized_phase
        hsv_data[:, :, 1] = 1.0  # Full saturation
        hsv_data[:, :, 2] = normalized_amplitude

        im6 = ax6.imshow(colors.hsv_to_rgb(hsv_data))
        ax6.set_title(f'Combined: Phase (color) & Amplitude (brightness) (t={time_idx})')

        # Plot mu fields if available
        if self.mu_binary is not None:
            if time_idx < len(self.mu_binary):
                im7 = ax7.imshow(self.mu_binary[time_idx], cmap='binary')
                ax7.set_title(f'Mu Binary (t={time_idx})')
                divider = make_axes_locatable(ax7)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im7, cax=cax)
            else:
                ax7.text(0.5, 0.5, 'Time index exceeds mu_binary length',
                         horizontalalignment='center', verticalalignment='center')
                ax7.set_title('Mu Binary - Not Available')

        if self.mu_raw is not None:
            if time_idx < len(self.mu_raw):
                im8 = ax8.imshow(self.mu_raw[time_idx], cmap='plasma')
                ax8.set_title(f'Mu Raw (t={time_idx})')
                divider = make_axes_locatable(ax8)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im8, cax=cax)
            else:
                ax8.text(0.5, 0.5, 'Time index exceeds mu_raw length',
                         horizontalalignment='center', verticalalignment='center')
                ax8.set_title('Mu Raw - Not Available')

        if self.mu_raw is not None and self.mu_binary is not None:
            if time_idx < len(self.mu_raw) and time_idx < len(self.mu_binary):
                # Calculate difference between raw and binary
                # Ensure they have the same shape for subtraction
                if self.mu_raw[time_idx].shape == self.mu_binary[time_idx].shape:
                    diff = self.mu_raw[time_idx] - self.mu_binary[time_idx]
                    im9 = ax9.imshow(diff, cmap='RdBu')
                    ax9.set_title(f'Mu Raw - Mu Binary (t={time_idx})')
                    divider = make_axes_locatable(ax9)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im9, cax=cax)
                else:
                    ax9.text(0.5, 0.5, 'Shape mismatch between mu_raw and mu_binary',
                             horizontalalignment='center', verticalalignment='center')
                    ax9.set_title('Mu Difference - Not Available')
            else:
                ax9.text(0.5, 0.5, 'Time index exceeds mu data length',
                         horizontalalignment='center', verticalalignment='center')
                ax9.set_title('Mu Difference - Not Available')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Frame saved to {save_path}")

        return fig

    def create_animation(self, output_filename, start_idx=0, end_idx=None,
                         interval=100, fps=10, dpi=100, mode='phase_amplitude'):
        """
        Create an animation of the fields.

        Args:
            output_filename: Name of the output file (without extension)
            start_idx: Starting time index
            end_idx: Ending time index (None means all frames)
            interval: Interval between frames in milliseconds
            fps: Frames per second for the saved animation
            dpi: Resolution
            mode: Visualization mode - options:
                  'phase_amplitude': Combined phase (color) and amplitude (brightness)
                  'state_only': Only state fields (no mu)
                  'mu_only': Only mu fields
        """
        if end_idx is None:
            end_idx = self.Nt

        frame_indices = range(start_idx, min(end_idx, self.Nt))

        if mode == 'phase_amplitude':
            fig, ax = plt.subplots(figsize=(8, 8))

            # Initial frame
            hsv_data = np.zeros((self.Nx, self.Ny, 3))
            normalized_phase = (self.A_phase[start_idx] + np.pi) / (2 * np.pi)
            normalized_amplitude = self.A_amplitude[start_idx] / (np.max(self.A_amplitude[start_idx]) + 1e-10)

            hsv_data[:, :, 0] = normalized_phase
            hsv_data[:, :, 1] = 1.0
            hsv_data[:, :, 2] = normalized_amplitude

            im = ax.imshow(colors.hsv_to_rgb(hsv_data))

            title = ax.set_title(f'Combined Phase & Amplitude, t={start_idx}')

            def update(frame_idx):
                hsv_data = np.zeros((self.Nx, self.Ny, 3))
                normalized_phase = (self.A_phase[frame_idx] + np.pi) / (2 * np.pi)
                normalized_amplitude = self.A_amplitude[frame_idx] / (np.max(self.A_amplitude[frame_idx]) + 1e-10)

                hsv_data[:, :, 0] = normalized_phase
                hsv_data[:, :, 1] = 1.0
                hsv_data[:, :, 2] = normalized_amplitude

                im.set_array(colors.hsv_to_rgb(hsv_data))
                title.set_text(f'Combined Phase & Amplitude, t={frame_idx}')
                return [im, title]

        elif mode == 'state_only':
            # Only state fields
            fig = plt.figure(figsize=(12, 8))
            gs = gridspec.GridSpec(2, 2, figure=fig)

            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])

            # Initial frame - A_r
            vmax = np.max(np.abs(self.A_r[start_idx]))
            im1 = ax1.imshow(self.A_r[start_idx], cmap='RdBu', vmin=-vmax, vmax=vmax)
            ax1.set_title('Real part (A_r)')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            # Initial frame - A_i
            vmax = np.max(np.abs(self.A_i[start_idx]))
            im2 = ax2.imshow(self.A_i[start_idx], cmap='RdBu', vmin=-vmax, vmax=vmax)
            ax2.set_title('Imaginary part (A_i)')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            # Initial frame - |A|
            im3 = ax3.imshow(self.A_amplitude[start_idx], cmap='viridis')
            ax3.set_title('Amplitude |A|')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

            # Initial frame - Phase
            im4 = ax4.imshow(self.A_phase[start_idx], cmap='hsv', vmin=-np.pi, vmax=np.pi)
            ax4.set_title('Phase arg(A)')
            plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

            plt.tight_layout()

            def update(frame_idx):
                # Update A_r
                vmax = np.max(np.abs(self.A_r[frame_idx]))
                im1.set_array(self.A_r[frame_idx])
                im1.set_clim(vmin=-vmax, vmax=vmax)

                # Update A_i
                vmax = np.max(np.abs(self.A_i[frame_idx]))
                im2.set_array(self.A_i[frame_idx])
                im2.set_clim(vmin=-vmax, vmax=vmax)

                # Update |A|
                im3.set_array(self.A_amplitude[frame_idx])
                im3.set_clim(vmin=0, vmax=np.max(self.A_amplitude[frame_idx]))

                # Update Phase
                im4.set_array(self.A_phase[frame_idx])

                return [im1, im2, im3, im4]

        elif mode == 'mu_only':
            # Only mu fields
            if self.mu_binary is None and self.mu_raw is None:
                print("No mu fields available for visualization.")
                return

            fig = plt.figure(figsize=(12, 6))

            if self.mu_binary is not None and self.mu_raw is not None:
                gs = gridspec.GridSpec(1, 2, figure=fig)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1])

                # Initial frame - Binary
                mu_b_idx = min(start_idx, len(self.mu_binary) - 1)
                im1 = ax1.imshow(self.mu_binary[mu_b_idx], cmap='binary')
                ax1.set_title('Mu Binary')
                plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

                # Initial frame - Raw
                mu_r_idx = min(start_idx, len(self.mu_raw) - 1)
                im2 = ax2.imshow(self.mu_raw[mu_r_idx], cmap='plasma')
                ax2.set_title('Mu Raw')
                plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

                def update(frame_idx):
                    mu_b_idx = min(frame_idx, len(self.mu_binary) - 1)
                    mu_r_idx = min(frame_idx, len(self.mu_raw) - 1)

                    im1.set_array(self.mu_binary[mu_b_idx])
                    im2.set_array(self.mu_raw[mu_r_idx])

                    return [im1, im2]

            elif self.mu_binary is not None:
                gs = gridspec.GridSpec(1, 1, figure=fig)
                ax1 = fig.add_subplot(gs[0, 0])

                # Initial frame - Binary
                mu_idx = min(start_idx, len(self.mu_binary) - 1)
                im1 = ax1.imshow(self.mu_binary[mu_idx], cmap='binary')
                ax1.set_title('Mu Binary')
                plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

                def update(frame_idx):
                    mu_idx = min(frame_idx, len(self.mu_binary) - 1)
                    im1.set_array(self.mu_binary[mu_idx])
                    return [im1]

            elif self.mu_raw is not None:
                gs = gridspec.GridSpec(1, 1, figure=fig)
                ax1 = fig.add_subplot(gs[0, 0])

                # Initial frame - Raw
                mu_idx = min(start_idx, len(self.mu_raw) - 1)
                im1 = ax1.imshow(self.mu_raw[mu_idx], cmap='plasma')
                ax1.set_title('Mu Raw')
                plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

                def update(frame_idx):
                    mu_idx = min(frame_idx, len(self.mu_raw) - 1)
                    im1.set_array(self.mu_raw[mu_idx])
                    return [im1]

        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from 'phase_amplitude', 'state_only', or 'mu_only'")

        plt.tight_layout()

        # Create animation
        anim = FuncAnimation(fig, update, frames=frame_indices,
                             interval=interval, blit=True)

        # Save animation as GIF
        gif_path = os.path.join(self.output_dir, f"{output_filename}.gif")

        try:
            # Try PillowWriter for GIF
            writer = PillowWriter(fps=fps)
            anim.save(gif_path, writer=writer, dpi=dpi)
            print(f"Animation saved to {gif_path}")
        except Exception as e:
            print(f"Failed to save as GIF: {e}")

            # Try to display inline for Jupyter notebooks
            try:
                print("Trying to display inline...")
                plt.close(fig)
                return HTML(anim.to_jshtml())
            except Exception as e2:
                print(f"Failed to display inline: {e2}")

        plt.close(fig)