import numpy as np
import matplotlib.pyplot as plt
import os
from cgle_visualization import CGLEVisualizer

# Create output directory for visualizations
output_dir = "./cgle_visualizations"
os.makedirs(output_dir, exist_ok=True)

# Path to your data files
# Replace these with your actual file paths
state_data_path = "data/state_data.npy"
mu_binary_path = "data/myus_binarized.npy"
mu_raw_path = "data/myus_raw.npy"

output_dir = "./cgle_visualizations"
os.makedirs(output_dir, exist_ok=True)

# Import the visualizer class
# Make sure the file is in the same directory as this script
from cgle_visualization import CGLEVisualizer


def main():
    """Main function to test the CGLE visualization with complex data."""
    print("Initializing CGLE Visualizer for complex data...")

    # Initialize the visualizer
    visualizer = CGLEVisualizer(output_dir=output_dir)

    # Load data (with proper handling of complex-valued state data)
    print("\nLoading data...")
    visualizer.load_data(
        state_path=state_data_path,
        mu_binary_path=mu_binary_path,
        mu_raw_path=mu_raw_path
    )

    # Example 1: Generate a few static frames
    print("\nGenerating static frames...")
    frame_indices = [0, 100, 500, 1000, 1499]  # Choose frames across your 1500 timesteps
    for frame_idx in frame_indices:
        try:
            fig = visualizer.visualize_frame(
                time_idx=frame_idx,
                fig_size=(16, 12),
                save_path=os.path.join(output_dir, f"frame_{frame_idx}.png")
            )
            plt.close(fig)  # Close to free memory
            print(f"Successfully generated frame {frame_idx}")
        except Exception as e:
            print(f"Error generating frame {frame_idx}: {e}")

    # Example 2: Create animation with phase and amplitude only (smaller and faster)
    print("\nCreating phase-amplitude animation...")
    try:
        visualizer.create_animation(
            output_filename="phase_amplitude_animation",
            start_idx=0,
            end_idx=50,  # Using a smaller subset for speed
            interval=200,  # milliseconds between frames
            fps=5,
            dpi=100,
            mode='phase_amplitude'
        )
        print("Successfully created phase-amplitude animation")
    except Exception as e:
        print(f"Error creating phase-amplitude animation: {e}")

    # Example 3: Create animation with just the state fields
    print("\nCreating state fields animation...")
    try:
        visualizer.create_animation(
            output_filename="state_animation",
            start_idx=0,
            end_idx=50,  # Using a smaller subset for speed
            interval=200,
            fps=5,
            dpi=100,
            mode='state_only'
        )
        print("Successfully created state fields animation")
    except Exception as e:
        print(f"Error creating state fields animation: {e}")

    # Example 4: Create animation of just the mu fields (if available)
    print("\nCreating mu fields animation...")
    try:
        visualizer.create_animation(
            output_filename="mu_animation",
            start_idx=0,
            end_idx=50,  # Using a smaller subset for speed
            interval=200,
            fps=5,
            dpi=100,
            mode='mu_only'
        )
        print("Successfully created mu fields animation")
    except Exception as e:
        print(f"Error creating mu fields animation: {e}")

    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()