import os
import cv2
import shutil
import zipfile
import psutil
import GPUtil
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_system_info():
    """Retrieves and displays GPU, RAM, and disk information for the system."""

    def check_gpu():
        """Retrieves and displays the GPU name, total and available memory, driver version, and load percentage."""
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return "No GPU found"
            for gpu in gpus:
                total_memory = gpu.memoryTotal
                available_memory = gpu.memoryFree
                available_percentage = (available_memory / total_memory) * 100
                return f"GPU: {gpu.name}, Total Memory: {total_memory}MB, Available Memory: {available_memory}MB ({available_percentage:.2f}%), Driver Version: {gpu.driver}, GPU Load: {gpu.load * 100:.2f}%"
        except Exception as e:
            return f"Error retrieving GPU information: {str(e)}"

    def check_ram():
        """Returns the total and available RAM in the system in gigabytes."""
        mem = psutil.virtual_memory()
        total_ram = mem.total / (1024 ** 3)
        available_ram = mem.available / (1024 ** 3)
        available_percentage = (mem.available / mem.total) * 100
        return f"RAM: Total: {total_ram:.2f} GB, Available: {available_ram:.2f} GB ({available_percentage:.2f}%)"

    def check_disk():
        """Returns the total and available disk space on the primary partition in gigabytes."""
        disk = psutil.disk_usage('/')
        total_disk = disk.total / (1024 ** 3)
        available_disk = disk.free / (1024 ** 3)
        available_percentage = (disk.free / disk.total) * 100
        return f"Disk: Total: {total_disk:.2f} GB, Available: {available_disk:.2f} GB ({available_percentage:.2f}%)"

    gpu_info = check_gpu()
    ram_info = check_ram()
    disk_info = check_disk()

    return f"{gpu_info}\n{ram_info}\n{disk_info}"


def load_polarization_data(
        pol_0_data_path, 
        pol_90_data_path, 
        pol_45_data_path, 
        pol_135_data_path, 
        normalization_coefficient=0.867 # Change based on your data
    ):
    """Load and normalize polarization data from four different angles using a specified coefficient."""
    pol_0   = np.load(pol_0_data_path)
    pol_45  = np.load(pol_45_data_path)
    pol_90  = np.load(pol_90_data_path)
    pol_135 = np.load(pol_135_data_path)
    
    pol_0_normalized   = normalization_coefficient * pol_0   / (pol_0 + pol_90)
    pol_45_normalized  = normalization_coefficient * pol_45  / (pol_135 + pol_45)
    pol_90_normalized  = normalization_coefficient * pol_90  / (pol_0 + pol_90)
    pol_135_normalized = normalization_coefficient * pol_135 / (pol_135 + pol_45)
    
    return pol_0_normalized, pol_45_normalized, pol_90_normalized, pol_135_normalized


def visualize_and_return_components_of_state(states, myu, index):
    """Visualize various components of a complex state slice and a myu slice for a given index."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    state_slice = states[index]
    myu_slice = myu[index]
    
    modulus_square = np.abs(state_slice)**2
    multiplication_real_imag = state_slice.real * state_slice.imag
    imaginary_part = state_slice.imag
    real_part = state_slice.real
    phase = np.angle(state_slice)
    
    components = [
        (modulus_square,            r"$|A|^2$"),
        (multiplication_real_imag,  r"$\Re\{A\} \cdot \Im\{A\}$"),
        (imaginary_part,            r"$\Im\{A\}$"),
        (real_part,                 r"$\Re\{A\}$"),
        (phase,                     r"$\Psi$"),
        (myu_slice,                 r"$\mu$")
    ]
    
    for ax, (component, title) in zip(axes.flat, components):
        im = ax.imshow(component, cmap='viridis')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = fig.colorbar(im, ax=ax, cax=cax)
        cbar.ax.tick_params(labelsize=10)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r'$N_x$', fontsize=10)
        ax.set_ylabel(r'$N_y$', fontsize=10)
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.set_aspect(10, adjustable='box')
    
    plt.tight_layout()
    plt.show()
    
    return {
            "modulus_square": np.abs(states)**2,
            "multiplication_real_imag": states.real * states.imag,
            "imaginary_part": states.imag,
            "real_part": states.real,
            "phase": np.angle(states),
            "myu_slice": myu
    }


def display_polarization(pol_0_data, pol_90_data, pol_45_data, pol_135_data, index=0):
    """Visualize polarization data at the given index in a 2x2 plot layout."""
    _, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    data = [pol_0_data, pol_90_data, pol_45_data, pol_135_data]
    titles = ['0 Degrees Polarization', '90 Degrees Polarization', '45 Degrees Polarization', '135 Degrees Polarization']
    
    axes_flat = axes.flatten()
    
    for ax, dat, title in zip(axes_flat, data, titles):
        im = ax.imshow(dat[index], cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel(r'$N_x$')
        ax.set_ylabel(r'$N_y$')
        ax.invert_yaxis()
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=10)
        cax.set_aspect(20, adjustable='box')
    
    plt.tight_layout()
    plt.show()


def separate_polarization_data(
        data, 
        pol_0_file_name   = '0_degrees_polarization.npy', 
        pol_45_file_name  = '45_degrees_polarization.npy', 
        pol_90_file_name  = '90_degrees_polarization.npy',
        pol_135_file_name = '135_degrees_polarization.npy',
        output_dir_name   = 'polarizations_data_directory'
    ):
    """Process and separate polarization data into four angles, saving each as an individual .npy file."""
    main_dir = output_dir_name
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)

    data = data.reshape(data.shape[0], data.shape[2], data.shape[1])
    num_frames, rows, cols = data.shape

    pol_0   = np.zeros((num_frames, rows // 2, cols // 2))
    pol_45  = np.zeros((num_frames, rows // 2, cols // 2))
    pol_90  = np.zeros((num_frames, rows // 2, cols // 2))
    pol_135 = np.zeros((num_frames, rows // 2, cols // 2))
    
    for i in range(num_frames):
        # Separate the polarizations
        pol_90[i]  = data[i, 0::2, 0::2]   # 90 degrees
        pol_45[i]  = data[i, 0::2, 1::2]   # 45 degrees
        pol_135[i] = data[i, 1::2, 0::2]   # 135 degrees
        pol_0[i]   = data[i, 1::2, 1::2]   # 0 degrees

    np.save(os.path.join(main_dir, pol_0_file_name), pol_0)
    np.save(os.path.join(main_dir, pol_45_file_name), pol_45)
    np.save(os.path.join(main_dir, pol_90_file_name), pol_90)
    np.save(os.path.join(main_dir, pol_135_file_name), pol_135)
        
    print(f"0 polarization in {os.path.join(main_dir, pol_0_file_name)}")
    print(f"45 polarization in {os.path.join(main_dir, pol_45_file_name)}")
    print(f"90 polarization in {os.path.join(main_dir, pol_90_file_name)}")
    print(f"135 polarization in {os.path.join(main_dir, pol_135_file_name)}")   
    print(f"The polarization data collection process has been done successfully. Main dir is {main_dir}")


def rename_images(images_dir: str):
    """Rename image files in a directory sequentially with a specified format."""
    image_files = sorted([os.path.join(images_dir, img) 
        for img in os.listdir(images_dir) if img.endswith(('png', 'jpg', 'jpeg', 'bmp'))])
    for i, image_file in enumerate(image_files):
        file_extension = os.path.splitext(image_file)[1]
        new_filename = os.path.join(images_dir, f'ID{i+1:05d}{file_extension}')
        os.rename(image_file, new_filename)
        
        
def dump_frames_to_folder(frames: np.ndarray, output_folder: str):
    """Save each frame from a 3D numpy array to a specified folder as an image file."""
    os.makedirs(output_folder, exist_ok=True)
    num_frames = frames.shape[0]
    for i in tqdm(range(num_frames)):
        frame = frames[i]
        filename = os.path.join(output_folder, f'ID{i+1:05d}.png')
        plt.imsave(filename, frame, cmap='viridis')


def load_images_to_ndarray(folder_path):
    """Load PNG images from a folder into a 3D numpy array."""
    image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')])
    
    if not image_files:
        raise ValueError("No PNG images found in the folder.")
    
    first_image = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
    height, width = first_image.shape
    
    num_images = len(image_files)
    data = np.zeros((num_images, height, width), dtype=first_image.dtype)
    
    for i, file_path in enumerate(image_files):
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image.shape != (height, width):
            raise ValueError(f"Image {file_path} has a different size than the first image.")
        data[i] = image
    
    return data


def images_to_video(images_dir: str, output_video_path: str, fps: int = 30):
    """Convert a sequence of images in a directory into a video file with a specified frame rate."""
    image_files = [os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.endswith(('png', 'jpg', 'jpeg'))]
    
    if not image_files:
        print(f"No images found in directory {images_dir}.")
        return

    first_frame = cv2.imread(image_files[0])
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print(f"Error: The video file {output_video_path} could not be opened for writing.")
        return

    for image_file in tqdm(image_files, desc="Writing images to video"):
        frame = cv2.imread(image_file)
        if frame is None:
            print(f"Error reading image {image_file}")
            continue
        video_writer.write(frame)

    video_writer.release()

    cap = cv2.VideoCapture(output_video_path)
    if not cap.isOpened():
        print(f"Error: The video file {output_video_path} could not be created.")
    else:
        print(f"Video file {output_video_path} has been successfully created.")
        cap.release()


def prepare_ndarray_frame(data, vmin, vmax):
    """Prepares a frame from a numpy array for video by plotting it and returning the image as an ndarray."""
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.invert_yaxis()
    ax.axis('off')
    frame = data
    im = ax.imshow(frame, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    plt.close(fig)
    return image


def process_ndarray_frame(args):
    """Processes a frame by preparing it with given vmin and vmax values."""
    frame, vmin, vmax = args
    return prepare_ndarray_frame(frame, vmin, vmax)


def create_video_from_ndarray(input_data, output_path, fps=30):
    """Creates a video from a 3D numpy array of frames, saving it to the specified output path."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Directory {output_path} created.")
    
    if input_data.ndim != 3:
        raise ValueError("Input array must have 3 dimensions (number of frames, width, height).")

    video_path = os.path.join(output_path, f"output_video_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:-8]}.mp4")
    
    vmin, vmax = input_data.min(), input_data.max()
    
    height, width = prepare_ndarray_frame(input_data[0], vmin, vmax).shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    args = [(frame, vmin, vmax) for frame in input_data]

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            frames = list(tqdm(executor.map(process_ndarray_frame, args), total=len(input_data), desc="Creating video"))
            for video_frame in frames:
                video_writer.write(cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR))
        video_writer.release()
        print(f"Video saved at {video_path}")
    except Exception as e:
        video_writer.release()
        raise RuntimeError(f"Failed to create video: {e}")


def save_array_images(array, dir_path, name):
    """Saves all images in a given array to a directory, displaying a progress bar."""
    def save_image(img, path):
        """Saves a single image to the specified path."""
        plt.imsave(path, img)
    tasks = [(img, os.path.join(dir_path, f"image_{i:05d}.png")) for i, img in enumerate(array)]
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda p: save_image(*p), tasks), total=len(tasks), desc=f"Saving {name} images"))


def save_arrays_as_images(output_path, **arrays):
    """Creates directories and saves multiple arrays of images to separate directories."""
    os.makedirs(output_path, exist_ok=True)
    for name, array in arrays.items():
        dir_path = os.path.join(output_path, f"{name}_directory")
        os.makedirs(dir_path, exist_ok=True)
        save_array_images(array, dir_path, name)
        

def merge_image_folders(output_path, *image_folders):
    """Merges images from multiple directories into a single directory, preserving order and renaming images based on their source directory."""
    os.makedirs(output_path, exist_ok=True)
    image_counter = 0

    for folder in image_folders:
        folder_name = os.path.basename(folder)
        images = sorted(os.listdir(folder))
        for i, img_name in enumerate(tqdm(images, desc=f"Merging images from {folder_name}")):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                src_path = os.path.join(folder, img_name)
                dst_name = f"{folder_name}_{i:06d}.png"
                dst_path = os.path.join(output_path, dst_name)
                shutil.copy(src_path, dst_path)
                image_counter += 1
                

def split_images_into_arrays(path_to_directory):
    """Splits images from a directory into four 3D arrays (grayscale) while preserving order and naming conventions."""
    items = sorted([os.path.join(path_to_directory, f) for f in os.listdir(path_to_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
    num_items = len(items)
    items_per_array = num_items // 4

    arrays = {
        'pol_0_normalized': [],
        'pol_45_normalized': [],
        'pol_90_normalized': [],
        'pol_135_normalized': []
    }

    array_names = list(arrays.keys())

    for idx, item in enumerate(tqdm(items, desc="Processing images")):
        img = cv2.imread(item) 
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        array_index = (idx // items_per_array) % 4
        arrays[array_names[array_index]].append(img_gray)
    
    for key in arrays:
        arrays[key] = np.array(arrays[key])
    
    return arrays['pol_0_normalized'],  arrays['pol_90_normalized'], arrays['pol_135_normalized'], arrays['pol_45_normalized']


def extract_zip_to_folder(zip_path):
    """Extracts a zip file into a folder named after the zip file, showing progress with tqdm."""
    zip_dir = os.path.dirname(zip_path)
    folder_name = os.path.splitext(os.path.basename(zip_path))[0]
    extract_dir = os.path.join(zip_dir, folder_name)
    
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in tqdm(zip_ref.infolist(), desc=f"Extracting {os.path.basename(zip_path)}"):
            zip_ref.extract(file, extract_dir)


def extract_multiple_zips(zip_file_paths):
    """Concurrently extracts multiple zip files into their respective folders, showing progress for each."""
    with ThreadPoolExecutor() as executor:
        future_to_zip = {executor.submit(extract_zip_to_folder, zip_path): zip_path for zip_path in zip_file_paths}
        for future in as_completed(future_to_zip):
            zip_path = future_to_zip[future]
            try:
                future.result()
            except Exception as exc:
                print(f'{zip_path} generated an exception: {exc}')


def dump_frames(path_to_video, path_to_output_folder):
    """Extracts and saves all frames from a video file to a specified output folder using concurrent processing."""
    if not os.path.exists(path_to_output_folder):
        os.makedirs(path_to_output_folder)

    cap = cv2.VideoCapture(path_to_video)
    if not cap.isOpened():
        raise Exception("Error opening video file")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def save_frame(frame_index, frame):
        cv2.imwrite(os.path.join(path_to_output_folder, f"frame_{frame_index:06}.jpg"), frame)

    with concurrent.futures.ThreadPoolExecutor() as executor, tqdm(total=total_frames, desc="Dumping frames") as pbar:
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            executor.submit(save_frame, frame_index, frame)
            frame_index += 1
            pbar.update(1)
    
    cap.release()
    print("Done dumping frames.")


def calculate_correlations(
    path_to_images_folder, 
    save_csvs=False, 
    path_to_csvs=None, 
    show_plot=False, 
    extension='.jpg', 
    fig_size=(20, 10), 
    save_fig=False
)-> None:
    """Calculates and optionally saves the correlation of grayscale image frames within a folder to the first frame, with optional plotting and CSV saving."""
    if save_csvs and path_to_csvs and not os.path.exists(path_to_csvs):
        os.makedirs(path_to_csvs)

    folder_name = os.path.basename(os.path.normpath(path_to_images_folder))
    image_files = [os.path.join(path_to_images_folder, f) for f in os.listdir(path_to_images_folder) if f.endswith(extension)]
    image_files.sort()
    
    base_image = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
    base_image = base_image.flatten()

    def calc_correlation(base_image, image_path, frame_number):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image.flatten()
        correlation = np.corrcoef(base_image, image)[0, 1]
        return os.path.basename(image_path), frame_number, correlation

    correlations = [('Frame 0', 0, 1)]

    with concurrent.futures.ThreadPoolExecutor() as executor, tqdm(total=len(image_files)-1, desc="Calculating Correlations") as pbar:
        futures = [executor.submit(calc_correlation, base_image, img_path, idx) for idx, img_path in enumerate(image_files[1:], start=1)]
        for future in concurrent.futures.as_completed(futures):
            correlations.append(future.result())
            pbar.update(1)

    correlations.sort(key=lambda x: x[0])

    if save_csvs and path_to_csvs:
        df = pd.DataFrame(correlations, columns=['Frame_Name', 'Frame_Number', 'Correlation'])
        csv_filename = f'correlations_{folder_name}.csv'
        csv_path = os.path.join(path_to_csvs, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"Correlations saved to {csv_path}")

    if show_plot:
        plt.figure(figsize=fig_size)
        frame_nums = [cor[1] for cor in correlations]
        cor_vals = [cor[2] for cor in correlations]
        plt.plot(frame_nums, cor_vals, 'b-', marker='o', markerfacecolor='red', markersize=5)
        plt.xlabel('Frame Number')
        plt.ylabel('Correlation Value')
        plt.title('Frame Correlations')
        plt.grid(True)
        if save_fig:
            fig_filename = f'correlations_plot_{folder_name}.png'
            plt.savefig(os.path.join(path_to_csvs if path_to_csvs else '', fig_filename))
            print(f"Figure saved to {fig_filename}")
        plt.show()


def calculate_frame_to_next_correlations(
    path_to_images_folder, 
    save_csvs=False, 
    path_to_csvs=None, 
    show_plot=False, 
    extension='.jpg', 
    fig_size=(20, 10), 
    save_fig=False
)-> None:
    """Calculates and optionally saves the correlation between consecutive grayscale image frames in a folder, with optional plotting and CSV saving."""
    if save_csvs and path_to_csvs and not os.path.exists(path_to_csvs):
        os.makedirs(path_to_csvs)

    folder_name = os.path.basename(os.path.normpath(path_to_images_folder))
    image_files = [os.path.join(path_to_images_folder, f) for f in os.listdir(path_to_images_folder) if f.endswith(extension)]
    image_files.sort()

    def calc_correlation(image_path_1, image_path_2, frame_number_1, frame_number_2):
        image_1 = cv2.imread(image_path_1, cv2.IMREAD_GRAYSCALE)
        image_1 = image_1.flatten()
        image_2 = cv2.imread(image_path_2, cv2.IMREAD_GRAYSCALE)
        image_2 = image_2.flatten()
        correlation = np.corrcoef(image_1, image_2)[0, 1]
        return os.path.basename(image_path_1), os.path.basename(image_path_2), frame_number_1, frame_number_2, correlation

    correlations = []

    with concurrent.futures.ThreadPoolExecutor() as executor, tqdm(total=len(image_files)-1, desc="Calculating Correlations") as pbar:
        futures = [executor.submit(calc_correlation, image_files[i], image_files[i+1], i, i+1) for i in range(len(image_files)-1)]
        for future in concurrent.futures.as_completed(futures):
            correlations.append(future.result())
            pbar.update(1)

    correlations.sort(key=lambda x: x[2])

    if save_csvs and path_to_csvs:
        df = pd.DataFrame(correlations, columns=['Frame_Name_1', 'Frame_Name_2', 'Frame_Number_1', 'Frame_Number_2', 'Correlation'])
        csv_filename = f'frame_to_next_correlations_{folder_name}.csv'
        csv_path = os.path.join(path_to_csvs, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"Correlations saved to {csv_path}")

    if show_plot:
        plt.figure(figsize=fig_size)
        frame_nums = [cor[2] for cor in correlations]
        cor_vals = [cor[4] for cor in correlations]
        plt.plot(frame_nums, cor_vals, 'b-', marker='o', markerfacecolor='red', markersize=5)
        plt.xlabel('Frame Number')
        plt.ylabel('Correlation Value')
        plt.title('Frame to Next Frame Correlations')
        plt.grid(True)
        if save_fig:
            fig_filename = f'frame_to_next_correlations_plot_{folder_name}.png'
            plt.savefig(os.path.join(path_to_csvs if path_to_csvs else '', fig_filename))
            print(f"Figure saved to {fig_filename}")
        plt.show()


def process_videos(
    path_to_videos_folder, 
    save_csvs=False, 
    path_to_csvs=None, 
    show_plot=False, 
    extension='.jpg', 
    fig_size=(20, 10), 
    save_fig=False,
    path_to_output_folder_dir=None,
    use_frame_to_next_correlations=False
):
    """Processes video files by extracting frames, calculating correlations, optionally saving results, and cleaning up temporary files."""
    video_files = [os.path.join(path_to_videos_folder, f) for f in os.listdir(path_to_videos_folder) if f.endswith('.avi') or f.endswith('.mp4')]
    
    for video_path in video_files:
        video_file_name = os.path.splitext(os.path.basename(video_path))[0]
        path_to_output_folder = os.path.join(path_to_output_folder_dir, video_file_name)
        
        dump_frames(video_path, path_to_output_folder)

        if use_frame_to_next_correlations:
            print("Switched to calculate_frame_to_next_correlations method.")
            calculate_frame_to_next_correlations(
                path_to_output_folder, 
                save_csvs=save_csvs, 
                path_to_csvs=path_to_csvs, 
                show_plot=show_plot, 
                extension=extension, 
                fig_size=fig_size, 
                save_fig=save_fig
            )
        else:
            calculate_correlations(
                path_to_output_folder, 
                save_csvs=save_csvs, 
                path_to_csvs=path_to_csvs, 
                show_plot=show_plot, 
                extension=extension, 
                fig_size=fig_size, 
                save_fig=save_fig
            )
        
        shutil.rmtree(path_to_output_folder)
        
    print("All videos have been processed successfully.")


def prepare_frame_all(data, myu_frame, vmin_module_square, vmax_module_square, vmin_real_imag, vmax_real_imag, vmin_phase, vmax_phase, vmin_myu, vmax_myu):
    """Prepares a composite frame from multiple visual representations of complex data and Myu for video creation."""
    fig, axes = plt.subplots(1, 4, figsize=(30, 5))

    titles = ['Module Square', 'Real x Imaginary', 'Phase', 'Myu']
    frames = [np.abs(data)**2, data.real * data.imag, np.angle(data), myu_frame]
    vmins = [vmin_module_square, vmin_real_imag, vmin_phase, vmin_myu]
    vmaxs = [vmax_module_square, vmax_real_imag, vmax_phase, vmax_myu]

    for ax, frame, title, vmin, vmax in zip(axes, frames, titles, vmins, vmaxs):
        im = ax.imshow(frame, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.invert_yaxis()
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Colorbar', rotation=270)

    plt.tight_layout()
    fig.canvas.draw()

    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    plt.close(fig)

    return image


def process_frame_all(args):
    """Processes a frame by preparing multiple visual representations of complex data and Myu with given min and max values."""
    frame, myu_frame, vmin_module_square, vmax_module_square, vmin_real_imag, vmax_real_imag, vmin_phase, vmax_phase, vmin_myu, vmax_myu = args
    return prepare_frame_all(frame, myu_frame, vmin_module_square, vmax_module_square, vmin_real_imag, vmax_real_imag, vmin_phase, vmax_phase, vmin_myu, vmax_myu)


def create_video_from_ndarray_for_states_and_myu(input_data, myu, output_path, fps=30):
    """Creates a video from a 3D numpy array of complex data frames and corresponding Myu frames, saving it to the specified output path."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Directory {output_path} created.")
    
    if input_data.ndim != 3:
        raise ValueError("Input array must have 3 dimensions (number of frames, width, height).")
    
    if not np.iscomplexobj(input_data):
        raise ValueError("Input array must be of complex type.")

    video_path = os.path.join(output_path, f"output_video_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:-8]}.mp4")
    
    module_square = np.abs(input_data)**2
    real_imag = input_data.real * input_data.imag
    phase = np.angle(input_data)
    
    vmin_module_square, vmax_module_square = module_square.min(), module_square.max()
    vmin_real_imag, vmax_real_imag = real_imag.min(), real_imag.max()
    vmin_phase, vmax_phase = phase.min(), phase.max()
    vmin_myu, vmax_myu = myu.min(), myu.max()
    
    height, width = prepare_frame_all(input_data[0], myu[0], vmin_module_square, vmax_module_square, vmin_real_imag, vmax_real_imag, vmin_phase, vmax_phase, vmin_myu, vmax_myu).shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    args = [(frame, myu_frame, vmin_module_square, vmax_module_square, vmin_real_imag, vmax_real_imag, vmin_phase, vmax_phase, vmin_myu, vmax_myu) 
            for frame, myu_frame in zip(input_data, myu)]

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            frames = list(tqdm(executor.map(process_frame_all, args), total=len(input_data), desc="Creating video"))
            for video_frame in frames:
                video_writer.write(cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR))
        video_writer.release()
        print(f"Video saved at {video_path}")
    except Exception as e:
        video_writer.release()
        raise RuntimeError(f"Failed to create video: {e}")
