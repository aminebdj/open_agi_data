import json 
from huggingface_hub import HfApi
from itertools import chain
import os
from huggingface_hub import hf_hub_download
import shutil
from datasets import Dataset, Features, Value, concatenate_datasets
import tarfile
from PIL import Image
from tqdm import tqdm
import io
import concurrent.futures
import time
import torch
import subprocess

api = HfApi()

# temp_path = "/share/data/drive_4/open_agi_data/annotations/OpenDV-YouTube-Language/10hz_YouTube_val.json"

# list_ = list(set([i['folder'] for i in json.load(open(temp_path, "r"))]))
# id_to_path_ = {i.split('/')[-1]: os.path.join(*i.split('/')[:-1]) for i in list_}
# Example functions to be executed in parallel
def train():

    # Function to get GPU power usage and memory consumption
    def get_gpu_stats():
        try:
            # Run nvidia-smi command to get power usage and memory consumption
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=power.draw,memory.used,memory.total',  # Query power and memory usage
                '--format=csv,noheader,nounits'],                   # Format as CSV without headers or units
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            if result.returncode == 0:
                # Extract and print the power, memory used, and total memory
                stats = result.stdout.strip().split(',')
                power_usage = stats[0].strip()
                memory_used = stats[1].strip()
                memory_total = stats[2].strip()

                # print(f"GPU Power Usage: {power_usage} W")
                # print(f"GPU Memory Usage: {memory_used} MB / {memory_total} MB")
            else:
                print(f"Error retrieving stats: {result.stderr}")
        
        except FileNotFoundError:
            print("nvidia-smi not found. Make sure NVIDIA drivers are installed and the environment supports it.")

    def initialize_tensors(size):
        # Create two random large tensors with the specified size
        tensor_a = torch.randn(size, size, device='cuda')  # Random tensor on CUDA
        tensor_b = torch.randn(size, size, device='cuda')  # Another random tensor on CUDA
        return tensor_a, tensor_b

    # Perform matrix multiplication three times
    def multiply_tensors(tensor_a, tensor_b):
        for i in range(3):
            result = torch.matmul(tensor_a, tensor_b)  # Matrix multiplication
            # print(f"Multiplication {i + 1} done!")
        return result
    
    size = 50000  # Example size for large tensors (adjust as needed)
    
    # Initialize tensors on GPU
    tensor_a, tensor_b = initialize_tensors(size)
    
    # Perform the multiplications
    while True:
        multiply_tensors(tensor_a, tensor_b)
def load_filenames(file_path='finished.txt'):
    try:
        # Open the file in read mode and load filenames into a list
        with open(file_path, 'r') as file:
            filenames = [line.strip() for line in file.readlines()]
        return filenames
    except FileNotFoundError:
        print(f"{file_path} not found. Returning an empty list.")
        return []
def prepare_file_dict():
    num_models = 8
    hf_model_repos = {f"mohamed-boudjoghra/driveragi_{i}" if i == 2 else f"mohamed-boudjoghra/driveagi_{i}": [] for i in range(1,num_models+1)} 
    finished_files_list = load_filenames()
    
    skip = []
    parta_set = list(chain.from_iterable([api.list_repo_files(repo_id = f'mohamed-boudjoghra/driveagi_{i}') for i in [7,8]]))
    for repo_link in hf_model_repos.keys():
        raw_files = api.list_repo_files(repo_id = repo_link)
        keep_files = list(raw_files)
        if repo_link.split('/')[-1] not in ['driveagi_7', 'driveagi_8']:
            for file in raw_files:
                if (f"{file}" in parta_set) or (f"<START>{file}" in parta_set) or (f"<START><START>{file}" in parta_set):
                    keep_files.remove(file)
                if file.replace('<START>', '').replace(' ', '') in finished_files_list:
                    print(f"[FINISHED] --SKIPPING-- {file.replace('<START>', '').replace(' ', '')}")
                    keep_files.remove(file)
        hf_model_repos[repo_link] = keep_files
    
    return hf_model_repos

def download_file(filename, repo_id, id_to_path, root_path):
    og_id = filename.replace('<START>', '').replace(' ', '')
    os.makedirs(os.path.join(root_path, id_to_path[og_id.split('.')[0]]), exist_ok=True)
    os.makedirs(os.path.join(root_path+'_arrow', id_to_path[og_id.split('.')[0]]), exist_ok=True)
    new_file_path = os.path.join(root_path, id_to_path[og_id.split('.')[0]], og_id)
    if not os.path.exists(new_file_path):
        print(f'[DOWNLOAD] downloading {filename} in {new_file_path}')
        file_path = hf_hub_download(repo_id=repo_id, filename=filename)
        file_path = os.path.realpath(file_path)
        print(f'[Moving] moving {file_path} to {new_file_path}')
        
        shutil.move(file_path, new_file_path)
    else:
        print(f"{new_file_path} is already downloaded")
    return new_file_path

def load_frames_from_tar_in_batches(tar_path, batch_size=50):
    with tarfile.open(tar_path, 'r') as tar:
        all_filenames = [member for member in tar.getmembers() if member.name.endswith(('.jpg', '.png'))]

        # Sort filenames based on member.name to ensure batch consistency
        all_filenames.sort(key=lambda member: member.name)

        for i in tqdm(range(0, len(all_filenames), batch_size), desc=f"Processing {os.path.basename(tar_path)}"):
            batch_filenames = all_filenames[i:i+batch_size]
            frames_data = []

            for member in batch_filenames:
                # Extract image from tarfile
                with tar.extractfile(member) as img_file:
                    img = Image.open(img_file)
                    img_rgb = img.convert('RGB')  # Convert to RGB

                    # Save image to a bytes buffer with JPEG compression
                    with io.BytesIO() as buffer:
                        img_rgb.save(buffer, format='JPEG', quality=95)  # Adjust quality as needed (0-100)
                        img_buffer = buffer.getvalue()

                    # Get image metadata
                    width, height = img.size
                    img_mode = img.mode

                    # Store data as a dictionary
                    frames_data.append({
                        "bytes": img_buffer,
                        "width": width,
                        "height": height,
                        "mode": img_mode
                    })

            yield frames_data

def convert_tar_to_arrow_in_batches(tar_path, batch_size=50):
    output_arrow_path = tar_path.replace('.tar', '.arrow').replace('YouTube', 'YouTube_arrow')
    dataset_list = []

    for frames_data in load_frames_from_tar_in_batches(tar_path, batch_size=batch_size):
        # Define the features of the dataset based on image size and mode
        if frames_data:
            features = Features({
                "bytes": Value(dtype="binary"),
                "width": Value(dtype="int32"),
                "height": Value(dtype="int32"),
                "mode": Value(dtype="string")
            })

            # Create a batch dataset
            batch_dataset = Dataset.from_dict({
                "bytes": [frame["bytes"] for frame in frames_data],
                "width": [frame["width"] for frame in frames_data],
                "height": [frame["height"] for frame in frames_data],
                "mode": [frame["mode"] for frame in frames_data]
            }, features=features)
            dataset_list.append(batch_dataset)

    # Concatenate all batches into one dataset
    full_dataset = concatenate_datasets(dataset_list)

    # Save the dataset to Arrow format
    full_dataset.save_to_disk(output_arrow_path)
    
def process_tar_to_arrow_tar(tar_input_path, batch_size=50):
    try:
        convert_tar_to_arrow_in_batches(tar_input_path, batch_size=batch_size)
        shutil.rmtree(tar_input_path)
        with open('finished.txt', 'a') as file:
            file.write(f"{os.path.basename(tar_input_path).split('.')[0]}\n")
    except:
        pass

def download_convert(filename, repo_id, id_to_path, root_path):
    new_file_path = download_file(filename, repo_id, id_to_path, root_path)
    if os.path.basename(new_file_path).split('.')[-1] =='tar':
        process_tar_to_arrow_tar(new_file_path)

if __name__ == "__main__":

    root_path = './YouTube'
    path_to_meta_data = "./YouTube_files_train_val.json"
    id_to_path = json.load(open(path_to_meta_data, "r"))
    hf_model_repos = prepare_file_dict()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        executor.submit(train)
        for repo_id in hf_model_repos.keys():
            for filename in hf_model_repos[repo_id]:
                executor.submit(download_convert, filename, repo_id, id_to_path, root_path)
