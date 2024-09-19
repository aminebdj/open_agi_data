import json 
from huggingface_hub import HfApi
from itertools import chain
import os
from huggingface_hub import hf_hub_download
import shutil
api = HfApi()

# temp_path = "/share/data/drive_4/open_agi_data/annotations/OpenDV-YouTube-Language/10hz_YouTube_val.json"

# list_ = list(set([i['folder'] for i in json.load(open(temp_path, "r"))]))
# id_to_path_ = {i.split('/')[-1]: os.path.join(*i.split('/')[:-1]) for i in list_}

path_to_meta_data = "./YouTube_files_train_val.json"
num_models = 8

id_to_path = json.load(open(path_to_meta_data, "r"))

hf_model_repos = {f"mohamed-boudjoghra/driveragi_{i}" if i == 2 else f"mohamed-boudjoghra/driveagi_{i}": [] for i in range(1,num_models+1)} 

skip = []
parta_set = list(chain.from_iterable([api.list_repo_files(repo_id = f'mohamed-boudjoghra/driveagi_{i}') for i in [7,8]]))
for repo_link in hf_model_repos.keys():
    raw_files = api.list_repo_files(repo_id = repo_link)
    keep_files = list(raw_files)
    if repo_link.split('/')[-1] not in ['driveagi_7', 'driveagi_8']:
        for file in raw_files:
            if (f"{file}" in parta_set) or (f"<START>{file}" in parta_set) or (f"<START><START>{file}" in parta_set):
                keep_files.remove(file)
    hf_model_repos[repo_link] = keep_files
all_files = set(list(chain.from_iterable(list(hf_model_repos.values()))))

root_path = '/scratch/project_465000695/llm/amine/YouTube_raw'
for repo_id in hf_model_repos.keys():
    for filename in hf_model_repos[repo_id]:
        og_id = filename.replace('<START>', '').replace(' ', '')
        os.makedirs(os.path.join(root_path, id_to_path[og_id.split('.')[0]]), exist_ok=True)
        new_file_path = os.path.join(root_path, id_to_path[og_id.split('.')[0]], og_id)
        print(f'[DOWNLOAD] downloading {filename} in {new_file_path}')
        file_path = hf_hub_download(repo_id=repo_id, filename=filename)
        file_path = os.path.realpath(file_path)
        print(f'[Moving] moving {file_path} to {new_file_path}')
        
        shutil.move(file_path, new_file_path)
i = 0