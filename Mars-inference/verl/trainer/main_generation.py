#运行过程中保存
import re
import os
import torch
import torch.distributed as dist
from datetime import timedelta
import multiprocessing as mp
import numpy as np
import hydra
import pandas as pd
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl.utils.model import compute_position_id_with_mask
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl import DataProto
from tensordict import TensorDict
from verl.workers.fsdp_workers import ActorRolloutRefWorker

# 1. Set environment variables at the top level
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0' # Some libraries require this
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29507'  # <-- Use a non-conflicting, standalone port


@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    from pprint import pprint
    pprint(OmegaConf.to_container(config, resolve=True))

    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get('trust_remote_code', False)
    tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=trust_remote_code)

    n_trajectories = config.rollout.n_trajectories
    if config.rollout.temperature == 0. and n_trajectories > 1:
        print(f"WARNING: Temperature is 0, so the {n_trajectories} generated trajectories will be identical.")

    # --- MODIFICATION 1: Resumption Logic ---
    output_path = config.data.output_path
    output_dir = os.path.dirname(output_path)
    makedirs(output_dir, exist_ok=True)
    
    start_batch = 0
    if os.path.exists(output_path):
        print(f"Output file found at {output_path}. Attempting to resume.")
        dataset = pd.read_parquet(output_path)
        # Ensure result columns exist
        if 'result' not in dataset.columns:
            dataset['result'] = [None] * len(dataset)
        if 'full_responses' not in dataset.columns:
            dataset['full_responses'] = [None] * len(dataset)
            
        # Find the first unprocessed index
        if dataset['result'].isnull().any():
            first_unprocessed_index = dataset['result'].isnull().idxmax()
            start_batch = first_unprocessed_index // config.data.batch_size
            print(f"Resuming from batch {start_batch + 1}")
        else: # If everything is already processed
            print("Processing is already complete. Exiting.")
            return
    else:
        print("No output file found. Starting a new run.")
        dataset = pd.read_parquet(config.data.path)
        # Initialize columns for results
        dataset['full_responses'] = [None] * len(dataset)
        dataset['result'] = [None] * len(dataset)
    
    chat_lst = dataset[config.data.prompt_key].tolist()
    # Convert if data are numpy arrays
    if isinstance(chat_lst[0], np.ndarray):
        chat_lst = [chat.tolist() for chat in chat_lst]
    db_id_lst = dataset['db_id'].tolist()
    data_src_lst = dataset['data'].tolist()
    
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rollout_worker = ActorRolloutRefWorker(config=config, role='rollout')
    rollout_worker.init_model()

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    num_batch = -(-total_samples // config_batch_size)

    # --- MODIFICATION 2: Iterate from the starting batch ---
    for batch_idx in range(start_batch, num_batch):
        print(f'[{batch_idx+1}/{num_batch}] Processing... generating {n_trajectories} trajectories per prompt.')
        
        start_index = batch_idx * config_batch_size
        end_index = min((batch_idx + 1) * config_batch_size, total_samples) # Use min for the last batch
        
        batch_chat_lst = chat_lst[start_index:end_index]
        batch_db_id_lst = db_id_lst[start_index:end_index]
        batch_data_src_lst = data_src_lst[start_index:end_index]
        
        inputs = tokenizer.apply_chat_template(
            batch_chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=config.rollout.prompt_length,
            return_tensors='pt',
            return_dict=True,
            tokenize=True
        )

        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        position_ids = compute_position_id_with_mask(attention_mask)

        tensor_batch = TensorDict({'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}, batch_size=len(batch_chat_lst))
        non_tensor_batch = {'db_id': np.array(batch_db_id_lst, dtype=object), 'data': np.array(batch_data_src_lst, dtype=object), 'raw_prompt': np.array(batch_chat_lst, dtype=object)}
        data = DataProto(batch=tensor_batch, non_tensor_batch=non_tensor_batch)

        output = rollout_worker.generate_sequences(data)
        
        if 'responses' in output.batch.keys():
            response_tokens = output.batch['responses']
            
            raw_text = tokenizer.batch_decode(response_tokens, skip_special_tokens=False)
            
            pad_token = tokenizer.pad_token if tokenizer.pad_token is not None else tokenizer.eos_token
            clean_raw_text = [text.replace(pad_token, '').strip() for text in raw_text]

            parsed_responses = []
            for text in clean_raw_text:
                solutions = re.findall(r"<solution>(.*?)</solution>", text, re.DOTALL)
                if solutions:
                    final_solution = solutions[-1].strip()
                    parsed_responses.append(final_solution)
                else:
                    parsed_responses.append("[No solution found in output]")

            # Reshape results for the current batch
            reshaped_raw_batch = [clean_raw_text[i:i + n_trajectories] for i in range(0, len(clean_raw_text), n_trajectories)]
            reshaped_parsed_batch = [parsed_responses[i:i + n_trajectories] for i in range(0, len(parsed_responses), n_trajectories)]
            
            # --- MODIFICATION 3: Update the DataFrame by batch ---
            # It's important to convert to object type so pandas handles lists correctly
            dataset.loc[start_index:end_index-1, 'full_responses'] = pd.Series(reshaped_raw_batch, index=dataset.index[start_index:end_index]).astype(object)
            dataset.loc[start_index:end_index-1, 'result'] = pd.Series(reshaped_parsed_batch, index=dataset.index[start_index:end_index]).astype(object)

        else:
            num_prompts_in_batch = len(batch_chat_lst)
            placeholders = ["[Agent output key not found]" for _ in range(n_trajectories)]
            placeholder_list = [placeholders for _ in range(num_prompts_in_batch)]
            
            dataset.loc[start_index:end_index-1, 'full_responses'] = pd.Series(placeholder_list, index=dataset.index[start_index:end_index]).astype(object)
            dataset.loc[start_index:end_index-1, 'result'] = pd.Series(placeholder_list, index=dataset.index[start_index:end_index]).astype(object)

        # --- MODIFICATION 4: Save after each batch ---
        dataset.to_parquet(output_path)
        print(f"Batch {batch_idx+1}/{num_batch} complete. Results saved to {output_path}")

    print(f"Generation finished. Final results are saved to {output_path}")

if __name__ == '__main__':    
    main()

