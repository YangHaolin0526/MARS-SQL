# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import os

from verl.trainer.ppo.ray_trainer import RayPPOTrainer

import ray
import hydra


def get_custom_reward_fn(config):
    import importlib.util

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")

    return getattr(module, function_name)


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
    # isolation, will solve in the future
    if ray.is_initialized():
        ray.shutdown()
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={
                'env_vars': {
                    'TOKENIZERS_PARALLELISM': 'true',
                    'NCCL_DEBUG': 'WARN',
                    'VLLM_LOGGING_LEVEL': 'WARN',
                    **dict(os.environ)
                }
            })

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:

    def run(self, config):
        from verl.utils.fs import copy_to_local
        # print initial config
        from pprint import pprint
        from omegaconf import OmegaConf
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        # instantiate tokenizer
        from verl.utils import hf_tokenizer, hf_processor
        trust_remote_code = config.data.get('trust_remote_code', True)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray import RayWorkerGroup
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == 'megatron':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        if config.trainer.get("hybrid_engine", True):
            role_worker_mapping = {
                Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
                Role.Critic: ray.remote(CriticWorker),
                Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
            }

            global_pool_id = 'global_pool'
            resource_pool_spec = {
                global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            }
            mapping = {
                Role.ActorRollout: global_pool_id,
                Role.Critic: global_pool_id,
                Role.RefPolicy: global_pool_id,
            }
        else:
            role_worker_mapping = {
                Role.Actor: ray.remote(ActorRolloutRefWorker),
                Role.Critic: ray.remote(CriticWorker),
                Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
                Role.Rollout: ray.remote(ActorRolloutRefWorker),
            }

            placement = config.trainer.placement
            resource_pool_spec = {
                "actor_pool": [placement["actor"]] * config.trainer.nnodes,
                "ref_pool": [placement["ref"]] * config.trainer.nnodes,
                "critic_pool": [placement.get("critic", 0)] * config.trainer.nnodes,
                "rollout_pool": [placement["rollout"]] * config.trainer.nnodes,
            }
            mapping = {
                Role.Actor: "actor_pool",
                Role.Critic: "critic_pool",
                Role.RefPolicy: "ref_pool",
                Role.Rollout: "rollout_pool",
            }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy == 'fsdp':
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == 'megatron':
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        #use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        if reward_manager_name == 'naive':
            from verl.workers.reward_manager import NaiveRewardManager
            reward_manager_cls = NaiveRewardManager
        elif reward_manager_name == 'prime':
            from verl.workers.reward_manager import PrimeRewardManager
            reward_manager_cls = PrimeRewardManager
        elif reward_manager_name == 'dapo':
            from verl.workers.reward_manager import DAPORewardManager
            reward_manager_cls = DAPORewardManager
        elif reward_manager_name == 'swebench':
            from verl.workers.reward_manager import SWEBenchRewardManager
            reward_manager_cls = SWEBenchRewardManager
        elif reward_manager_name == 'sql':
            from verl.workers.reward_manager import SQLRewardManager
            reward_manager_cls = SQLRewardManager
        else:
            raise NotImplementedError

        compute_score = get_custom_reward_fn(config)
        reward_fn = reward_manager_cls(tokenizer=tokenizer,
                                       num_examine=0,
                                       config=config,
                                       compute_score=compute_score)

        # Note that we always use function-based RM for validation
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer,
                                           num_examine=1,
                                           config=config,
                                           compute_score=compute_score)
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RayPPOTrainer(config=config,
                                tokenizer=tokenizer,
                                processor=processor,
                                role_worker_mapping=role_worker_mapping,
                                resource_pool_manager=resource_pool_manager,
                                ray_worker_group_cls=ray_worker_group_cls,
                                reward_fn=reward_fn,
                                val_reward_fn=val_reward_fn)
        trainer.init_workers()
        trainer.fit()


if __name__ == '__main__':
    main()

# """
# 2 stage version of PPO trainer for schema linking and multi-turn text2sql training.
# """
# import os
# import re
# import sqlite3
# import numpy as np
# from typing import Set, Dict, List

# import ray
# import hydra
# import torch
# from omegaconf import OmegaConf
# from pprint import pprint

# from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
# from verl.utils.fs import copy_to_local
# from verl.utils import hf_tokenizer, hf_processor
# from verl import DataProto
# from tensordict import TensorDict
# import importlib.util


# # =================================================================================
# #  HELPER FUNCTIONS (No changes here)
# # =================================================================================

# def get_custom_reward_fn(config):
#     reward_fn_config = config.get("custom_reward_function") or {}
#     file_path = reward_fn_config.get("path")
#     if not file_path:
#         return None
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"Reward function file '{file_path}' not found.")
#     spec = importlib.util.spec_from_file_location("custom_module", file_path)
#     module = importlib.util.module_from_spec(spec)
#     try:
#         spec.loader.exec_module(module)
#     except Exception as e:
#         raise RuntimeError(f"Error loading module from '{file_path}': {e}")
#     function_name = reward_fn_config.get("name")
#     if not hasattr(module, function_name):
#         raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")
#     print(f"using customized reward function '{function_name}' from '{file_path}'")
#     return getattr(module, function_name)

# def get_create_statements(db_path: str) -> Dict[str, str]:
#     if not os.path.exists(db_path): return {}
#     try:
#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()
#         cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
#         result = {name.lower(): sql for name, sql in cursor.fetchall() if sql}
#         conn.close()
#         return result
#     except Exception: return {}

# def simplify_create_statement(create_sql: str, used_cols: Set[str]) -> str:
#     start = create_sql.find('(')
#     end = create_sql.rfind(')')
#     if start < 0 or end <= start:
#         return re.sub(r'\s+', ' ', create_sql).strip().rstrip(';') + ';'
#     header = create_sql[:start].strip()
#     cols_blob = create_sql[start+1:end]
#     parts = re.split(r',(?![^()]*\))', cols_blob)
#     col_defs = []
#     name_pattern = re.compile(r'^\s*(?:`([^`]+)`|"([^"]+)"|\[([^\]]+)\]|([A-Za-z_][\w]*))\s+', re.VERBOSE)
#     for part in parts:
#         part = part.strip()
#         if not part or part.upper().startswith(("PRIMARY KEY", "FOREIGN KEY", "UNIQUE", "CHECK", "CONSTRAINT")):
#             continue
#         m = name_pattern.match(part)
#         if m:
#             col_name = next(g for g in m.groups() if g is not None).lower()
#             if not used_cols or col_name in used_cols:
#                 col_defs.append(part)
#         else:
#             col_name_simple = part.split()[0].replace('`', '').replace('"', '').replace('[', '').replace(']', '').lower()
#             if not used_cols or col_name_simple in used_cols:
#                  col_defs.append(part)
#     cols_section = ', '.join(col_defs)
#     stmt = f"{header}({cols_section});"
#     return re.sub(r'\s+', ' ', stmt).strip()

# def build_schema_from_names(db_id: str, base_db_path: str, model_output: str) -> str:
#     try:
#         tables_match = re.search(r"used_tables\s*=\s*(\[.*?\])", model_output, re.DOTALL)
#         columns_match = re.search(r"used_columns\s*=\s*(\[.*?\])", model_output, re.DOTALL)
#         if not tables_match or not columns_match:
#             return f"[Error: Could not parse lists from: {model_output[:100]}]"
#         used_tables = eval(tables_match.group(1))
#         used_columns = eval(columns_match.group(1))
#     except Exception as e:
#         return f"[Error: Failed to eval model output. Details: {e}]"
#     sqlite_file = os.path.join(base_db_path, db_id, f"{db_id}.sqlite")
#     create_map = get_create_statements(sqlite_file)
#     if not create_map: return f"[Error: No CREATE statements for db_id '{db_id}']"
#     simplified_statements = []
#     used_tables_lower = {t.lower() for t in used_tables}
#     for tbl_lower in used_tables_lower:
#         full_sql = create_map.get(tbl_lower)
#         if not full_sql: continue
#         cols_for_this_table = {col.lower() for t, col in used_columns if t.lower() == tbl_lower}
#         stmt = simplify_create_statement(full_sql, cols_for_this_table)
#         simplified_statements.append(stmt)
#     return ' '.join(simplified_statements)

# def get_full_schema(db_id: str, base_db_path: str) -> str:
#     sqlite_file = os.path.join(base_db_path, db_id, f"{db_id}.sqlite")
#     create_map = get_create_statements(sqlite_file)
#     full_schema = " ".join(create_map.values())
#     return re.sub(r'\s+', ' ', full_schema).strip()

# # =================================================================================
# #  NEW: Smart Reward Manager
# # =================================================================================
# from verl.workers.reward_manager import SQLRewardManager

# class TwoStageRewardManager:
#     """
#     A "smart" reward manager that distinguishes between Stage 1 and Stage 2.
#     """
#     def __init__(self, tokenizer, config, compute_score, **kwargs):
#         print("Initializing TwoStageRewardManager...")
#         # Internally, it holds the real SQLRewardManager for Stage 2
#         self.sql_reward_manager = SQLRewardManager(
#             tokenizer=tokenizer,
#             config=config,
#             compute_score=compute_score,
#             **kwargs
#         )

#     def get_reward(self, data: DataProto):
#         # Check for our custom flag in the data
#         is_stage1 = data.non_tensor_batch.get('is_stage1', [False])[0]

#         if is_stage1:
#             # STAGE 1: This is just for generation. Return a neutral (zero) reward.
#             # print("Reward Manager: Detected Stage 1. Returning neutral reward.")
#             batch_size = data.batch['input_ids'].shape[0]
#             return torch.zeros(batch_size, device=data.batch['input_ids'].device)
#         else:
#             # STAGE 2: This is the real deal. Use the SQLRewardManager to get the actual reward.
#             # print("Reward Manager: Detected Stage 2. Calculating SQL reward.")
#             return self.sql_reward_manager.get_reward(data)

#     def __call__(self, data: DataProto):
#         return self.get_reward(data)

# # =================================================================================
# #  MAIN SCRIPT
# # =================================================================================

# @hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
# def main(config):
#     if ray.is_initialized():
#         ray.shutdown()
#     os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
#     ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
#     runner = TaskRunner.remote()
#     ray.get(runner.run_custom_training_loop.remote(config))

# @ray.remote(num_cpus=1)
# class TaskRunner:
#     def setup_workers(self, config):
#         """Initializes all necessary components and workers."""
#         pprint(OmegaConf.to_container(config, resolve=True))
#         OmegaConf.resolve(config)

#         local_path = copy_to_local(config.actor_rollout_ref.model.path)
#         self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
#         self.tokenizer.padding_side = 'left'
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
        
#         processor = hf_processor(local_path, use_fast=True)
        
#         from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
#         from verl.single_controller.ray import RayWorkerGroup

#         role_worker_mapping = {
#             Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
#             Role.Critic: ray.remote(CriticWorker),
#             Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
#         }
#         resource_pool_spec = {'global_pool': [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
#         mapping = {role: 'global_pool' for role in role_worker_mapping}
#         resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

#         # 核心改动：使用我们新的 TwoStageRewardManager
#         compute_score = get_custom_reward_fn(config)
#         self.smart_reward_fn = TwoStageRewardManager(
#             tokenizer=self.tokenizer,
#             config=config,
#             num_examine=1,
#             compute_score=compute_score
#         )
        
#         self.trainer = RayPPOTrainer(
#             config=config,
#             tokenizer=self.tokenizer,
#             processor=processor,
#             role_worker_mapping=role_worker_mapping,
#             resource_pool_manager=resource_pool_manager,
#             ray_worker_group_cls=RayWorkerGroup,
#             reward_fn=self.smart_reward_fn, # Pass the smart reward function here
#             val_reward_fn=self.smart_reward_fn
#         )
#         self.trainer.init_workers()

#     async def run_custom_training_loop(self, config):
#         """
#         This is the main custom training loop that orchestrates the two-stage process.
#         """
#         self.setup_workers(config)
        
#         # NOTE: In a real run, this loop would iterate over your actual dataloader
#         # for epoch in range(num_epochs):
#         #   for batch in dataloader:
#         #     ... (the logic below) ...
#         mock_data_batch = {
#             "prompt_linking": [[{"role": "user", "content": "Get the names of singers. Database: concert_singer"}]],
#             "prompt_sql_template": [[{"role": "user", "content": "Given the database schema: {linked_schema}\n\nGet the names of singers."}]],
#             "db_id": ["concert_singer"],
#             "ground_truth_sql": ["SELECT name FROM singer"]
#         }
        
#         # --- STAGE 1: Generate Schema Linking Lists via a "dummy" rollout ---
#         print("🚀 Stage 1: Generating schema linking lists...")
        
#         stage1_prompts = self.tokenizer.apply_chat_template(
#             mock_data_batch["prompt_linking"], add_generation_prompt=True, padding=True, return_tensors='pt'
#         )

#         # Prepare data with the 'is_stage1' flag
#         stage1_data = DataProto(
#             batch=TensorDict({
#                 'input_ids': stage1_prompts['input_ids'],
#                 'attention_mask': stage1_prompts['attention_mask']
#             }, batch_size=len(mock_data_batch["prompt_linking"])),
#             non_tensor_batch={'is_stage1': np.array([True])}
#         )

#         # Call trainer.rollout(). Our smart reward function will return 0 for this step.
#         # We only care about the generated text. We will NOT learn from these experiences.
#         # NOTE: The generation parameters (greedy/sample) are now controlled by your main hydra config.
#         # For this stage, greedy (`do_sample: false`) is recommended.
#         stage1_experiences = self.trainer.rollout(stage1_data)

#         # Safely extract the generated text
#         response_ids = stage1_experiences.batch.get('responses')
#         if response_ids is None:
#             raise ValueError("Could not find 'responses' in the experiences from Stage 1 rollout.")
        
#         input_len = stage1_prompts['input_ids'].shape[1]
#         stage1_texts = self.tokenizer.batch_decode(response_ids[:, input_len:], skip_special_tokens=True)
#         print(f"✅ Stage 1 Generated Text: {stage1_texts[0]}")

#         # --- Intermediate Step: Build Schema and Prepare Stage 2 Prompts ---
#         stage2_prompts_text = []
#         base_db_path = config.data.base_db_path 
#         for i, text in enumerate(stage1_texts):
#             db_id = mock_data_batch["db_id"][i]
#             linked_schema = build_schema_from_names(db_id, base_db_path, text)
            
#             if linked_schema.startswith("[Error"):
#                 print(f"⚠️ Schema linking failed for {db_id}. Reason: {linked_schema}. Falling back to full schema.")
#                 linked_schema = get_full_schema(db_id, base_db_path)
            
#             template = mock_data_batch["prompt_sql_template"][i]
#             template[0]['content'] = template[0]['content'].format(linked_schema=linked_schema)
#             stage2_prompts_text.append(template)
        
#         print(f"✅ Stage 2 Prompt ready: {stage2_prompts_text[0][0]['content']}")

#         # --- STAGE 2: Generate Final SQL and Collect REAL Experiences ---
#         print("\n🚀 Stage 2: Performing PPO rollout for SQL generation...")
        
#         stage2_inputs = self.tokenizer.apply_chat_template(
#             stage2_prompts_text, add_generation_prompt=True, padding=True, return_tensors='pt',
#             truncation=True, max_length=config.rollout.prompt_length
#         )

#         # Prepare data for Stage 2. Note the ABSENCE of the 'is_stage1' flag.
#         stage2_data = DataProto(
#             batch=TensorDict({
#                 'input_ids': stage2_inputs['input_ids'],
#                 'attention_mask': stage2_inputs['attention_mask']
#             }, batch_size=len(stage2_prompts_text)),
#             non_tensor_batch={
#                 'raw_prompt': np.array(stage2_prompts_text, dtype=object),
#                 'ground_truth': np.array(mock_data_batch['ground_truth_sql'], dtype=object)
#             }
#         )

#         # This rollout will use the smart reward function to get a REAL reward.
#         final_experiences = self.trainer.rollout(stage2_data)
        
#         # --- STEP 3: PPO Learning Step ---
#         print("\n🚀 Step 3: Running PPO learning step...")
        
#         # Learn ONLY from the experiences of the final, meaningful stage.
#         metrics = self.trainer.learn(final_experiences)
        
#         print("\n✅ Training step complete!")
#         pprint(metrics)


# if __name__ == '__main__':
#     main()


# """
# 2 stage version of PPO trainer for schema linking and multi-turn text2sql training. stage 1 也有reward bonus.
# """
# import os
# import re
# import sqlite3
# import numpy as np
# from typing import Set, Dict, List
# import importlib.util

# from verl.trainer.ppo.ray_trainer import RayPPOTrainer

# import ray
# import hydra


# def get_custom_reward_fn(config):
#     # import importlib.util

#     reward_fn_config = config.get("custom_reward_function") or {}
#     file_path = reward_fn_config.get("path")
#     if not file_path:
#         return None

#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

#     spec = importlib.util.spec_from_file_location("custom_module", file_path)
#     module = importlib.util.module_from_spec(spec)
#     try:
#         spec.loader.exec_module(module)
#     except Exception as e:
#         raise RuntimeError(f"Error loading module from '{file_path}': {e}")

#     function_name = reward_fn_config.get("name")

#     if not hasattr(module, function_name):
#         raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

#     print(f"using customized reward function '{function_name}' from '{file_path}'")

#     return getattr(module, function_name)


# # --- Schema building helper functions (from main_generation.py) ---
# def get_create_statements(db_path: str) -> Dict[str, str]:
#     """
#     Read all CREATE TABLE statements from the SQLite file, return mapping table -> SQL.
#     """
#     if not os.path.exists(db_path):
#         print(f"Warning: SQLite file not found at {db_path}")
#         return {}
#     try:
#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()
#         cursor.execute(
#             "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
#         )
#         result = {name.lower(): sql for name, sql in cursor.fetchall() if sql}
#         conn.close()
#         return result
#     except Exception as e:
#         print(f"Warning: Could not read schema from {db_path}. Error: {e}")
#         return {}


# def minify_sql_schema(sql_text: str) -> str:
#     """
#     Compresses the SQL schema string by removing unnecessary spaces, newlines, and comments to reduce token count.
#     """
#     lines = sql_text.splitlines()
#     processed_lines = []
#     for line in lines:
#         stripped_line = line.strip()
#         if stripped_line and not stripped_line.startswith('--'):
#             processed_lines.append(stripped_line)
#     single_line_schema = ' '.join(processed_lines)
#     minified_schema = re.sub(r'\s+', ' ', single_line_schema)
#     return minified_schema.strip()


# def simplify_create_statement(create_sql: str, used_cols: Set[str]) -> str:
#     """
#     Simplifies a CREATE TABLE statement to only include used columns and removes constraints.
#     """
#     start = create_sql.find('(')
#     end = create_sql.rfind(')')
#     if start < 0 or end <= start:
#         return re.sub(r'\s+', ' ', create_sql).strip().rstrip(';') + ';'

#     header = create_sql[:start].strip()
#     cols_blob = create_sql[start+1:end]
    
#     # Split by comma only at the top level (handles nested parentheses in CHECK constraints etc.)
#     parts = re.split(r',(?![^()]*\))', cols_blob)
    
#     col_defs = []
#     name_pattern = re.compile(r"""
#         ^\s*
#         (?: 
#           `([^`]+)`     |
#           "([^"]+)"     |
#           \[([^\]]+)\]  |
#           ([A-Za-z_][\w]*)
#         )
#         \s+
#     """, re.VERBOSE)

#     for part in parts:
#         part = part.strip()
#         if not part:
#             continue
#         up = part.upper()
#         if up.startswith(("PRIMARY KEY", "FOREIGN KEY", "UNIQUE", "CHECK", "CONSTRAINT")):
#             continue
        
#         m = name_pattern.match(part)
#         if not m:
#             # Fallback for columns without explicit types or other formats
#             col_name_simple = part.split()[0].replace('`', '').replace('"', '').replace('[', '').replace(']', '').lower()
#             if not used_cols or col_name_simple in used_cols:
#                  col_defs.append(part)
#             continue

#         col_name = next(g for g in m.groups() if g is not None).lower()
#         if not used_cols or col_name in used_cols:
#             col_defs.append(part)

#     cols_section = ', '.join(col_defs)
#     stmt = f"{header}({cols_section});"
#     return re.sub(r'\s+', ' ', stmt).strip()


# def build_schema_from_names(db_id: str, base_db_path: str, model_output: str) -> str:
#     """
#     The main "glue" function. It parses the model's output to get table/column lists,
#     then uses the helper functions to build the simplified CREATE TABLE statements.
#     """
#     try:
#         # Extract the lists from the model's output string
#         tables_match = re.search(r"used_tables\s*=\s*(\[.*?\])", model_output, re.DOTALL)
#         columns_match = re.search(r"used_columns\s*=\s*(\[.*?\])", model_output, re.DOTALL)

#         if not tables_match or not columns_match:
#             return "[Error: Could not parse lists from model output]"

#         # Safely evaluate the list strings into Python lists
#         used_tables = eval(tables_match.group(1))
#         used_columns = eval(columns_match.group(1))
        
#         # Ensure they are lists of strings/tuples
#         if not isinstance(used_tables, list) or not isinstance(used_columns, list):
#              return "[Error: Parsed output is not a list]"

#     except Exception as e:
#         return f"[Error: Failed to eval model output lists. Details: {e}]"

#     sqlite_file = os.path.join(base_db_path, db_id, f"{db_id}.sqlite")
#     create_map = get_create_statements(sqlite_file)
#     if not create_map:
#         return f"[Error: Could not retrieve CREATE statements for db_id '{db_id}']"

#     simplified_statements = []
#     used_tables_lower = {t.lower() for t in used_tables}

#     for tbl_lower in used_tables_lower:
#         full_sql = create_map.get(tbl_lower)
#         if not full_sql:
#             continue
        
#         # Collect all columns the model identified for this specific table
#         cols_for_this_table = {col.lower() for t, col in used_columns if t.lower() == tbl_lower}
        
#         stmt = simplify_create_statement(full_sql, cols_for_this_table)
#         simplified_statements.append(stmt)

#     return ' '.join(simplified_statements)


# @hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
# def main(config):
#     # Determine which stage to run based on config
#     stage = config.trainer.get('stage', 'stage1')  # Default to stage1 if not specified
    
#     if stage == 'stage1':
#         print("Running Stage 1: Schema Linking PPO Training")
#         run_ppo_stage1(config)
#     elif stage == 'stage2':
#         print("Running Stage 2: Multi-turn Text2SQL PPO Training")
#         run_ppo_stage2(config)
#     else:
#         raise ValueError(f"Unknown stage: {stage}. Must be 'stage1' or 'stage2'")


# def run_ppo_stage1(config) -> None:
#     """
#     Stage 1: PPO training for schema linking (table/column identification)
#     """
#     print("=== PPO Stage 1: Schema Linking Training ===")
    
#     # Use schema linking specific reward manager
#     if not config.reward_model.get("reward_manager"):
#         print("Setting reward manager to 'naive' for Stage 1")
#         config.reward_model.reward_manager = "naive"
    
#     # Run standard PPO training
#     _run_ppo_core(config)


# def run_ppo_stage2(config) -> None:
#     """
#     Stage 2: PPO training for multi-turn text2sql with schema linking integration
#     """
#     print("=== PPO Stage 2: Multi-turn Text2SQL Training ===")
    
#     # Check if we have a stage1 model path for initialization
#     stage1_model_path = config.trainer.get('stage1_model_path')
#     if stage1_model_path:
#         print(f"Using Stage 1 model for initialization: {stage1_model_path}")
#         # Update the model path to use stage1 trained model
#         config.actor_rollout_ref.model.path = stage1_model_path
    
#     # Use two-stage reward manager that integrates schema linking
#     if not config.reward_model.get("reward_manager"):
#         print("Setting reward manager to 'sql' for Stage 2")
#         config.reward_model.reward_manager = "sql"
    
#     # Add schema linking integration to the reward manager
#     config.reward_model.use_schema_linking = True
#     config.reward_model.base_db_path = config.data.get('base_db_path', '')
    
#     # Run PPO training with enhanced reward function
#     _run_ppo_core(config)


# def _run_ppo_core(config) -> None:
#     """
#     Core PPO training logic shared by both stages
#     """
#     # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
#     # isolation, will solve in the future
#     if ray.is_initialized():
#         ray.shutdown()
#     os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
#     if not ray.is_initialized():
#         # this is for local ray cluster
#         ray.init(
#             runtime_env={
#                 'env_vars': {
#                     'TOKENIZERS_PARALLELISM': 'true',
#                     'NCCL_DEBUG': 'WARN',
#                     'VLLM_LOGGING_LEVEL': 'WARN',
#                     **dict(os.environ)
#                 }
#             })

#     runner = TaskRunner.remote()
#     ray.get(runner.run.remote(config))


# @ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
# class TaskRunner:

#     def run(self, config):
#         from verl.utils.fs import copy_to_local
#         # print initial config
#         from pprint import pprint
#         from omegaconf import OmegaConf
#         pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
#         OmegaConf.resolve(config)

#         # download the checkpoint from hdfs
#         local_path = copy_to_local(config.actor_rollout_ref.model.path)
#         # instantiate tokenizer
#         from verl.utils import hf_tokenizer, hf_processor
#         trust_remote_code = config.data.get('trust_remote_code', True)
#         tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
#         processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

#         # define worker classes
#         if config.actor_rollout_ref.actor.strategy == 'fsdp':
#             assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
#             from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
#             from verl.single_controller.ray import RayWorkerGroup
#             ray_worker_group_cls = RayWorkerGroup

#         elif config.actor_rollout_ref.actor.strategy == 'megatron':
#             assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
#             from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
#             from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
#             ray_worker_group_cls = NVMegatronRayWorkerGroup

#         else:
#             raise NotImplementedError

#         from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

#         if config.trainer.get("hybrid_engine", True):
#             role_worker_mapping = {
#                 Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
#                 Role.Critic: ray.remote(CriticWorker),
#                 Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
#             }

#             global_pool_id = 'global_pool'
#             resource_pool_spec = {
#                 global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
#             }
#             mapping = {
#                 Role.ActorRollout: global_pool_id,
#                 Role.Critic: global_pool_id,
#                 Role.RefPolicy: global_pool_id,
#             }
#         else:
#             role_worker_mapping = {
#                 Role.Actor: ray.remote(ActorRolloutRefWorker),
#                 Role.Critic: ray.remote(CriticWorker),
#                 Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
#                 Role.Rollout: ray.remote(ActorRolloutRefWorker),
#             }

#             placement = config.trainer.placement
#             resource_pool_spec = {
#                 "actor_pool": [placement["actor"]] * config.trainer.nnodes,
#                 "ref_pool": [placement["ref"]] * config.trainer.nnodes,
#                 "critic_pool": [placement.get("critic", 0)] * config.trainer.nnodes,
#                 "rollout_pool": [placement["rollout"]] * config.trainer.nnodes,
#             }
#             mapping = {
#                 Role.Actor: "actor_pool",
#                 Role.Critic: "critic_pool",
#                 Role.RefPolicy: "ref_pool",
#                 Role.Rollout: "rollout_pool",
#             }

#         # we should adopt a multi-source reward function here
#         # - for rule-based rm, we directly call a reward score
#         # - for model-based rm, we call a model
#         # - for code related prompt, we send to a sandbox if there are test cases
#         # - finally, we combine all the rewards together
#         # - The reward type depends on the tag of the data
#         if config.reward_model.enable:
#             if config.reward_model.strategy == 'fsdp':
#                 from verl.workers.fsdp_workers import RewardModelWorker
#             elif config.reward_model.strategy == 'megatron':
#                 from verl.workers.megatron_workers import RewardModelWorker
#             else:
#                 raise NotImplementedError
#             role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
#             mapping[Role.RewardModel] = global_pool_id

#         #use reference model
#         if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
#             role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
#             mapping[Role.RefPolicy] = global_pool_id

#         reward_manager_name = config.reward_model.get("reward_manager", "naive")
#         if reward_manager_name == 'naive':
#             from verl.workers.reward_manager import NaiveRewardManager
#             reward_manager_cls = NaiveRewardManager
#         elif reward_manager_name == 'prime':
#             from verl.workers.reward_manager import PrimeRewardManager
#             reward_manager_cls = PrimeRewardManager
#         elif reward_manager_name == 'dapo':
#             from verl.workers.reward_manager import DAPORewardManager
#             reward_manager_cls = DAPORewardManager
#         elif reward_manager_name == 'swebench':
#             from verl.workers.reward_manager import SWEBenchRewardManager
#             reward_manager_cls = SWEBenchRewardManager
#         elif reward_manager_name == 'sql':
#             from verl.workers.reward_manager import SQLRewardManager
#             reward_manager_cls = SQLRewardManager
#         elif reward_manager_name == 'two_stage_sql':
#             # New reward manager for two-stage training
#             reward_manager_cls = TwoStageSQLRewardManager
#         else:
#             raise NotImplementedError

#         compute_score = get_custom_reward_fn(config)
        
#         # Enhanced reward function for Stage 2
#         if config.trainer.get('stage') == 'stage2':
#             reward_fn = TwoStageRewardWrapper(
#                 base_reward_manager=reward_manager_cls,
#                 tokenizer=tokenizer,
#                 config=config,
#                 compute_score=compute_score,
#                 base_db_path=config.data.get('base_db_path', '')
#             )
#             val_reward_fn = TwoStageRewardWrapper(
#                 base_reward_manager=reward_manager_cls,
#                 tokenizer=tokenizer,
#                 config=config,
#                 compute_score=compute_score,
#                 base_db_path=config.data.get('base_db_path', ''),
#                 num_examine=1
#             )
#         else:
#             # Standard reward function for Stage 1
#             reward_fn = reward_manager_cls(tokenizer=tokenizer,
#                                            num_examine=0,
#                                            config=config,
#                                            compute_score=compute_score)

#             val_reward_fn = reward_manager_cls(tokenizer=tokenizer,
#                                                num_examine=1,
#                                                config=config,
#                                                compute_score=compute_score)
        
#         resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

#         trainer = RayPPOTrainer(config=config,
#                                 tokenizer=tokenizer,
#                                 processor=processor,
#                                 role_worker_mapping=role_worker_mapping,
#                                 resource_pool_manager=resource_pool_manager,
#                                 ray_worker_group_cls=ray_worker_group_cls,
#                                 reward_fn=reward_fn,
#                                 val_reward_fn=val_reward_fn)
#         trainer.init_workers()
#         trainer.fit()


# class TwoStageRewardWrapper:
#     """
#     Wrapper for Stage 2 reward function that integrates schema linking
#     """
#     def __init__(self, base_reward_manager, tokenizer, config, compute_score, base_db_path, num_examine=0):
#         self.base_reward_manager = base_reward_manager(
#             tokenizer=tokenizer,
#             num_examine=num_examine,
#             config=config,
#             compute_score=compute_score
#         )
#         self.tokenizer = tokenizer
#         self.config = config
#         self.base_db_path = base_db_path
#         self.num_examine = num_examine
        
#     def __call__(self, data):
#         """
#         Enhanced reward calculation for Stage 2
#         """
#         # First, get base rewards from the underlying reward manager
#         base_rewards = self.base_reward_manager(data)
        
#         # For Stage 2, we can add schema linking quality bonus
#         if hasattr(data, 'non_tensor_batch') and 'db_id' in data.non_tensor_batch:
#             db_ids = data.non_tensor_batch['db_id']
#             responses = data.batch.get('responses', None)
            
#             if responses is not None:
#                 # Decode responses to check schema linking quality
#                 response_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
                
#                 # Calculate schema linking bonus
#                 schema_bonuses = []
#                 for i, (response_text, db_id) in enumerate(zip(response_texts, db_ids)):
#                     bonus = self._calculate_schema_bonus(response_text, db_id)
#                     schema_bonuses.append(bonus)
                
#                 # Combine base rewards with schema bonuses
#                 enhanced_rewards = base_rewards + np.array(schema_bonuses) * self.config.reward_model.get('schema_bonus_weight', 0.1)
#                 return enhanced_rewards
        
#         return base_rewards
    
#     def _calculate_schema_bonus(self, response_text, db_id):
#         """
#         Calculate bonus reward based on schema linking quality
#         """
#         try:
#             # Build schema from model output
#             schema_result = build_schema_from_names(db_id, self.base_db_path, response_text)
            
#             # Give bonus if schema building was successful
#             if not schema_result.startswith("[Error") and schema_result.strip():
#                 return 1.0  # Positive bonus for successful schema linking
#             else:
#                 return -0.5  # Penalty for failed schema linking
#         except Exception as e:
#             return -0.5  # Penalty for exceptions
    
#     def get_reward(self, data):
#         """Compatibility method"""
#         return self(data)


# if __name__ == '__main__':
#     main()