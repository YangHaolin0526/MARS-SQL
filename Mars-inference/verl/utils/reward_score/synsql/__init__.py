import multiprocessing as mp
from typing import Optional
from pathlib import Path

from datasets.utils.py_utils import multiprocess 
from .utils import execute_sql_wrapper
from time import perf_counter
import re
import random 
import logging
import json 

THINK_START, THINK_END = "<think>", "</think>"
SQL_START, SQL_END = "<sql>", "</sql>"
SOLUTION_START, SOLUTION_END = "<solution>", "</solution>"
OBS_START, OBS_END = "<observation>", "</observation>"

def verify_format_and_extract(output: str):
    # Attempt to extract SQL from markdown code block first
    markdown_sql_match = re.search(r"```sql\s*\n(.*?)\n```", output, re.DOTALL | re.IGNORECASE)

    if markdown_sql_match:
        pred_sql = markdown_sql_match.group(1).strip()
        # For this new format, try to find thoughts if they exist anywhere in the output.
        # If your new format *requires* <think> tags, this logic is fine.
        # If thoughts are optional or not present for markdown SQL, adjust accordingly.
        thoughts = re.findall(r"<think>(.*?)</think>", output, re.DOTALL | re.IGNORECASE)
        if not thoughts:
            # If markdown SQL is primary, decide if thoughts are mandatory.
            # If optional, this placeholder or returning an empty list might be okay.
            # If mandatory, you might return False here.
            thoughts = ["No explicit <think> tags found with markdown SQL."] # Placeholder or []

        # If markdown SQL is found, we consider this a valid extraction path.
        # The 'solution_text' concept from the original XML format might not apply here, so return None.
        return True, thoughts, pred_sql, None

    # --- Fallback to original XML-like tag parsing if markdown SQL block is NOT found ---

    # Check for mandatory <solution> tags
    if output.count(SOLUTION_START) != 1 or output.count(SOLUTION_END) != 1:
        return False, None, None, None # Invalid format: missing or multiple <solution> tags

    pre_solution, tail = output.split(SOLUTION_START, 1)
    solution_text, _ = tail.split(SOLUTION_END, 1) # _ captures content after </solution>

    # Check if the content *within* <solution> ... </solution> contains other control tags
    if re.search(r"</?(think|sql|observation)\b", solution_text, re.IGNORECASE):
        return False, None, None, None # Invalid: <solution> block contains other control tags

    # Extract thoughts from before the <solution> block
    thoughts = re.findall(r"<think>(.*?)</think>", pre_solution, re.DOTALL | re.IGNORECASE) # Use re.DOTALL for multi-line thoughts
    if not thoughts:
        return False, None, None, None # Invalid format: no <think> tags found before <solution>

    # Validate that content between </observation> and <solution> (if any) must be <think>
    # This ensures thoughts immediately precede the solution if there were prior observations.
    # Or, if there are no observations, thoughts just need to be somewhere in pre_solution.
    last_obs_end = -1
    for m_obs in re.finditer(r"</observation>", pre_solution, re.IGNORECASE):
        last_obs_end = m_obs.end()

    if last_obs_end != -1: # If there was at least one observation
        content_after_last_obs = pre_solution[last_obs_end:].lstrip()
        if content_after_last_obs and not content_after_last_obs.lower().startswith(THINK_START):
            # Content exists after the last observation but before <solution>, and it's not a <think> block.
            return False, None, None, None

    # Try to extract SQL from <sql> tags if they exist *within pre_solution*
    # This assumes SQL might appear before the <solution> block in the original format.
    sql_matches = re.findall(r"<sql>(.*?)</sql>", pre_solution, re.DOTALL | re.IGNORECASE) # Use re.DOTALL
    pred_sql = sql_matches[-1].strip() if sql_matches else None # Use the last SQL found before solution

    # For the original XML format, if no <sql> tag yielded SQL, it's considered invalid for execution.
    if pred_sql is None:
        return False, thoughts, None, solution_text # No SQL extracted for this path

    return True, thoughts, pred_sql, solution_text.strip() # Return stripped solution_text

    
def calculate_reward_parallel(db_files, completions, references, questions, num_cpus=32, timeout=30, n_agent: Optional[int] = None, log_dir: Optional[str] = None):
    """
    Calculate rewards in parallel for SynSQL.
    
    Args:
        db_files: List of database files to execute the SQL queries.
        completions: List of model outputs containing the SQL solutions.
        references: List of ground truth SQL queries.
        num_cpus: Number of CPU cores to use for parallel processing.
        timeout: Timeout for each SQL execution.

    Returns:
        List of rewards for each completion.
    """
    if log_dir:
        assert n_agent is not None, "n_agent must be provided for logging"
    start = perf_counter()
    print(f"synsql_utils::calculate_reward_parallel: calculating {len(completions)} rewards", flush=True)

    # serial filter for format reward
    rewards = [0.0] * len(completions)
    num_comparisons = 0
    to_execute = []
    
    # serially filter for format reward
    for i, output in enumerate(completions):
        is_valid, _, pred_sql, _ = verify_format_and_extract(output)
        if not is_valid:
            rewards[i] = -1.0
        else:
            num_comparisons += 1
            to_execute.append((i, db_files[i], pred_sql, timeout, output))
            to_execute.append((i, db_files[i], references[i], timeout, output))
    
    if len(to_execute) == 0:
        print(f"[DEBUG]: All format wrong, completions: {completions}")
        
    # parallely execute for correctness reward
    exec_start = perf_counter()
    print(f"synsql_utils::calculate_reward_parallel: executing {len(to_execute)} SQL statements in parallel", flush=True)
    # NOTE (sumanthrh): We use mp context instead of the global context to avoid changing the default start method
    # this can affect dataloading code since PyTorch uses fork by default.
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_cpus) as pool:
        results = pool.starmap(execute_sql_wrapper, to_execute)
    exec_end = perf_counter()
    print(f"synsql_utils::calculate_reward_parallel: executed {len(to_execute)} SQL statements in {exec_end - exec_start:.2f} seconds", flush=True)

    # evaluate the results
    # NOTE(shu): for printing purpose 
    correct_examples = []
    wrong_examples = [] 
    for i in range(num_comparisons):
        idx, _, p_sql, pred_results, _, pred_completion = results[i * 2]
        _, _, g_sql, gt_results, _, _ = results[i * 2 + 1]
        
        if pred_results is not None and gt_results is not None and pred_results == gt_results:
            rewards[idx] = 1.0
            correct_examples.append((idx, p_sql, g_sql, pred_completion))
            # print(f"[DEBUG-SHU-EXECUTE]: CORRECT, SQL is {p_sql}", flush=True)
        else:
            rewards[idx] = 0.0
            wrong_examples.append((idx, p_sql, g_sql, pred_completion))
            # print(f"[DEBUG-SHU-EXECUTE]: WRONG, SQL is {p_sql}, \nGOLD SQL is {g_sql}", flush=True)
    
    # log to directory
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        for index in range(len(completions)):
            traj_file = log_dir / f"traj_{index}.json"
            traj_data = {"completion": completions[index], "db_file": db_files[index], "reference": references[index], "reward": rewards[index], "question": questions[index]}
            with open(traj_file, "w") as f:
                json.dump(traj_data, f, default=lambda x: str(x))

    end = perf_counter()
    print(f"synsql_utils::calculate_reward_parallel: calculated {len(completions)} rewards in {end - start:.2f} seconds", flush=True)
    return rewards