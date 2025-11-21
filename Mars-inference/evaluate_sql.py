#16 trajectory
import argparse
import pandas as pd
import re
import os
import sys
import signal
import multiprocessing as mp
from time import perf_counter
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple, Any
import sqlite3
import ast
from collections import Counter

# --- Add new dependencies for parallel execution ---
try:
    from func_timeout import func_timeout, FunctionTimedOut
except ImportError:
    print("Warning: 'func_timeout' library not installed. Please run 'pip install func-timeout'.")
    func_timeout, FunctionTimedOut = None, None

# --- Data Extraction Functions ---
def extract_candidate_sqls(row, n=16):
    """
    从 row['result'] 和 row['full_responses'] 中分别取出 n 条 SQL 候选，
    优先用 result[i]（若以 SELECT 开头），否则从 full_responses[i] 中抽 <sql> 标签。
    """
    candidates = []
    results = row.get('result', [])
    fulls   = row.get('full_responses', [])

    for i in range(n):
        sql = None
        # 1) 如果 result[i] 是合法 SELECT，就用它
        if i < len(results) and isinstance(results[i], str) and results[i].strip().upper().startswith("SELECT"):
            sql = results[i].strip().strip('"')
        # 2) 否则在 full_responses[i] 里匹配 <sql>…</sql>
        elif i < len(fulls) and isinstance(fulls[i], str):
            blocks = re.findall(r"<sql>(.*?)</sql>", fulls[i], re.DOTALL)
            if blocks:
                sql = blocks[-1].strip()
        candidates.append(sql)
    return candidates


def extract_ground_truth_sql(row):
    """Safely extracts the ground_truth SQL from the reward_model column."""
    if 'reward_model' not in row or pd.isna(row['reward_model']):
        return None
    
    reward_model_data = row['reward_model']
    if isinstance(reward_model_data, dict):
        return reward_model_data.get('ground_truth')

    reward_model_string = str(reward_model_data)
    try:
        reward_dict = ast.literal_eval(reward_model_string)
        if isinstance(reward_dict, dict):
            return reward_dict.get('ground_truth')
    except (ValueError, SyntaxError, TypeError):
        match = re.search(r"['\"]ground_truth['\"]\s*:\s*['\"](.*?)['\"]", reward_model_string)
        if match:
            return match.group(1)
    return None

# --- Parallel SQL Execution Functions ---

def execute_sql(data_idx, db_file, sql):
    if not sql: return data_idx, db_file, sql, "SQL query is empty", 0
    try:
        conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
        cursor = conn.cursor()
        conn.execute("BEGIN TRANSACTION;")
        cursor.execute(sql)
        execution_res = frozenset(cursor.fetchall())
        conn.rollback()
        conn.close()
        return data_idx, db_file, sql, execution_res, 1
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        if 'conn' in locals() and conn:
            conn.rollback()
            conn.close()
        return data_idx, db_file, sql, error_msg, 0

def execute_sql_wrapper(data_idx, db_file, sql, timeout):
    if func_timeout is None:
        raise ImportError("Please run 'pip install func-timeout' to use the timeout feature.")
    try:
        res = func_timeout(timeout, execute_sql, args=(data_idx, db_file, sql))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        res = (data_idx, db_file, sql, "Function TimeOut", 0)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        res = (data_idx, db_file, sql, error_msg, 0)
    return res
    
def execute_sqls_parallel(db_files, pred_sqls, num_cpus=48, timeout=120):
    """Executes a list of SQL statements in parallel."""
    args = [
        (idx, db_file, sql, timeout)
        for idx, (db_file, sql) in enumerate(zip(db_files, pred_sqls))
    ]
    print(f"--> [Parallel Execution]: Preparing to execute {len(args)} SQL statements...")
    start = perf_counter()
    original_sigchld_handler = signal.getsignal(signal.SIGCHLD)
    try:
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        mp.set_start_method("spawn", force=True)
        with mp.Pool(processes=num_cpus) as pool:
            results_list = list(tqdm(pool.starmap(execute_sql_wrapper, args), total=len(args), desc="Executing SQL in parallel"))
            results = sorted(results_list, key=lambda x: x[0])
    finally:
        signal.signal(signal.SIGCHLD, original_sigchld_handler)
    end = perf_counter()
    print(f"--> [Parallel Execution]: Completed in {end - start:.2f} seconds.")
    return results

def check_row_match(pred_row: Tuple, gt_row: Tuple) -> bool:
    """
    Checks if a single predicted row matches a ground truth row loosely.
    
    Criteria for Match:
    1. Strict Equality.
    2. Permutation: ('A', 1) matches (1, 'A').
    3. Subset: ('A',) matches ('A', 1). (Prediction is a subset of GT).
       Note: We convert to set/counter to ignore order.
    """
    # Convert to list to handle internal logic
    p_list = list(pred_row)
    g_list = list(gt_row)

    # Case 1: Exact match (most common, fastest check)
    if pred_row == gt_row:
        return True

    p_counter = Counter(p_list)
    g_counter = Counter(g_list)
    
    # Check if p_counter is a "sub-counter" of g_counter
    # i.e., for every key in p, count(p) <= count(g)
    for key, count in p_counter.items():
        if g_counter[key] < count:
            return False
            
    return True

def advanced_result_compare(res1: Optional[frozenset], res2: Optional[frozenset]) -> bool:
    """
    Compares two SQL execution results with lenient rules.
    res1: Prediction
    res2: Ground Truth
    """
    if res1 is None or res2 is None:
        return False
    
    # If they are identical objects or empty
    if res1 == res2:
        return True

    # Convert to lists for manipulation
    pred_rows = list(res1)
    gt_rows = list(res2)

    # Metric: Row count must be equal.
    if len(pred_rows) != len(gt_rows):
        return False
    
    # Make a copy of GT rows to track consumption
    available_gt_indices = set(range(len(gt_rows)))

    for p_row in pred_rows:
        match_found_for_this_row = False
        
        for g_idx in list(available_gt_indices): # iterate copy or handle safe removal
            g_row = gt_rows[g_idx]
            
            if check_row_match(p_row, g_row):
                match_found_for_this_row = True
                available_gt_indices.remove(g_idx)
                break
        
        if not match_found_for_this_row:
            # If any prediction row cannot find a matching GT row, fail.
            return False

    return True

def format_result_for_logging(result, max_rows=10):
    """Formats an execution result for logging."""
    if isinstance(result, (frozenset, set, list)):
        result_list = list(result)
        if not result_list: return "[]"
        preview = result_list[:max_rows]
        log_str = repr(preview)
        if len(result_list) > max_rows:
            log_str = log_str[:-1] + f", ... ({len(result_list) - max_rows} more rows)]"
        return log_str
    else:
        return repr(result)

def main(args):
    """Main execution workflow."""
    print(f"--> Reading Parquet file: {args.input_file}")
    try:
        df = pd.read_parquet(args.input_file)
        print(f"    Found {len(df)} total entries to evaluate.")
    except Exception as e:
        print(f"Error: Could not read the input file. {e}")
        return

    # Step 1: Collect all SQL queries
    print("--> Step 1: Collecting all ground truth and candidate SQL queries...")
    db_files_to_run, sqls_to_run, job_metadata = [], [], []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Collecting jobs"):
        db_id = row.get('db_id')
        if not db_id: continue
        db_path = os.path.join(args.db_path, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path): continue

        gt_sql = extract_ground_truth_sql(row)
        db_files_to_run.append(db_path)
        sqls_to_run.append(gt_sql)
        job_metadata.append({'df_index': index, 'type': 'ground_truth'})

        candidate_sqls = extract_candidate_sqls(row, n=16)
        for i, c_sql in enumerate(candidate_sqls):
            db_files_to_run.append(db_path)
            sqls_to_run.append(c_sql)
            job_metadata.append({'df_index': index, 'type': 'candidate', 'candidate_idx': i})

    # Step 2: Execute all queries in parallel
    parallel_results = execute_sqls_parallel(
        db_files_to_run, sqls_to_run, num_cpus=args.num_cpus, timeout=args.timeout
    )

    # Step 3: Reorganize results
    print("--> Step 2: Reorganizing SQL execution results...")
    execution_results_map = {}
    for i, result_tuple in enumerate(parallel_results):
        meta = job_metadata[i]
        df_index = meta['df_index']
        _, _, _, execution_res, success_flag = result_tuple
        
        if df_index not in execution_results_map:
            execution_results_map[df_index] = {'ground_truth': None, 'candidates': [None]*16}
        
        if meta['type'] == 'ground_truth':
            execution_results_map[df_index]['ground_truth'] = (execution_res, success_flag)
        else:
            c_idx = meta['candidate_idx']
            execution_results_map[df_index]['candidates'][c_idx] = (execution_res, success_flag)
            
    # Step 4: Process results
    print("--> Step 3: Evaluating Best-of-16 matches (with Lenient Rules)...")
    correct_predictions = 0
    total_evaluated = 0

    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        f_out.write("Index\tDB_ID\tMatch_Status\tGround_Truth_SQL\tBest_Candidate_SQL\tPredicted_Result_Preview\tGround_Truth_Result_Preview\n")
        
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Evaluating results"):
            db_id = row['db_id']
            results = execution_results_map.get(index)
            gt_sql = extract_ground_truth_sql(row)

            if not results or not results['ground_truth']:
                f_out.write(f"{index}\t{db_id}\tSKIPPED_NO_GT\t{repr(gt_sql)}\tNone\tN/A\tN/A\n")
                continue

            gt_result, gt_success = results['ground_truth']
            
            total_evaluated += 1
            
            # --- NEW LOGIC: Condition 1 (GT Error counts as Correct) ---
            if gt_success != 1:
                correct_predictions += 1
                f_out.write(
                    f"{index}\t{db_id}\tMATCH_GT_ERROR\t"
                    f"{repr(gt_sql)}\tN/A\t"
                    f"GT_FAILED\t"
                    f"{format_result_for_logging(gt_result)}\n"
                )
                continue
            
            # Standard Evaluation Logic
            match_found = False
            best_candidate_sql = None
            log_output_result = "No successful execution"
            
            candidate_sqls = extract_candidate_sqls(row, n=16)
            first_error_found = None

            for i, cand_res_tuple in enumerate(results['candidates']):
                if not cand_res_tuple: continue
                
                pred_result, pred_success = cand_res_tuple
                current_sql = candidate_sqls[i]
                if not current_sql: continue

                if pred_success == 1:
                    if log_output_result == "No successful execution":
                        log_output_result = pred_result
                        best_candidate_sql = current_sql

                    if advanced_result_compare(pred_result, gt_result):
                        match_found = True
                        best_candidate_sql = current_sql
                        log_output_result = pred_result
                        break
                elif first_error_found is None:
                    first_error_found = pred_result

            if match_found:
                status = "MATCH"
                correct_predictions += 1
            else:
                status = "MISMATCH"
                if log_output_result == "No successful execution":
                    log_output_result = first_error_found if first_error_found else "All candidates were empty or failed"
            
            f_out.write(
                f"{index}\t{db_id}\t{status}\t"
                f"{repr(gt_sql)}\t{repr(best_candidate_sql)}\t"
                f"{format_result_for_logging(log_output_result)}\t"
                f"{format_result_for_logging(gt_result)}\n"
            )

    print("\n✅ Evaluation complete.")
    experiment_name = os.path.basename(args.input_file).replace(".parquet", "")

    if total_evaluated > 0:
        accuracy = (correct_predictions / total_evaluated) * 100
        summary = (
            f"Experiment: {experiment_name}\n"
            f"Correct Predictions: {correct_predictions}\n"
            f"Total Queries Evaluated: {total_evaluated}\n"
            f"Execution Accuracy (Best-of-16, Lenient): {accuracy:.2f}%\n\n"
        )
    else:
        summary = f"Experiment: {experiment_name}\nNo queries were successfully evaluated.\n\n"

    print("\n--- Evaluation Summary ---\n" + summary)
    
    if args.score_file:
        with open(args.score_file, 'a', encoding='utf-8') as f_score:
            f_score.write(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SQL queries with Lenient Metrics & Parallel Execution.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input Parquet file.")
    parser.add_argument("--db_path", type=str, required=True, help="Base directory path for databases.")
    parser.add_argument("--output_file", type=str, default="evaluation_log.tsv", help="Path for the output log file.")
    parser.add_argument("--score_file", type=str, default="result_score.txt", help="Path for the evaluation summary file.")
    parser.add_argument("--num_cpus", type=int, default=48, help="Number of CPU cores for parallel SQL execution.")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds for each SQL query execution.")
    
    args = parser.parse_args()
    if func_timeout is None:
        sys.exit(1)
    main(args)