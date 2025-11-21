# import argparse
# import pandas as pd
# import re
# import os
# from tqdm import tqdm
# from typing import List, Dict, Optional, Tuple
# import sqlite3
# import concurrent.futures
# import numpy as np
# from collections import Counter

# # --- 列名配置 ---
# # 注意：不再需要 PROMPT_COLUMN，但为了提取SQL，保留其他列
# RESULT_COLUMN = 'result'
# FULL_RESPONSES_COLUMN = 'full_responses'
# DB_ID_COLUMN = 'db_id'

# # --- 数据提取与数据库执行函数 (这些函数保持不变) ---

# def extract_candidate_sqls(row: pd.Series, n: int = 8) -> List[Optional[str]]:
#     """提取候选SQL。"""
#     candidates = []
#     results = row.get(RESULT_COLUMN, [])
#     fulls = row.get(FULL_RESPONSES_COLUMN, [])
#     if not isinstance(results, (list, pd.Series, np.ndarray)): results = []
#     if not isinstance(fulls, (list, pd.Series, np.ndarray)): fulls = []
#     for i in range(n):
#         sql = None
#         if i < len(results) and isinstance(results[i], str) and results[i].strip().upper().startswith("SELECT"):
#             sql = results[i].strip().strip('"')
#         elif i < len(fulls) and isinstance(fulls[i], str):
#             blocks = re.findall(r"<solution>(.*?)</solution>|<sql>(.*?)</sql>", fulls[i], re.DOTALL)
#             if blocks:
#                 last_block_content = next((s for s in blocks[-1] if s), None)
#                 if last_block_content:
#                     sql = last_block_content.strip()
#         candidates.append(sql)
#     return candidates

# def execute_sql_safe(sql: str, db_path: str) -> Tuple[Optional[List], Optional[str]]:
#     """在一个安全的环境中执行SQL查询，使用1000行样本。"""
#     if not sql:
#         return None, "[Execution Error: SQL query is empty]"
#     if not os.path.exists(db_path):
#         return None, f"[Execution Error: Database file not found at {db_path}]"
#     try:
#         con = sqlite3.connect(":memory:")
#         cursor = con.cursor()
#         cursor.execute(f"ATTACH DATABASE '{db_path}' AS orig")
#         cursor.execute("SELECT name FROM orig.sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
#         tables = [row[0] for row in cursor.fetchall()]
#         for table in tables:
#             cursor.execute(f"CREATE TABLE `{table}` AS SELECT * FROM orig.`{table}` LIMIT 1000")
#         cursor.execute("PRAGMA case_sensitive_like = true;")
#         if "LIKE" in sql.upper():
#             sql = sql.replace("LIKE", "GLOB").replace("%", "*")
#         cursor.execute(sql)
#         result = cursor.fetchall()
#         con.close()
#         return result, None
#     except Exception as e:
#         if 'con' in locals() and con:
#             con.close()
#         return None, f"[Execution Error: {e}]"

# def execute_sql_with_timeout(sql: str, db_path: str, timeout_sec: int = 15) -> Tuple[Optional[List], Optional[str]]:
#     """使用超时机制执行SQL查询。"""
#     with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
#         future = executor.submit(execute_sql_safe, sql, db_path)
#         try:
#             return future.result(timeout=timeout_sec)
#         except concurrent.futures.TimeoutError:
#             return None, "[Execution Timeout]"
#         except Exception as e:
#             return None, f"[Future Execution Error: {e}]"

# # --- 主函数 (Self-Consistency版本) ---

# def main(args):
#     """主执行流程 (Self-Consistency版本)"""
#     print(f"--> 正在加载输入Parquet文件: {args.input_file}")
#     df = pd.read_parquet(args.input_file)
#     print(f"    文件加载成功，共找到 {len(df)} 行数据。")

#     # 用于存储最终结果的列表
#     selected_indices = []
#     selected_sqls = []
#     consistency_counts = []
#     original_indices = []

#     print("--> 正在为每个问题执行所有候选SQL并进行自洽性检查...")
#     for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="执行SQL并选择"):
#         db_id = row.get(DB_ID_COLUMN)
#         db_path = os.path.join(args.db_path, db_id, f"{db_id}.sqlite")

#         if not db_id or not os.path.exists(db_path):
#             continue
        
#         candidate_sqls = extract_candidate_sqls(row, n=8)
        
#         execution_results = []
#         valid_candidate_indices = []

#         # 1. 执行所有有效的候选SQL
#         for i, sql in enumerate(candidate_sqls):
#             if sql:
#                 result, error = execute_sql_with_timeout(sql, db_path, timeout_sec=20)
#                 execution_results.append((result, error))
#                 valid_candidate_indices.append(i)

#         if not execution_results:
#             # 如果没有任何有效的SQL可以执行
#             original_indices.append(index)
#             selected_indices.append(-1)
#             selected_sqls.append(None)
#             consistency_counts.append(0)
#             continue
        
#         # 2. 对结果进行计数以找到最一致的答案
#         # 为了能对结果进行计数，需要将它们转换为可哈希的格式
#         # 成功的结果转为元组的元组，错误或None转为特定的字符串
#         hashable_results = []
#         for res, err in execution_results:
#             if err:
#                 hashable_results.append(f"ERROR: {err}") # 将错误视为一种独特的结果
#             elif res is None:
#                 hashable_results.append("RESULT: None")
#             else:
#                 # 将 [[1, 'a'], [2, 'b']] 转换为 ((1, 'a'), (2, 'b'))
#                 hashable_results.append(tuple(map(tuple, res)))
        
#         if not hashable_results:
#             # 再次检查，以防万一
#             original_indices.append(index)
#             selected_indices.append(-1)
#             selected_sqls.append(None)
#             consistency_counts.append(0)
#             continue

#         # 3. 统计结果频次并选择
#         result_counts = Counter(hashable_results)
#         most_common_result, max_count = result_counts.most_common(1)[0]
        
#         # 找到第一个产生这个最常见结果的SQL的索引
#         first_occurrence_index = hashable_results.index(most_common_result)
#         best_candidate_original_idx = valid_candidate_indices[first_occurrence_index]
#         best_sql = candidate_sqls[best_candidate_original_idx]

#         # 4. 保存结果
#         original_indices.append(index)
#         selected_indices.append(best_candidate_original_idx + 1) # 索引从1开始
#         selected_sqls.append(best_sql)
#         consistency_counts.append(max_count)

#     # 将结果合并回原始DataFrame
#     if original_indices:
#         results_df = pd.DataFrame({
#             'df_index': original_indices,
#             'sc_selected_index': selected_indices,
#             'llm_selected_sql': selected_sqls,
#             'sc_consistency_count': consistency_counts
#         }).set_index('df_index')
#         df = df.join(results_df)
#         df['sc_selected_index'] = df['sc_selected_index'].fillna(-1).astype(int)
#         df['sc_consistency_count'] = df['sc_consistency_count'].fillna(0).astype(int)
#     else:
#         # 如果没有处理任何行
#         df['sc_selected_index'] = -1
#         df['sc_selected_sql'] = None
#         df['sc_consistency_count'] = 0

#     print(f"--> 正在保存结果到: {args.output_file}")
#     df.to_parquet(args.output_file)
#     print("✅ 流程成功完成 (Self-Consistency)。")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="使用Self-Consistency方法，通过执行SQL并选择结果最一致的方案。")
#     parser.add_argument("--input_file", type=str, required=True, help="输入的Parquet文件路径。")
#     parser.add_argument("--db_path", type=str, required=True, help="存放所有数据库文件的根目录路径。")
#     parser.add_argument("--output_file", type=str, required=True, help="输出的Parquet文件路径。")
#     # 不再需要模型相关的参数
#     # parser.add_argument("--model_name", type=str, required=True, help="用于VLLM的Hugging Face模型名称或路径。")
#     # parser.add_argument("--tensor-parallel-size", type=int, default=1, help="VLLM使用的GPU数量。")
#     args = parser.parse_args()
#     main(args)

#并行执行完整的表格
import argparse
import pandas as pd
import re
import os
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import sqlite3
import concurrent.futures
import numpy as np
from collections import Counter
from collections import defaultdict

# --- Parallel Execution Imports ---
from func_timeout import func_timeout, FunctionTimedOut
import multiprocessing as mp
import sys
from time import perf_counter
import signal


# --- Column Configuration ---
RESULT_COLUMN = 'result'
FULL_RESPONSES_COLUMN = 'full_responses'
DB_ID_COLUMN = 'db_id'

# --- Data Extraction Function (Unchanged) ---
def extract_candidate_sqls(row: pd.Series, n: int = 8) -> List[Optional[str]]:
    """Extracts candidate SQL queries from a DataFrame row."""
    candidates = []
    results = row.get(RESULT_COLUMN, [])
    fulls = row.get(FULL_RESPONSES_COLUMN, [])
    if not isinstance(results, (list, pd.Series, np.ndarray)): results = []
    if not isinstance(fulls, (list, pd.Series, np.ndarray)): fulls = []
    for i in range(n):
        sql = None
        if i < len(results) and isinstance(results[i], str) and results[i].strip().upper().startswith("SELECT"):
            sql = results[i].strip().strip('"')
        elif i < len(fulls) and isinstance(fulls[i], str):
            blocks = re.findall(r"<solution>(.*?)</solution>|<sql>(.*?)</sql>", fulls[i], re.DOTALL)
            if blocks:
                last_block_content = next((s for s in blocks[-1] if s), None)
                if last_block_content:
                    sql = last_block_content.strip()
        candidates.append(sql)
    return candidates

# --- New Parallel SQL Execution Functions (Adapted from your example) ---

def execute_sql(task_idx: int, db_path: str, sql: str) -> Tuple:
    """
    Executes a single SQL query against a full SQLite database file.
    This function is run by a worker process.
    """
    if not os.path.exists(db_path):
        return task_idx, db_path, sql, f"[Execution Error: Database file not found at {db_path}]", 0

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        
        # --- Logic ported from original execute_sql_safe ---
        cursor.execute("PRAGMA case_sensitive_like = true;")
        if "LIKE" in sql.upper():
            sql_to_exec = sql.replace("LIKE", "GLOB").replace("%", "*")
        else:
            sql_to_exec = sql
        # --- End of ported logic ---
            
        cursor.execute(sql_to_exec)
        execution_res = frozenset(cursor.fetchall())
        conn.close()
        return task_idx, db_path, sql, execution_res, 1
    except Exception as e:
        error_msg = f"[Execution Error: {type(e).__name__}: {str(e)}]"
        if 'conn' in locals() and conn:
            conn.close()
        return task_idx, db_path, sql, error_msg, 0

def execute_sql_wrapper(task_idx: int, db_path: str, sql: str, timeout: int) -> Tuple:
    """
    A wrapper to apply a timeout to the SQL execution function.
    """
    try:
        res = func_timeout(timeout, execute_sql, args=(task_idx, db_path, sql))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        res = (task_idx, db_path, sql, "[Execution Timeout]", 0)
    except Exception as e:
        error_msg = f"[Future Execution Error: {type(e).__name__}: {str(e)}]"
        res = (task_idx, db_path, sql, error_msg, 0)
    return res

def execute_sqls_parallel(db_paths: List[str], pred_sqls: List[str], num_cpus: int, timeout: int) -> List[Tuple]:
    """
    Executes a list of SQL statements in parallel on the full databases.
    """
    args = [
        (idx, db_path, sql, timeout)
        for idx, (db_path, sql) in enumerate(zip(db_paths, pred_sqls))
    ]
    
    print(f"--> Executing {len(args)} SQL statements in parallel using {num_cpus} processes...")
    start = perf_counter()
    
    original_sigchld_handler = signal.getsignal(signal.SIGCHLD)
    try:
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        mp.set_start_method("spawn", force=True)
        with mp.Pool(processes=num_cpus) as pool:
            # THIS IS THE LINE TO CHANGE: imap -> starmap
            # The tqdm wrapper will now show progress as chunks of tasks complete, 
            # rather than one by one, but it will still work.
            results = pool.starmap(execute_sql_wrapper, args)
            # To get a more responsive progress bar with starmap, you can wrap the input
            # but for simplicity, just changing the function name is the most direct fix.
            # For example: results = list(tqdm(pool.starmap(execute_sql_wrapper, args), total=len(args)))
    finally:
        signal.signal(signal.SIGCHLD, original_sigchld_handler)
        
    end = perf_counter()
    print(f"    Finished execution in {end - start:.2f} seconds.")
    # We don't need to wrap in tqdm or list() anymore as starmap returns a list directly
    return results
# --- Main Logic (Refactored for Parallelism) ---

def main(args):
    """Main execution flow using parallel processing for self-consistency."""
    print(f"--> Loading input Parquet file: {args.input_file}")
    df = pd.read_parquet(args.input_file)
    print(f"    File loaded successfully with {len(df)} rows.")

    # Step 1: Gather all SQL execution tasks from the DataFrame
    print("--> Preparing all candidate SQL queries for execution...")
    tasks_for_parallel_exec = []
    # This list maps the flat task index back to its original row and candidate index
    task_info_map = [] 

    for df_index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Gathering Tasks"):
        db_id = row.get(DB_ID_COLUMN)
        if not db_id:
            continue
        
        db_path = os.path.join(args.db_path, db_id, f"{db_id}.sqlite")
        candidate_sqls = extract_candidate_sqls(row, n=8)
        
        for candidate_idx, sql in enumerate(candidate_sqls):
            if sql and os.path.exists(db_path):
                # We store the original DataFrame index and the candidate's index (0-7)
                task_info_map.append({'df_index': df_index, 'candidate_idx': candidate_idx, 'sql': sql})
                tasks_for_parallel_exec.append((db_path, sql))

    if not tasks_for_parallel_exec:
        print("No valid SQL queries found to execute. Exiting.")
        # Add empty columns and save
        df['sc_selected_index'] = -1
        df['llm_selected_sql'] = None
        df['sc_consistency_count'] = 0
        df.to_parquet(args.output_file)
        return

    # Unzip tasks for the parallel function
    db_paths, pred_sqls = zip(*tasks_for_parallel_exec)

    # Step 2: Execute all tasks in parallel
    # The result format is: (task_idx, db_path, sql, execution_res, success_flag)
    parallel_results = execute_sqls_parallel(db_paths, pred_sqls, num_cpus=args.num_cpus, timeout=args.timeout)

    # Step 3: Group results by original DataFrame index and apply self-consistency
    print("--> Grouping results and applying self-consistency logic...")
    results_by_df_index = defaultdict(list)
    for i, presult in enumerate(parallel_results):
        task_idx, _, _, execution_res, success_flag = presult
        
        # Get the original DataFrame index and candidate info
        original_info = task_info_map[task_idx]
        df_index = original_info['df_index']

        # Store the result (already hashable frozenset or string error) and original candidate info
        results_by_df_index[df_index].append({
            'result': execution_res if success_flag == 1 else str(execution_res),
            'candidate_idx': original_info['candidate_idx'],
            'sql': original_info['sql']
        })

    # Step 4: Iterate through grouped results to find the most consistent one for each row
    final_selections = []
    for df_index in tqdm(df.index, desc="Selecting Best SQL"):
        row_results = results_by_df_index.get(df_index)

        if not row_results:
            final_selections.append({
                'df_index': df_index,
                'sc_selected_index': -1,
                'llm_selected_sql': None,
                'sc_consistency_count': 0
            })
            continue

        # Use Counter to find the most frequent execution result
        result_counts = Counter(r['result'] for r in row_results)
        most_common_result, max_count = result_counts.most_common(1)[0]
        
        # Find the first candidate SQL that produced this most common result
        # Sort by candidate_idx to ensure we pick the earliest one in case of a tie
        best_candidate = sorted(
            [r for r in row_results if r['result'] == most_common_result], 
            key=lambda x: x['candidate_idx']
        )[0]
        
        final_selections.append({
            'df_index': df_index,
            'sc_selected_index': best_candidate['candidate_idx'] + 1, # Index from 1
            'llm_selected_sql': best_candidate['sql'],
            'sc_consistency_count': max_count
        })

    # Step 5: Merge results back into the original DataFrame
    print("--> Merging results and saving output...")
    results_df = pd.DataFrame(final_selections).set_index('df_index')
    df = df.join(results_df)

    # Fill in NaNs for rows that had no valid SQLs to begin with
    df['sc_selected_index'] = df['sc_selected_index'].fillna(-1).astype(int)
    df['sc_consistency_count'] = df['sc_consistency_count'].fillna(0).astype(int)

    df.to_parquet(args.output_file, index=True)
    print(f"✅ Process successfully completed. Output saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use Self-Consistency by executing all candidate SQLs in parallel and choosing the most frequent result.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input Parquet file.")
    parser.add_argument("--db_path", type=str, required=True, help="Root directory path containing all database files.")
    parser.add_argument("--output_file", type=str, required=True, help="Path for the output Parquet file.")
    parser.add_argument("--num_cpus", type=int, default=48, help="Number of worker processes for parallel SQL execution.")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds for each individual SQL query execution.")
    
    args = parser.parse_args()
    main(args)