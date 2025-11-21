#final loop+execution result+external knowledge
import argparse
import pandas as pd
import re
import os
import sys
import signal
import multiprocessing as mp
from time import perf_counter
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import sqlite3
import numpy as np

# --- Dependencies for parallel execution and VLLM ---
try:
    from func_timeout import func_timeout, FunctionTimedOut
except ImportError:
    print("警告: 'func_timeout'库未安装。请运行 'pip install func-timeout'。")
    func_timeout, FunctionTimedOut = None, None

try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
except ImportError:
    print("警告: VLLM或Hugging Face Transformers库未安装。此脚本的VLLM部分将无法运行。")
    LLM, SamplingParams, AutoTokenizer = None, None, None

# --- Column Configuration ---
PROMPT_COLUMN = 'prompt'
RESULT_COLUMN = 'result'
FULL_RESPONSES_COLUMN = 'full_responses'
DB_ID_COLUMN = 'db_id'

GENRM_PROMPT_TEMPLATE = """
**Task Background:**
You are an expert SQL data analyst. Your task is to verify if a proposed solution correctly answers a user's question.

**Problem:**
{question}

**External Knowledge:**
{external_knowledge}

**Proposed Solution:**
{solution_text}

---
**Your Task:**
Based on all the information, is the SQL query in the solution logically correct for answering the question?
You must answer with "Yes" or "No" first, before any other text.

Is the answer correct (Yes/No)?
"""

# --- Data Extraction Functions ---

def extract_question_from_prompt(prompt_data) -> Optional[str]:
    """Parses the user's question from the 'prompt' column."""
    try:
        for message in prompt_data:
            if isinstance(message, dict) and message.get('role') == 'user':
                content = message.get('content', '')
                match = re.search(r"Question:\s*(.*)", content, re.DOTALL)
                if match:
                    return match.group(1).strip()
    except TypeError: return None
    return None

def extract_knowledge_from_prompt(prompt_data) -> str:
    """
    Parses the 'External Knowledge' from the 'prompt' column.
    Returns 'N/A' if not found.
    """
    try:
        for message in prompt_data:
            if isinstance(message, dict) and message.get('role') == 'user':
                content = message.get('content', '')
                # Regex to find text between 'External Knowledge:' and 'Question:'
                match = re.search(r"External Knowledge:\s*(.*?)\s*\n\nQuestion:", content, re.DOTALL)
                if match:
                    return match.group(1).strip()
    except TypeError:
        return "N/A"
    return "N/A"


def extract_candidate_sqls(row: pd.Series, n: int = 16) -> List[Optional[str]]:
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
                if last_block_content: sql = last_block_content.strip()
        candidates.append(sql)
    return candidates

# --- Trajectory Simplification and SQL Execution (Unchanged) ---
def extract_last_two_thinks(trajectory: Optional[str]) -> Optional[str]:
    if not isinstance(trajectory, str): return trajectory
    tag = "<think>"
    last_pos = trajectory.rfind(tag)
    if last_pos == -1: return trajectory
    second_last_pos = trajectory.rfind(tag, 0, last_pos)
    if second_last_pos != -1: return trajectory[second_last_pos:]
    else: return trajectory[last_pos:]

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
        raise ImportError("请运行 'pip install func-timeout' 来使用超时功能。")
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
    args = [(idx, db_file, sql, timeout) for idx, (db_file, sql) in enumerate(zip(db_files, pred_sqls))]
    print(f"--> [并行执行]: 准备执行 {len(args)} 条SQL语句...")
    start = perf_counter()
    original_sigchld_handler = signal.getsignal(signal.SIGCHLD)
    try:
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        mp.set_start_method("spawn", force=True)
        with mp.Pool(processes=num_cpus) as pool:
            results = list(tqdm(pool.starmap(execute_sql_wrapper, args), total=len(args), desc="并行执行SQL"))
    finally:
        signal.signal(signal.SIGCHLD, original_sigchld_handler)
    end = perf_counter()
    print(f"--> [并行执行]: 在 {end - start:.2f} 秒内完成。")
    return results

def format_solution_text(reasoning: str, sql: str, exec_result: Optional[List], exec_error: Optional[str]) -> str:
    MAX_RESULTS = 10
    observation_text = ""
    if exec_error:
        observation_text = f"[Execution Error: {exec_error}]"
    elif exec_result is None:
        observation_text = "[Execution resulted in None]"
    elif not exec_result:
        observation_text = "[Query returned NO RESULTS]"
    else:
        result_preview = exec_result[:MAX_RESULTS]
        observation_text = "Result preview (first {} rows):\n".format(len(result_preview)) + str(result_preview)
        if len(exec_result) > MAX_RESULTS:
            observation_text += f"\n...({len(exec_result) - MAX_RESULTS} more rows not shown)"
    solution = f"""Reasoning from Generator:
{reasoning if reasoning else "[No reasoning available]"}
Final Extracted SQL:
{sql if sql else "[No SQL extracted]"}
Execution Observation:
{observation_text}
"""
    return solution

def get_yes_no_token_ids(tokenizer) -> Tuple[int, int]:
    yes_tokens = ["Yes", " Yes", "yes", " yes"]; no_tokens = ["No", " No", "no", " no"]
    yes_id = next((tokenizer.convert_tokens_to_ids(t) for t in yes_tokens if tokenizer.convert_tokens_to_ids(t) != tokenizer.unk_token_id), -1)
    no_id = next((tokenizer.convert_tokens_to_ids(t) for t in no_tokens if tokenizer.convert_tokens_to_ids(t) != tokenizer.unk_token_id), -1)
    if yes_id == -1 or no_id == -1: raise ValueError("无法找到'Yes'和'No'的明确token ID。")
    print(f"Found Token ID for 'Yes': {yes_id}, for 'No': {no_id}")
    return yes_id, no_id

# --- Main Function ---
def main(args):
    """主执行流程"""
    if LLM is None:
        print("错误: VLLM或Transformers库是运行此脚本所必需的，请先安装。")
        return

    print(f"--> 正在加载输入Parquet文件: {args.input_file}")
    df = pd.read_parquet(args.input_file)
    print(f"    文件加载成功，共找到 {len(df)} 行数据。")

    # --- SQL Collection and Execution (Unchanged) ---
    print("--> 步骤 1: 正在从数据集中收集所有SQL查询...")
    db_files_to_run, sqls_to_run, job_metadata = [], [], []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="收集中..."):
        db_id = row.get(DB_ID_COLUMN)
        if not db_id: continue
        db_path = os.path.join(args.db_path, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path): continue
        final_sqls = extract_candidate_sqls(row, n=16)
        for i, sql in enumerate(final_sqls):
            db_files_to_run.append(db_path)
            sqls_to_run.append(sql)
            job_metadata.append({'df_index': index, 'candidate_idx': i})
    
    parallel_results = execute_sqls_parallel(db_files_to_run, sqls_to_run, num_cpus=args.num_cpus, timeout=args.timeout)

    print("--> 步骤 2: 正在将SQL执行结果重新组织...")
    execution_results_map = {}
    for i, result_tuple in enumerate(parallel_results):
        meta = job_metadata[i]
        df_index, candidate_idx = meta['df_index'], meta['candidate_idx']
        _, _, _, execution_res, success_flag = result_tuple
        exec_result, exec_error = (list(execution_res), None) if success_flag == 1 else (None, str(execution_res))
        execution_results_map[(df_index, candidate_idx)] = (exec_result, exec_error)

    # --- Prompt Building and VLLM Scoring ---
    print("--> 步骤 3: 正在为每个候选方案准备Prompt...")
    all_prompts, problem_metadata = [], []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="构建Prompts"):
        
        # --- <<< CHANGE 3: Extract question AND knowledge >>> ---
        prompt_content = row.get(PROMPT_COLUMN)
        question = extract_question_from_prompt(prompt_content)
        external_knowledge = extract_knowledge_from_prompt(prompt_content)

        if not question: continue

        final_sqls = extract_candidate_sqls(row, n=16)
        full_responses = row.get(FULL_RESPONSES_COLUMN, [])
        if not isinstance(full_responses, (list, pd.Series, np.ndarray)): full_responses = []

        num_valid_candidates = 0
        for i in range(16):
            sql = final_sqls[i] if i < len(final_sqls) else None
            reasoning = full_responses[i] if i < len(full_responses) else None
            
            if sql or (reasoning and isinstance(reasoning, str)):
                exec_result, exec_error = execution_results_map.get((index, i), (None, "Result not found"))
                simplified_reasoning = extract_last_two_thinks(reasoning)
                solution_text = format_solution_text(simplified_reasoning, sql, exec_result, exec_error)
                
                # --- <<< CHANGE 4: Format prompt with all three components >>> ---
                prompt = GENRM_PROMPT_TEMPLATE.format(
                    question=question,
                    external_knowledge=external_knowledge,
                    solution_text=solution_text
                )
                all_prompts.append(prompt)
                num_valid_candidates += 1

        if num_valid_candidates > 0:
            problem_metadata.append({'df_index': index, 'num_candidates': num_valid_candidates, 'final_sqls': final_sqls})
    
    # --- VLLM Scoring and Saving (Unchanged) ---
    print(f"--> 步骤 4: 共生成 {len(all_prompts)} 个prompt。正在使用VLLM进行打分...")
    llm = LLM(model=args.model_name, tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    yes_token_id, no_token_id = get_yes_no_token_ids(tokenizer)
    
    sampling_params = SamplingParams(n=args.num_samples, temperature=0.2, max_tokens=5, logprobs=10)

    if not all_prompts:
        df['llm_selected_index'] = -1
        df['llm_selected_sql'] = None
        df['llm_selection_score'] = 0.0
    else:
        outputs = llm.generate(all_prompts, sampling_params)
        print("--> 推理完成。正在计算平均分数并选择最佳方案...")
        
        scores = [np.mean([probs.get(yes_token_id, 0.0) for sample in o.outputs if sample.logprobs and (probs := {tid: np.exp(lp.logprob) for tid, lp in sample.logprobs[0].items()})]) if o.outputs else 0.0 for o in outputs]

        selected_indices, selected_sqls, selected_scores = [], [], []
        current_score_index = 0
        for meta in tqdm(problem_metadata, desc="选择最佳方案"):
            num_candidates = meta['num_candidates']
            candidate_scores = scores[current_score_index : current_score_index + num_candidates]
            
            if not candidate_scores: best_candidate_local_idx, max_score = -1, 0.0
            else: best_candidate_local_idx, max_score = np.argmax(candidate_scores), candidate_scores[np.argmax(candidate_scores)]

            selected_indices.append(best_candidate_local_idx + 1 if best_candidate_local_idx != -1 else -1)
            selected_scores.append(max_score)
            final_sql = meta['final_sqls'][best_candidate_local_idx] if best_candidate_local_idx != -1 else None
            selected_sqls.append(final_sql)
            current_score_index += num_candidates

        results_df = pd.DataFrame({'df_index': [d['df_index'] for d in problem_metadata], 'llm_selected_index': selected_indices, 'llm_selected_sql': selected_sqls, 'llm_selection_score': selected_scores}).set_index('df_index')
        df = df.join(results_df)
        df['llm_selected_index'] = df['llm_selected_index'].fillna(-1).astype(int)
        df['llm_selection_score'] = df['llm_selection_score'].fillna(0.0)

    print(f"--> 正在保存结果到: {args.output_file}")
    df.to_parquet(args.output_file)
    print("✅ 流程成功完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用GenRM方法，通过并行执行完整SQL并对多次采样的'Yes'概率取平均来选择最佳方案。")
    parser.add_argument("--input_file", type=str, required=True, help="输入的Parquet文件路径。")
    parser.add_argument("--db_path", type=str, required=True, help="存放所有数据库文件的根目录路径。")
    parser.add_argument("--output_file", type=str, required=True, help="输出的Parquet文件路径。")
    parser.add_argument("--model_name", type=str, required=True, help="用于VLLM的Hugging Face模型名称或路径。")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="VLLM使用的GPU数量。")
    parser.add_argument("--num_samples", type=int, default=8, help="为每个候选方案生成的判断次数，用于计算平均分。")
    parser.add_argument("--num_cpus", type=int, default=48, help="用于并行执行SQL的CPU核心数。")
    parser.add_argument("--timeout", type=int, default=120, help="每个SQL查询的执行超时时间（秒）。")
    
    args = parser.parse_args()
    if func_timeout is None:
        sys.exit(1)
    main(args)