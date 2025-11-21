# select_with_execution_api.py
import argparse
import pandas as pd
import re
import os
from tqdm import tqdm
from typing import List, Optional, Tuple
import sqlite3
import concurrent.futures
import time
import multiprocessing as mp
import signal
from time import perf_counter

# --- 依赖: OpenAI API ---
try:
    from openai import OpenAI, APIError
except ImportError:
    print("错误: openai库未安装。请运行 'pip install openai' 进行安装。")
    OpenAI, APIError = None, None

# --- 依赖: 并行执行 ---
try:
    from func_timeout import func_timeout, FunctionTimedOut
except ImportError:
    print("警告: 'func_timeout'库未安装。请运行 'pip install func-timeout' 来使用超时功能。")
    func_timeout, FunctionTimedOut = None, None


# --- 列名配置 ---
PROMPT_COLUMN = 'prompt'
RESULT_COLUMN = 'result'
FULL_RESPONSES_COLUMN = 'full_responses'
DB_ID_COLUMN = 'db_id'

# --- Prompt模板 (保持不变) ---
SELECTION_PROMPT_TEMPLATE = """You are an expert SQL data analyst. Your task is to select the BEST SQL query that correctly answers a user's question.

You are given several candidates. For each candidate, you will see its reasoning, the SQL query itself, and importantly, **the result of executing that query on the database.** A query might look correct but return an error or empty/wrong data. You must use the execution observation to make your final decision.

Here is the user's question:
"{question}"

Evaluate the following candidates based on ALL available information. Does the "Execution Observation" for a candidate actually answer the user's question?
---
{formatted_candidates}
---
**Final Analysis:** Considering the reasoning, the SQL code, and especially the **execution results**, which single candidate provides the most correct and complete answer to the user's question?

**Instructions for your response:**
- Respond with ONLY the index number of the single best candidate.
- If multiple candidates produce correct results, select the one with the LOWEST index number.
- Do not include any other words, symbols, or explanations.

Best candidate index:
"""

# --- 数据提取与数据库执行函数 ---

def extract_question_from_prompt(prompt_data) -> Optional[str]:
    """从'prompt'列中解析出用户的问题。"""
    try:
        for message in prompt_data:
            if isinstance(message, dict) and message.get('role') == 'user':
                content = message.get('content', '')
                match = re.search(r"Question:\s*(.*)", content, re.DOTALL)
                if match:
                    return match.group(1).strip()
    except TypeError:
        return None
    return None

def extract_candidate_sqls(row: pd.Series, n: int = 8) -> List[Optional[str]]:
    """提取候选SQL。"""
    candidates = []
    results = row.get(RESULT_COLUMN, [])
    fulls = row.get(FULL_RESPONSES_COLUMN, [])
    if not isinstance(results, (list, pd.Series, type(pd.Series([1]).values))): results = []
    if not isinstance(fulls, (list, pd.Series, type(pd.Series([1]).values))): fulls = []
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

def simplify_reasoning_text(reasoning: Optional[str]) -> str:
    """从完整的推理轨迹中提取最后一部分。"""
    if not isinstance(reasoning, str):
        return "[No reasoning available]"
    tag = "<think>"
    last_pos = reasoning.rfind(tag)
    if last_pos == -1: return reasoning
    second_last_pos = reasoning.rfind(tag, 0, last_pos)
    if second_last_pos != -1: return reasoning[second_last_pos:]
    else: return reasoning[last_pos:]

# --- <<< 修改点: 更新了SQL执行和结果格式化逻辑 >>> ---
def execute_sql(data_idx, db_file, sql):
    """在完整的数据库文件上执行单个SQL查询。"""
    if not sql:
        return data_idx, db_file, sql, "SQL query is empty", 0
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
    """为SQL执行提供超时封装。"""
    if func_timeout is None:
        raise ImportError("请运行 'pip install func-timeout' 来使用超时功能。")
    try:
        res = func_timeout(timeout, execute_sql, args=(data_idx, db_file, sql))
    except KeyboardInterrupt:
        import sys
        sys.exit(0)
    except FunctionTimedOut:
        res = (data_idx, db_file, sql, "Function TimeOut", 0)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        res = (data_idx, db_file, sql, error_msg, 0)
    return res

def execute_sqls_parallel(db_files, pred_sqls, num_cpus=64, timeout=50):
    """并行执行SQL查询列表。"""
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

def format_execution_result(result: Optional[List], error: Optional[str]) -> str:
    """格式化SQL执行结果以便在Prompt中展示，最多显示10行。"""
    MAX_ROWS = 10
    if error: return f"[Execution Error: {error}]"
    if result is None: return "[Execution resulted in None]"
    if not result: return "[Query returned NO RESULTS]"
    
    result_preview = result[:MAX_ROWS]
    formatted = f"Result preview (first {len(result_preview)} rows):\n{str(result_preview)}"
    if len(result) > MAX_ROWS:
        formatted += f"\n...({len(result) - MAX_ROWS} more rows not shown)"
    return formatted

# --- API调用函数 (保持不变) ---
def call_llm_api_with_retry(client: OpenAI, model: str, prompt: str, max_retries: int = 3) -> Optional[str]:
    """调用OpenAI兼容的API，并带有错误处理和重试逻辑。"""
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "You are an expert SQL data analyst."}, {"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=50
            )
            return completion.choices[0].message.content
        except APIError as e:
            print(f"API调用错误: {e}。正在进行第 {attempt + 1}/{max_retries} 次重试...")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"发生未知错误: {e}。正在进行第 {attempt + 1}/{max_retries} 次重试...")
            time.sleep(2 ** attempt)
    print(f"API调用失败 {max_retries} 次，放弃该prompt。")
    return None

# --- 主函数 ---
def main(args):
    """主执行流程"""
    if OpenAI is None or func_timeout is None:
        return

    print("--> 正在初始化API客户端...")
    try:
        client = OpenAI(base_url=args.api_base_url, api_key=args.api_key)
    except Exception as e:
        print(f"初始化API客户端失败: {e}"); return

    print(f"--> 正在加载输入Parquet文件: {args.input_file}")
    df = pd.read_parquet(args.input_file)
    print(f"    文件加载成功，共找到 {len(df)} 行数据。")

    # --- <<< 修改点：阶段1 - 收集所有待执行的SQL任务 >>> ---
    print("--> 阶段 1: 正在从数据集中收集所有SQL查询...")
    db_files_to_run, sqls_to_run, job_metadata = [], [], []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="收集中"):
        db_id = row.get(DB_ID_COLUMN)
        if not db_id: continue
        db_path = os.path.join(args.db_path, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path): continue
        final_sqls = extract_candidate_sqls(row, n=8)
        for i, sql in enumerate(final_sqls):
            db_files_to_run.append(db_path)
            sqls_to_run.append(sql)
            job_metadata.append({'df_index': index, 'candidate_idx': i})
    
    # --- <<< 修改点：阶段2 - 并行执行所有SQL >>> ---
    parallel_results = execute_sqls_parallel(db_files_to_run, sqls_to_run, num_cpus=args.num_cpus, timeout=args.sql_timeout)

    # --- <<< 修改点：阶段3 - 将执行结果映射回原数据 >>> ---
    print("--> 阶段 3: 正在将SQL执行结果重新组织...")
    execution_results_map = {}
    for i, result_tuple in enumerate(parallel_results):
        meta = job_metadata[i]
        df_index, candidate_idx = meta['df_index'], meta['candidate_idx']
        _, _, _, execution_res, success_flag = result_tuple
        exec_result, exec_error = (list(execution_res), None) if success_flag == 1 else (None, str(execution_res))
        execution_results_map[(df_index, candidate_idx)] = (exec_result, exec_error)

    # --- <<< 修改点：阶段4 - 构建Prompt并准备API调用 >>> ---
    print("--> 阶段 4: 正在准备带有【执行结果】的Prompt...")
    prompts_for_api, valid_rows_info = [], []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="构建Prompts"):
        question = extract_question_from_prompt(row.get(PROMPT_COLUMN))
        if not question: continue

        final_sqls = extract_candidate_sqls(row, n=8)
        full_responses = row.get(FULL_RESPONSES_COLUMN, [])
        if not isinstance(full_responses, (list, pd.Series, type(pd.Series([1]).values))): full_responses = []

        candidate_dossiers = []
        has_valid_candidate = False
        for i in range(8):
            sql = final_sqls[i] if i < len(final_sqls) else None
            response_text = full_responses[i] if i < len(full_responses) else None
            
            if sql or (response_text and isinstance(response_text, str)):
                has_valid_candidate = True
                exec_result, exec_error = execution_results_map.get((index, i), (None, "Result not found"))
                observation_text = format_execution_result(exec_result, exec_error)
                simplified_reasoning = simplify_reasoning_text(response_text)
                dossier = f"""--- Candidate {i+1} ---
Reasoning from Generator (Final Steps):
{simplified_reasoning}

Final Extracted SQL:
{sql if sql else "[No SQL extracted]"}

Execution Observation:
{observation_text}
-------------------"""
                candidate_dossiers.append(dossier)

        if not has_valid_candidate: continue

        selection_prompt = SELECTION_PROMPT_TEMPLATE.format(question=question, formatted_candidates="\n".join(candidate_dossiers))
        prompts_for_api.append(selection_prompt)
        valid_rows_info.append({'df_index': index, 'final_sqls': final_sqls})
    
    # --- 阶段 5: 调用API并处理结果 (逻辑与之前类似) ---
    print(f"--> 阶段 5: 生成了 {len(prompts_for_api)} 个有效的prompt。正在通过API进行推理...")
    api_outputs = []
    if prompts_for_api:
        for prompt in tqdm(prompts_for_api, desc="调用API中"):
            response = call_llm_api_with_retry(client, args.api_model, prompt)
            api_outputs.append(response)

    print("--> 推理完成。正在处理结果...")
    selected_indices, selected_sqls = [], []
    for i, generated_text in tqdm(enumerate(api_outputs), total=len(api_outputs), desc="处理结果"):
        llm_choice_idx = -1
        if generated_text:
            match = re.search(r'\d+', generated_text.strip())
            if match:
                try: llm_choice_idx = int(match.group(0))
                except (ValueError, IndexError): llm_choice_idx = -1
        
        original_final_sqls = valid_rows_info[i]['final_sqls']
        final_sql = None
        if 1 <= llm_choice_idx <= len(original_final_sqls):
            final_sql = original_final_sqls[llm_choice_idx - 1]
        
        selected_indices.append(llm_choice_idx)
        selected_sqls.append(final_sql)

    if valid_rows_info:
        results_df = pd.DataFrame({
            'df_index': [d['df_index'] for d in valid_rows_info],
            'llm_selected_index': selected_indices, 'llm_selected_sql': selected_sqls
        }).set_index('df_index')
        df = df.join(results_df)
        df['llm_selected_index'] = pd.to_numeric(df['llm_selected_index'], errors='coerce').fillna(-1).astype(int)
    else:
        df['llm_selected_index'] = -1
        df['llm_selected_sql'] = None

    print(f"--> 正在保存结果到: {args.output_file}")
    df.to_parquet(args.output_file)
    print("✅ 流程成功完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="通过【并行执行完整SQL】并调用大模型API来选择最佳查询。")
    parser.add_argument("--input_file", type=str, required=True, help="输入的Parquet文件路径。")
    parser.add_argument("--db_path", type=str, required=True, help="存放所有数据库文件的根目录路径。")
    parser.add_argument("--output_file", type=str, required=True, help="输出的Parquet文件路径。")
    parser.add_argument("--api_key", type=str, default="", help="用于API调用的密钥。")
    parser.add_argument("--api_base_url", type=str, default="https://api.nuwaapi.com/v1", help="API的base URL。")
    parser.add_argument("--api_model", type=str, default="gpt-4.1", help="要调用的模型名称。")
    # --- <<< 新增参数 >>> ---
    parser.add_argument("--num_cpus", type=int, default=48, help="用于并行执行SQL的CPU核心数。")
    parser.add_argument("--sql_timeout", type=int, default=90, help="每个SQL查询的执行超时时间（秒）。")
    
    args = parser.parse_args()
    if OpenAI is None or func_timeout is None:
        exit(1)
    main(args)