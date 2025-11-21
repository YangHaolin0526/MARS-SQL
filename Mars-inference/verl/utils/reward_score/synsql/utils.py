# import sqlite3
from func_timeout import func_timeout, FunctionTimedOut
# import sys

# def execute_sql(data_idx, db_file, sql):
#     try:
#         conn = sqlite3.connect(db_file)
#         cursor = conn.cursor()
#         conn.execute("BEGIN TRANSACTION;")
#         cursor.execute(sql)
#         execution_res = frozenset(cursor.fetchall())
#         conn.rollback()
#         conn.close()
#         return data_idx, db_file, sql, execution_res, 1
#     except Exception as e:
#         print(f"Error executing SQL: {e}")
#         conn.rollback()
#         conn.close()
#         return data_idx, db_file, sql, None, 0

import sqlite3
import os # For checking file existence (optional debug)

def execute_sql(data_idx, db_file, sql):
    conn = None  # Initialize conn to None
    try:
        # --- Optional Debugging ---
        print(f"[execute_sql DEBUG] Attempting to connect to: {db_file}")
        if not os.path.exists(db_file):
            print(f"[execute_sql DEBUG] ERROR: DB file does not exist at path: {db_file}")
        #     # You might want to return the error immediately if the file doesn't exist
        #     # Or let sqlite3.connect() handle it, which it will.
        # else:
        #     print(f"[execute_sql DEBUG] DB file exists: {db_file}")
        # --- End Optional Debugging ---

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        # BEGIN TRANSACTION and ROLLBACK are often not strictly necessary for read-only SELECT queries
        # but don't harm. If there were writes, you'd need COMMIT on success.
        # conn.execute("BEGIN TRANSACTION;") # Optional
        cursor.execute(sql)
        execution_res = frozenset(cursor.fetchall())
        # conn.rollback() # Optional for read-only
        return data_idx, db_file, sql, execution_res, 1  # Success
    except sqlite3.Error as e_sqlite: # Catch SQLite specific errors for better info
        print(f"SQLite error for DB: '{db_file}' with SQL: '{sql}'. Error: {e_sqlite}")
        return data_idx, db_file, sql, None, 0  # Failure
    except Exception as e: # Catch other potential errors
        print(f"Generic error executing SQL for DB: '{db_file}' with SQL: '{sql}'. Error: {e}")
        return data_idx, db_file, sql, None, 0  # Failure
    finally:
        if conn:  # Only attempt to close if conn was successfully assigned
            conn.close()

def execute_sql_wrapper(data_idx, db_file, sql, timeout, output_str):
    try:
        res = func_timeout(timeout, execute_sql, args=(data_idx, db_file, sql))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        print(f"Data index:{data_idx}\nSQL:\n{sql}\nTime Out!")
        print("-"*30)
        res = (data_idx, db_file, sql, None, 0)
    except Exception as e:
        print(f"Error executing SQL: {e}")
        res = (data_idx, db_file, sql, None, 0)

    # Append the output to the tuple
    if isinstance(res, tuple):
        res = res + (output_str,)
        
    return res