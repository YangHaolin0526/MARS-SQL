import os
import json
import re
from tqdm import tqdm
import argparse
import ray

def minify_sql_schema(sql_text: str) -> str:
    """
    压缩 SQL schema 字符串，移除不必要的空格、换行和注释以减少 token 数量。
    """
    lines = sql_text.splitlines()
    processed_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line and not stripped_line.startswith('--'):
            processed_lines.append(stripped_line)
    single_line_schema = ' '.join(processed_lines)
    minified_schema = re.sub(r'\s+', ' ', single_line_schema)
    return minified_schema.strip()


@ray.remote
class SchemaMemoryStoreActor:
    """
    A Ray Actor that serves as a global, stateful service for schemas.
    Any other worker in the Ray cluster can get a handle to this actor
    by its name and call its methods.
    """
    def __init__(self, base_db_path: str, cache_file_path: str):
        self.schema_store = {}
        self.base_db_path = base_db_path
        
        # 在 Actor 初始化时，直接加载或构建数据
        if not self._load_from_json(cache_file_path):
            self._build()
            self._save_to_json(cache_file_path)
        
        print(f"SchemaMemoryStoreActor is ready. Loaded {len(self.schema_store)} schemas.")


    def _get_schema_for_db(self, db_id: str) -> str:
        """
        为单个 db_id 读取、处理并返回其 schema。
        这是一个内部辅助方法。
        """
        schema_file_path = os.path.join(self.base_db_path, db_id, 'schema.sql')
        try:
            if os.path.exists(schema_file_path):
                with open(schema_file_path, 'r', encoding='utf-8') as f:
                    original_schema = f.read()
                    # 现在这个函数调用是有效的
                    return minify_sql_schema(original_schema)
            else:
                return f"-- Schema file not found for db_id '{db_id}'."
        except Exception as e:
            return f"-- Error reading schema file for db_id '{db_id}': {e}"

    def _build(self):
        """
        遍历所有数据库目录，构建完整的 schema 存储库。
        """
        print(f"开始从 {self.base_db_path} 构建 Schema 存储库...")
        db_ids = [d for d in os.listdir(self.base_db_path) if os.path.isdir(os.path.join(self.base_db_path, d))]
        
        if not db_ids:
            print(f"警告: 在指定路径 '{self.base_db_path}' 下没有找到任何数据库目录。")
            return

        for db_id in tqdm(db_ids, desc="Processing Schemas"):
            self.schema_store[db_id] = self._get_schema_for_db(db_id)
        
        print(f"构建完成！共加载了 {len(self.schema_store)} 个数据库的 Schema。")

    def _save_to_json(self, file_path: str):
        """
        将内存中的 schema 存储库保存到 JSON 文件中。
        """
        print(f"正在将 Schema 存储库缓存到 {file_path}...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.schema_store, f, indent=2) # 使用 indent=2 更紧凑
        print("保存成功。")

    def _load_from_json(self, file_path: str) -> bool:
        """
        从 JSON 文件加载 schema 存储库。
        """
        if not os.path.exists(file_path):
            return False
        
        print(f"从缓存文件 {file_path} 加载 Schema 存储库...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.schema_store = json.load(f)
            print(f"加载成功！共加载了 {len(self.schema_store)} 个数据库的 Schema。")
            return True
        except (json.JSONDecodeError, IOError) as e:
            print(f"加载缓存失败: {e}。将重新构建。")
            return False

    def retrieve(self, db_id: str) -> str:
        """
        Retrieves a schema by db_id. This method can be called remotely.
        e.g., ray.get(actor_handle.retrieve.remote(db_id))
        """
        return self.schema_store.get(db_id, f"-- Error: Schema for db_id '{db_id}' not found.")


# =================================================================== #
# 2. 补充独立运行模块，用于构建和管理缓存
# =================================================================== #
if __name__ == '__main__':
    # --- 配置: 你可以根据需要修改这里的默认路径 ---
    DEFAULT_DATABASES_BASE_PATH = '/import/home3/jzhanggr/haolin/bird_data/train/train_databases'
    DEFAULT_CACHE_FILE = '/import/home3/jzhanggr/haolin/SkyRL/SkyRL-653-data/data/schema_memory_store.json'
    
    parser = argparse.ArgumentParser(
        description="构建或更新数据库 Schema 的缓存文件。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--db_path', 
        type=str, 
        default=DEFAULT_DATABASES_BASE_PATH, 
        help="包含所有数据库子目录的根路径。"
    )
    parser.add_argument(
        '--cache_file', 
        type=str, 
        default=DEFAULT_CACHE_FILE, 
        help="用于保存/加载 Schema 缓存的 JSON 文件路径。"
    )
    parser.add_argument(
        '--force_rebuild', 
        action='store_true', 
        help="强制重新构建缓存，即使缓存文件已存在。"
    )
    args = parser.parse_args()

    print("Schema Memory Store 管理脚本")
    print("-" * 30)
    
    store = SchemaMemoryStore(args.db_path)
    
    if args.force_rebuild or not os.path.exists(args.cache_file):
        if args.force_rebuild:
            print("检测到 --force_rebuild 参数，将强制重建缓存。")
        else:
            print("缓存文件不存在，开始首次构建。")
        store.build()
        store.save_to_json(args.cache_file)
    else:
        print(f"缓存文件 '{args.cache_file}' 已存在。脚本执行完毕。")
        print("如需更新，请使用 --force_rebuild 标志。")

    print("-" * 30)