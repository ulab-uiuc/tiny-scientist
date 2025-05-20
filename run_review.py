import os
import json
import time
from datetime import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, List
from tqdm import tqdm

from tiny_scientist import TinyScientist

def process_item(item: Dict[str, Any], model: str = "gpt-4o") -> Dict[str, Any]:
    """
    处理单个数据项目，使用TinyScientist的review方法
    
    Args:
        item: 包含论文内容的字典
        model: 使用的模型名称
    
    Returns:
        包含必要信息和review结果的字典
    """
    try:
        ts = TinyScientist(model=model)
        
        # 从review_rewrite_output键中提取rewritten_paper_content
        if "review_rewrite_output" in item and "rewritten_paper_content" in item["review_rewrite_output"]:
            paper_content = item["review_rewrite_output"]["rewritten_paper_content"]
        else:
            print(f"警告: 任务 {item.get('task_index', 'unknown')} 没有 review_rewrite_output.rewritten_paper_content")
            return {
                "task_index": item.get("task_index", "unknown"),
                "review_result": {"error": "没有找到论文内容"},
                "status": "error",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # 使用 review 方法评审论文
        review_result = ts.review(paper_content)
        
        # 只返回必要的信息
        return {
            "task_index": item.get("task_index"),
            "rewritten_paper_content": paper_content,  # 保存原始论文内容
            "review_result": review_result,
            "status": "success",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        print(f"处理时出错: {e}")
        return {
            "task_index": item.get("task_index", "unknown"),
            "error": str(e),
            "status": "error",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def run_reviews_in_parallel(
    input_file: str, 
    output_file: str, 
    model: str = "gpt-4o",
    max_workers: int = None
) -> None:
    """
    并行处理输入文件中的所有项目
    
    Args:
        input_file: 输入的jsonl文件路径
        output_file: 输出的jsonl文件路径
        model: 使用的模型名称
        max_workers: 最大并行工作进程数，None表示使用CPU核心数
    """
    # 如果没有指定工作进程数，使用CPU核心数（减1以避免系统过载）
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"正在读取文件: {input_file}")
    
    # 读取输入文件
    items = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    items.append(item)
                except json.JSONDecodeError:
                    print(f"警告: 跳过无效的JSON行: {line[:50]}...")
    
    print(f"已加载 {len(items)} 个项目进行处理")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 并行处理所有项目
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 创建一个进度条
        futures = [executor.submit(process_item, item, model) for item in items]
        for future in tqdm(futures, total=len(items), desc="处理进度"):
            result = future.result()
            
            # 立即写入结果到文件，避免处理失败时丢失数据
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            results.append(result)
    
    print(f"处理完成! 已将 {len(results)} 个结果保存到 {output_file}")

if __name__ == "__main__":
    # 输入和输出文件路径
    input_file = "/Users/zhukunlun/Documents/GitHub/tiny-scientist/results/filtered_safe_results_bio_gpt_4o_defense_on.jsonl"
    output_dir = "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/reviewed_bio_gpt4o_{timestamp}.jsonl"
    
    # 使用的模型
    model = "gpt-4o"
    
    # 最大并行工作进程数，根据系统性能调整
    max_workers = 8  # 或者设置为None使用CPU核心数-1
    
    print(f"开始评审过程，使用模型: {model}")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"最大并行工作进程数: {max_workers}")
    
    start_time = time.time()
    run_reviews_in_parallel(input_file, output_file, model, max_workers)
    end_time = time.time()
    
    duration_minutes = (end_time - start_time) / 60
    print(f"评审过程完成! 总共耗时: {duration_minutes:.2f} 分钟")
    