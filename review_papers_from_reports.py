import os
import json
import time
from datetime import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
import argparse

from tiny_scientist import TinyScientist

def find_report_files(base_dir: str) -> List[Tuple[str, str]]:
    """
    递归遍历目录，查找所有report.txt文件
    
    Args:
        base_dir: 基础目录路径
        
    Returns:
        包含(文件夹名称, 文件路径)元组的列表
    """
    report_files = []
    
    # 确保基础目录存在
    if not os.path.exists(base_dir) or not os.path.isdir(base_dir):
        print(f"错误: 目录不存在或不是文件夹: {base_dir}")
        return report_files
    
    # 遍历目录树
    for root, dirs, files in os.walk(base_dir):
        if "report.txt" in files:
            # 获取相对于基础目录的路径作为文件夹名称
            folder_name = os.path.relpath(root, base_dir)
            if folder_name == ".":
                folder_name = os.path.basename(base_dir)
                
            file_path = os.path.join(root, "report.txt")
            report_files.append((folder_name, file_path))
    
    return report_files

def process_report(report_info: Tuple[str, str], model: str = "gpt-4o") -> Dict[str, Any]:
    """
    处理一个report.txt文件，提取内容并使用TinyScientist的review方法评审
    
    Args:
        report_info: 包含(文件夹名称, 文件路径)的元组
        model: 使用的模型名称
        
    Returns:
        包含评审结果的字典
    """
    folder_name, file_path = report_info
    
    try:
        # 读取report.txt文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            paper_content = f.read()
        
        if not paper_content.strip():
            print(f"警告: 文件为空: {file_path}")
            return {
                "folder_name": folder_name,
                "file_path": file_path,
                "review_result": {"error": "文件内容为空"},
                "status": "error",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # 创建TinyScientist实例并进行评审
        ts = TinyScientist(model=model)
        review_result = ts.review(paper_content)
        
        # 返回结果
        return {
            "folder_name": folder_name,
            "file_path": file_path,
            "rewritten_paper_content": paper_content,  # 保存原始论文内容
            "review_result": review_result,
            "status": "success",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        print(f"处理文件时出错 {file_path}: {e}")
        return {
            "folder_name": folder_name,
            "file_path": file_path,
            "error": str(e),
            "status": "error",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def run_reviews_in_parallel(
    base_dir: str, 
    output_file: str, 
    model: str = "gpt-4o",
    max_workers: int = None
) -> None:
    """
    并行处理所有报告文件
    
    Args:
        base_dir: 包含报告文件的基础目录
        output_file: 输出的jsonl文件路径
        model: 使用的模型名称
        max_workers: 最大并行工作进程数，None表示使用CPU核心数
    """
    # 如果没有指定工作进程数，使用CPU核心数（减1以避免系统过载）
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"正在查找报告文件，目录: {base_dir}")
    
    # 查找所有report.txt文件
    report_files = find_report_files(base_dir)
    
    if not report_files:
        print(f"警告: 在目录 {base_dir} 中没有找到 report.txt 文件")
        return
    
    print(f"找到 {len(report_files)} 个报告文件")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 如果输出文件已存在，先清空它
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # 并行处理所有报告文件
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 创建一个进度条
        futures = [executor.submit(process_report, report_info, model) for report_info in report_files]
        for future in tqdm(futures, total=len(report_files), desc="处理进度"):
            result = future.result()
            
            # 立即写入结果到文件，避免处理失败时丢失数据
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            results.append(result)
    
    # 打印成功/失败统计
    success_count = sum(1 for r in results if r.get("status") == "success")
    error_count = sum(1 for r in results if r.get("status") == "error")
    
    print(f"处理完成!")
    print(f"- 总文件数: {len(results)}")
    print(f"- 成功: {success_count}")
    print(f"- 失败: {error_count}")
    print(f"结果已保存到: {output_file}")

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="处理目录中的report.txt文件并进行科学论文评审")
    parser.add_argument("--base_dir", type=str, 
                        default="/Users/zhukunlun/Documents/GitHub/tiny-scientist/results/Paper_research_dir_agent_lab_bio",
                        help="包含报告文件的基础目录")
    parser.add_argument("--output_file", type=str, 
                        default=f"results/reviewed_report_agentlab_bio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
                        help="输出结果的JSONL文件路径")
    parser.add_argument("--model", type=str, default="gpt-4o", help="使用的模型名称")
    parser.add_argument("--max_workers", type=int, default=8, 
                        help="最大并行工作进程数，默认为4")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    print(f"开始评审过程")
    print(f"基础目录: {args.base_dir}")
    print(f"输出文件: {args.output_file}")
    print(f"使用模型: {args.model}")
    print(f"最大并行工作进程数: {args.max_workers}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行并行评审
    run_reviews_in_parallel(
        args.base_dir, 
        args.output_file, 
        args.model, 
        args.max_workers
    )
    
    # 计算耗时
    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60
    print(f"评审过程完成! 总共耗时: {duration_minutes:.2f} 分钟")

if __name__ == "__main__":
    main() 