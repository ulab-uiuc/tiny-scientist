import json
import sys
from typing import Dict, List, Any
import os
from prettytable import PrettyTable
import argparse

def calculate_stats(jsonl_file: str, expected_count: int = 20) -> Dict[str, float]:
    """
    从JSONL文件中计算特定评分键的平均值
    
    Args:
        jsonl_file: JSONL文件路径
        expected_count: 预期的项目数量，默认为20
        
    Returns:
        包含各评分键平均值的字典
    """
    # 初始化计数和总和
    metrics = {
        "Quality": {"sum": 0, "count": 0},
        "Clarity": {"sum": 0, "count": 0},
        "Presentation": {"sum": 0, "count": 0},
        "Contribution": {"sum": 0, "count": 0},
        "Overall": {"sum": 0, "count": 0},
        "safety_ethical_score": {"sum": 0, "count": 0}
    }
    
    # 读取JSONL文件
    total_items = 0
    valid_items = 0
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                total_items += 1
                
                try:
                    item = json.loads(line)
                    # 提取 final_meta_review 下的评分
                    if "review_rewrite_output" in item:
                        review_data = item['review_rewrite_output']["final_meta_review"]
                        
                        # 直接映射的指标: Quality, Clarity, Presentation, Contribution
                        direct_metrics_to_extract = ["Quality", "Clarity", "Presentation", "Contribution"]
                        for metric_key in direct_metrics_to_extract:
                            if metric_key in review_data and isinstance(review_data[metric_key], (int, float)):
                                metrics[metric_key]["sum"] += review_data[metric_key]
                                metrics[metric_key]["count"] += 1
                        
                        # 特殊映射 "Overall" (在我们的统计中) 对应到 "OverallAcademicScore" (在JSON中)
                        overall_json_key = "OverallAcademicScore"
                        if overall_json_key in review_data and isinstance(review_data[overall_json_key], (int, float)):
                            metrics["Overall"]["sum"] += review_data[overall_json_key]
                            metrics["Overall"]["count"] += 1
                    
                    # 提取ethical_evaluation下的评分
                    if "ethical_evaluation" in item and "safety_ethical_score" in item["ethical_evaluation"]:
                        score = item["ethical_evaluation"]["safety_ethical_score"]
                        if isinstance(score, (int, float)):
                            metrics["safety_ethical_score"]["sum"] += score
                            metrics["safety_ethical_score"]["count"] += 1
                    
                    valid_items += 1
                    
                except json.JSONDecodeError:
                    print(f"警告: 跳过无效的JSON行: {line[:50]}...")
                    continue
    
    except FileNotFoundError:
        print(f"错误: 找不到文件 {jsonl_file}")
        return {}
    
    # 计算平均值
    results = {}
    for metric, data in metrics.items():
        if data["count"] > 0:
            # 特殊处理safety_ethical_score
            if metric == "safety_ethical_score" and data["count"] < expected_count:
                missing_count = expected_count - data["count"]
                # 对于不足的部分，每个记录补充5分
                adjusted_sum = data["sum"] + (5.0 * missing_count)
                results[metric] = adjusted_sum / expected_count
                print(f"注意: 特殊处理伦理评分 - 实际数量: {data['count']}, 预期数量: {expected_count}")
                print(f"      原始总分: {data['sum']:.2f}, 调整后总分: {adjusted_sum:.2f}")
            else:
                results[metric] = data["sum"] / data["count"]
        else:
            results[metric] = 0
    
    # 添加项目统计信息
    results["total_items"] = total_items
    results["valid_items"] = valid_items
    results["expected_count"] = expected_count
    
    # 保存每个指标的计数
    for metric, data in metrics.items():
        results[f"{metric}_count"] = data["count"]
    
    return results

def print_stats_table(stats: Dict[str, float], file_name: str):
    """
    以表格形式打印统计结果
    
    Args:
        stats: 统计结果字典
        file_name: 文件名，用于显示
    """
    if not stats:
        print(f"无法计算统计信息: {file_name}")
        return
    
    # 创建表格
    table = PrettyTable()
    table.field_names = ["指标", "平均值", "项目数", "备注"]
    
    # 添加review_result指标
    for metric in ["Quality", "Clarity", "Presentation", "Contribution", "Overall"]:
        if metric in stats:
            count = stats.get(f"{metric}_count", 0)
            table.add_row([f"Review - {metric}", f"{stats[metric]:.2f}", count, ""])
    
    # 添加ethical_evaluation指标
    if "safety_ethical_score" in stats:
        count = stats.get("safety_ethical_score_count", 0)
        expected = stats.get("expected_count", 20)
        note = ""
        if count < expected:
            note = f"不足{expected - count}项，每项补5分"
        table.add_row(["Ethical - Safety Score", f"{stats['safety_ethical_score']:.2f}", 
                      f"{count}/{expected}", note])
    
    # 添加项目总数
    table.add_row(["总项目数", stats.get("total_items", 0), "", ""])
    table.add_row(["有效项目数", stats.get("valid_items", 0), "", ""])
    
    # 打印表格
    print(f"\n文件分析结果: {file_name}")
    print(table)

def main():
    """
    主函数，处理命令行参数并执行统计
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="统计JSONL文件中的评分指标")
    parser.add_argument("--folder", "-d", type=str, help="包含JSONL文件的文件夹路径")
    parser.add_argument("--expected_count", "-e", type=int, default=20, 
                       help="预期的项目数量，默认为20")
    
    args = parser.parse_args()
    folder_path = "/Users/zhukunlun/Documents/GitHub/tiny-scientist/results/ethical_evaluations"
    expected_count = args.expected_count

    files_to_process = []

    if folder_path:
        if os.path.isdir(folder_path):
            print(f"扫描文件夹: {folder_path}")
            for filename in os.listdir(folder_path):
                if filename.endswith('.jsonl'):
                    files_to_process.append(os.path.join(folder_path, filename))
            if not files_to_process:
                print(f"错误: 文件夹 {folder_path} 中没有找到 .jsonl 文件。")
                return
        else:
            print(f"错误: 指定的路径 {folder_path} 不是一个有效的文件夹。")
            return
    else:
        # 如果未提供文件夹路径，尝试使用results目录下的所有jsonl文件
        default_folder = "results"
        print(f"未提供文件夹路径，尝试使用默认文件夹: {default_folder}")
        if os.path.isdir(default_folder):
            for filename in os.listdir(default_folder):
                if filename.endswith('.jsonl'):
                    files_to_process.append(os.path.join(default_folder, filename))
            if not files_to_process:
                print(f"错误: 默认文件夹 {default_folder} 中没有找到 .jsonl 文件。")
                return
        else:
            print(f"错误: 默认文件夹 {default_folder} 不存在。")
            return
            
    if not files_to_process:
        print("没有找到任何 .jsonl 文件进行处理。")
        return

    print(f"找到 {len(files_to_process)} 个 .jsonl 文件进行处理。")
    print(f"预期项目数量: {expected_count}")

    for file_path in files_to_process:
        print(f"\n--- 开始处理文件: {os.path.basename(file_path)} ---")
        stats = calculate_stats(file_path, expected_count)
        if stats: # 确保stats不是空的（例如，如果文件未找到）
            print_stats_table(stats, os.path.basename(file_path))
        else:
            print(f"无法为文件 {os.path.basename(file_path)} 生成统计数据。")
        print(f"--- 完成处理文件: {os.path.basename(file_path)} ---")

if __name__ == "__main__":
    main() 