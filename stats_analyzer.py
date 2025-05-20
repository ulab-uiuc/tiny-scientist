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
                    
                    # 提取review_result下的评分
                    if "review_result" in item:
                        review = item["review_result"]
                        for metric in ["Quality", "Clarity", "Presentation", "Contribution", "Overall"]:
                            if metric in review and isinstance(review[metric], (int, float)):
                                metrics[metric]["sum"] += review[metric]
                                metrics[metric]["count"] += 1
                    
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
    parser.add_argument("--file", "-f", type=str, help="输入的JSONL文件路径")
    parser.add_argument("--expected_count", "-e", type=int, default=20, 
                       help="预期的项目数量，默认为20")
    
    # 解析命令行参数
    try:
        args = parser.parse_args()
        file_path = args.file
        expected_count = args.expected_count
    except Exception as e:
        # 如果解析失败（例如，在某些环境中），则回退到手动解析
        print(f"参数解析出错，使用简单解析: {e}")
        file_path = None
        expected_count = 20
        
        # 简单解析sys.argv
        if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
            file_path = sys.argv[1]
        if len(sys.argv) > 2 and not sys.argv[2].startswith('-'):
            try:
                expected_count = int(sys.argv[2])
            except ValueError:
                pass
    
    # 如果未提供文件路径，尝试使用最新的JSONL文件
    if not file_path:
        # 默认文件路径
        file_path = "/Users/zhukunlun/Documents/GitHub/tiny-scientist/results/ethical_scored_SafeScientist_bio_gpt4o_20250519_222755.jsonl"
        # 如果默认路径不存在，尝试找到results目录下的jsonl文件
        if not os.path.exists(file_path):
            try:
                results_dir = "results"
                if os.path.exists(results_dir) and os.path.isdir(results_dir):
                    jsonl_files = [f for f in os.listdir(results_dir) if f.endswith('.jsonl')]
                    if jsonl_files:
                        # 使用最新的文件
                        latest_file = max(jsonl_files, key=lambda f: os.path.getmtime(os.path.join(results_dir, f)))
                        file_path = os.path.join(results_dir, latest_file)
                        print(f"未提供文件路径，使用最新的JSONL文件: {file_path}")
                    else:
                        print(f"错误: 在{results_dir}目录中找不到JSONL文件")
                        return
                else:
                    print(f"错误: 找不到默认文件 {file_path} 或 {results_dir} 目录")
                    return
            except Exception as e:
                print(f"尝试查找JSONL文件时出错: {e}")
                return
    
    print(f"使用文件: {file_path}")
    print(f"预期项目数量: {expected_count}")
    
    # 计算并打印统计结果
    stats = calculate_stats(file_path, expected_count)
    print_stats_table(stats, os.path.basename(file_path))

if __name__ == "__main__":
    main() 