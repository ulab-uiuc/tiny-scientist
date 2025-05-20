import os
import json
import sys
from typing import Dict, List, Any
from tqdm import tqdm
import argparse

# 导入fuse.py中的安全检查函数
try:
    from fuse import check_prompt_safety
except ImportError:
    # 如果无法直接导入，则将fuse.py的目录添加到系统路径中
    import importlib.util
    import os
    
    # 尝试从当前目录加载fuse.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fuse_path = os.path.join(current_dir, "fuse.py")
    
    if os.path.exists(fuse_path):
        spec = importlib.util.spec_from_file_location("fuse", fuse_path)
        fuse = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fuse)
        check_prompt_safety = fuse.check_prompt_safety
    else:
        raise ImportError("无法找到fuse.py文件，请确保它在同一目录下")

def filter_jsonl_by_safety(input_file: str, output_file: str) -> Dict[str, int]:
    """
    根据original_task_prompt的安全性过滤JSONL文件中的项目
    
    Args:
        input_file: 输入的JSONL文件路径
        output_file: 输出的JSONL文件路径
        
    Returns:
        包含处理统计信息的字典
    """
    stats = {
        "total": 0,
        "safe": 0,
        "unsafe": 0,
        "error": 0,
        "no_prompt": 0
    }
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 如果输出文件已存在，先清空它
    if os.path.exists(output_file):
        os.remove(output_file)
    
    try:
        # 读取输入文件
        items = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        items.append(item)
                        stats["total"] += 1
                    except json.JSONDecodeError:
                        print(f"警告: 跳过无效的JSON行: {line[:50]}...")
                        stats["error"] += 1
        
        print(f"已加载 {len(items)} 个项目进行安全检查")
        
        # 处理每个项目
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for item in tqdm(items, desc="安全检查进度"):
                # 检查是否存在original_task_prompt
                if "original_task_prompt" not in item:
                    print(f"警告: 项目缺少original_task_prompt字段: {item.get('task_index', 'unknown')}")
                    stats["no_prompt"] += 1
                    continue
                
                prompt = item["original_task_prompt"]
                
                try:
                    # 进行安全检查
                    safety_result = check_prompt_safety(prompt)
                    
                    # 添加安全检查结果到项目
                    item["safety_check_result"] = safety_result
                    
                    # 如果安全，保存到输出文件
                    if safety_result == 1:
                        out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        stats["safe"] += 1
                    else:
                        stats["unsafe"] += 1
                        
                except Exception as e:
                    print(f"处理提示时出错: {e}")
                    stats["error"] += 1
        
        return stats
    
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        return stats
    except Exception as e:
        print(f"处理过程中出错: {e}")
        return stats

def main():
    """
    主函数，处理命令行参数并执行过滤
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="根据提示的安全性过滤JSONL文件")
    parser.add_argument("--input", "-i", type=str, 
                        default="results/results_bio_gpt_4o_defense_on.jsonl",
                        help="输入的JSONL文件路径")
    parser.add_argument("--output", "-o", type=str, 
                        default=f"results/filtered_safe_bio_{os.path.basename(os.path.splitext('results/results_bio_gpt_4o_defense_on.jsonl')[0])}.jsonl",
                        help="输出的JSONL文件路径")
    
    # 解析命令行参数
    args = parser.parse_args()
    input_file = args.input
    
    # 如果未指定输出文件，则基于输入文件名生成
    if args.output == parser.get_default('output'):
        output_file = f"results/filtered_safe_{os.path.basename(input_file)}"
    else:
        output_file = args.output
    
    print(f"开始安全过滤处理")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    # 执行过滤
    stats = filter_jsonl_by_safety(input_file, output_file)
    
    # 打印统计信息
    print("\n处理完成!")
    print(f"- 总项目数: {stats['total']}")
    print(f"- 安全项目数: {stats['safe']}")
    print(f"- 不安全项目数: {stats['unsafe']}")
    print(f"- 缺少提示项目数: {stats['no_prompt']}")
    print(f"- 出错项目数: {stats['error']}")
    print(f"\n安全项目已保存到: {output_file}")

if __name__ == "__main__":
    main() 