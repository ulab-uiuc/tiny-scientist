#!/usr/bin/env python3
"""
将两个jsonl文件按intent分组，每组两条（with_tools/without_tools）顺序随机，所有组顺序也随机，输出为csv
"""

import json
import pandas as pd
import random
import os
from typing import Dict, Any

def load_jsonl_to_dict(file_path: str) -> Dict[str, Any]:
    """加载jsonl为intent->数据的字典"""
    data = {}
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    intent = item.get("intent")
                    if intent:
                        data[intent] = item
                except Exception as e:
                    print(f"Error parsing line in {file_path}: {e}")
    return data

def flatten_dict(d: dict) -> dict:
    """展平嵌套字典，保持原始key名称"""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten_dict(v))
        else:
            out[k] = v
    return out

def group_shuffle_and_to_csv():
    with_tools_file = "comparison_test/results_with_tools.jsonl"
    without_tools_file = "comparison_test/results_without_tools.jsonl"
    output_csv = "comparison_test/group_shuffled_results.csv"

    # 加载数据
    with_tools = load_jsonl_to_dict(with_tools_file)
    without_tools = load_jsonl_to_dict(without_tools_file)

    # 找到所有intent
    all_intents = list(set(with_tools.keys()) & set(without_tools.keys()))
    print(f"共有{len(all_intents)}个intent配对")

    # intent组整体打乱
    random.shuffle(all_intents)

    all_rows = []
    for intent in all_intents:
        # 组内顺序随机，直接展平
        pair = [flatten_dict(with_tools[intent]),
                flatten_dict(without_tools[intent])]
        random.shuffle(pair)
        for row in pair:
            all_rows.append(row)

    # 转为DataFrame
    df = pd.DataFrame(all_rows)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"已保存到 {output_csv}")
    print(df.head(4).to_string())

if __name__ == "__main__":
    group_shuffle_and_to_csv() 