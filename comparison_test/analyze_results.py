#!/usr/bin/env python3
"""
统计脚本：分析工具使用对比结果
读取四个csv文件，计算tool use的win rate和平均分数
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def load_and_process_data():
    """加载并处理数据"""
    
    # 读取基础文件（包含tool_use信息）
    base_file = "human_annotate_results/group_shuffled_results_annota.csv"
    base_df = pd.read_csv(base_file)
    
    # 读取三个标注文件
    annota_files = [
        "human_annotate_results/group_shuffled_results_annota_1.csv",
        "human_annotate_results/group_shuffled_results_annota_2.csv", 
        "human_annotate_results/group_shuffled_results_annota_3.csv"
    ]
    
    # 加载所有标注数据
    annota_dfs = []
    for file in annota_files:
        df = pd.read_csv(file)
        annota_dfs.append(df)
    
    print(f"基础文件行数: {len(base_df)}")
    print(f"标注文件行数: {[len(df) for df in annota_dfs]}")
    
    # 确保所有文件的行数一致
    assert len(base_df) == len(annota_dfs[0]) == len(annota_dfs[1]) == len(annota_dfs[2]), "文件行数不一致"
    
    return base_df, annota_dfs

def calculate_quality_scores(base_df: pd.DataFrame, annota_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """计算每个项目的质量得分"""
    
    # 提取Quality列并计算平均分
    quality_scores = []
    for i in range(len(base_df)):
        scores = []
        for df in annota_dfs:
            if pd.notna(df.iloc[i]['Quality']):
                scores.append(df.iloc[i]['Quality'])
        
        if scores:
            avg_score = sum(scores) / len(scores)
        else:
            avg_score = np.nan
            
        quality_scores.append(avg_score)
    
    # 创建结果DataFrame
    result_df = base_df.copy()
    result_df['Quality_Avg'] = quality_scores
    
    return result_df

def analyze_tool_use_performance(result_df: pd.DataFrame) -> Dict:
    """分析工具使用性能"""
    
    # 分离true和false组
    true_group = result_df[result_df['tool_use'] == True]
    false_group = result_df[result_df['tool_use'] == False]
    
    print(f"\n=== 数据统计 ===")
    print(f"True组数量: {len(true_group)}")
    print(f"False组数量: {len(false_group)}")
    
    # 计算平均分数
    true_avg = true_group['Quality_Avg'].mean()
    false_avg = false_group['Quality_Avg'].mean()
    
    print(f"\n=== 平均分数 ===")
    print(f"Tool Use True 平均分数: {true_avg:.3f}")
    print(f"Tool Use False 平均分数: {false_avg:.3f}")
    print(f"差异 (True - False): {true_avg - false_avg:.3f}")
    
    # 计算win rate
    # 按intent分组，比较每组内true和false的得分
    win_count = 0
    total_comparisons = 0
    
    for intent in result_df['intent'].unique():
        intent_data = result_df[result_df['intent'] == intent]
        if len(intent_data) == 2:  # 确保有true和false两组
            true_score = intent_data[intent_data['tool_use'] == True]['Quality_Avg'].iloc[0]
            false_score = intent_data[intent_data['tool_use'] == False]['Quality_Avg'].iloc[0]
            
            if pd.notna(true_score) and pd.notna(false_score):
                total_comparisons += 1
                if true_score > false_score:
                    win_count += 1
                    print(f"Win: {intent[:50]}... (True: {true_score:.2f} > False: {false_score:.2f})")
                else:
                    print(f"Lose: {intent[:50]}... (True: {true_score:.2f} < False: {false_score:.2f})")
    
    win_rate = win_count / total_comparisons if total_comparisons > 0 else 0
    
    print(f"\n=== Win Rate 分析 ===")
    print(f"总比较次数: {total_comparisons}")
    print(f"True获胜次数: {win_count}")
    print(f"Win Rate: {win_rate:.3f} ({win_rate*100:.1f}%)")
    
    return {
        'true_avg': true_avg,
        'false_avg': false_avg,
        'win_rate': win_rate,
        'win_count': win_count,
        'total_comparisons': total_comparisons
    }

def detailed_analysis(result_df: pd.DataFrame):
    """详细分析"""
    
    print(f"\n=== 详细分析 ===")
    
    # 按intent分组显示详细对比
    for intent in result_df['intent'].unique():
        intent_data = result_df[result_df['intent'] == intent]
        if len(intent_data) == 2:
            true_row = intent_data[intent_data['tool_use'] == True].iloc[0]
            false_row = intent_data[intent_data['tool_use'] == False].iloc[0]
            
            print(f"\nIntent: {intent[:60]}...")
            print(f"  True (tool_use=True):  {true_row['Quality_Avg']:.3f}")
            print(f"  False (tool_use=False): {false_row['Quality_Avg']:.3f}")
            print(f"  差异: {true_row['Quality_Avg'] - false_row['Quality_Avg']:.3f}")

def main():
    """主函数"""
    print("开始分析工具使用对比结果...")
    
    # 加载数据
    base_df, annota_dfs = load_and_process_data()
    
    # 计算质量得分
    result_df = calculate_quality_scores(base_df, annota_dfs)
    
    # 保存处理后的数据
    output_file = "comparison_test/processed_results.csv"
    result_df.to_csv(output_file, index=False)
    print(f"\n处理后的数据已保存到: {output_file}")
    
    # 分析性能
    stats = analyze_tool_use_performance(result_df)
    
    # 详细分析
    detailed_analysis(result_df)
    
    # 输出总结
    print(f"\n=== 总结 ===")
    print(f"Tool Use True 平均分数: {stats['true_avg']:.3f}")
    print(f"Tool Use False 平均分数: {stats['false_avg']:.3f}")
    print(f"Win Rate: {stats['win_rate']:.3f} ({stats['win_rate']*100:.1f}%)")
    
    if stats['true_avg'] > stats['false_avg']:
        print("结论: 使用工具(Tool Use=True)在平均分数上表现更好")
    else:
        print("结论: 不使用工具(Tool Use=False)在平均分数上表现更好")
    
    if stats['win_rate'] > 0.5:
        print("结论: 使用工具(Tool Use=True)在配对比较中获胜率更高")
    else:
        print("结论: 不使用工具(Tool Use=False)在配对比较中获胜率更高")

if __name__ == "__main__":
    main() 