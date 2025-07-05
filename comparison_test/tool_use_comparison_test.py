import json
import time
import os
from typing import List, Dict, Any
from tiny_scientist import TinyScientist

def generate_test_cases() -> List[str]:
    """生成10个类似但不同的测试用例"""
    base_intent = "Benchmarking adaptive step size strategies using a convex quadratic optimization function"
    
    test_cases = [
        "Benchmarking adaptive step size strategies using a convex quadratic optimization function",
        "Evaluating adaptive learning rate methods for neural network training on image classification tasks",
        "Comparing gradient descent variants for logistic regression optimization with regularization",
        "Analyzing momentum-based optimization algorithms for deep learning model convergence",
        "Investigating adaptive optimization techniques for reinforcement learning policy gradients",
        "Studying adaptive parameter tuning methods for support vector machine training",
        "Examining adaptive step size algorithms for convex optimization problems",
        "Comparing adaptive optimization strategies for natural language processing models",
        "Analyzing adaptive learning rate schedules for computer vision applications",
        "Investigating adaptive optimization methods for time series forecasting models"
    ]
    
    return test_cases

def load_finished_intents(jsonl_path: str) -> set:
    """加载已完成的intent集合"""
    finished = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    finished.add(data.get("intent"))
                except Exception:
                    continue
    return finished

def run_single_test(scientist: TinyScientist, intent: str, tool_use: bool) -> Dict[str, Any]:
    """运行单个测试用例"""
    print(f"\n{'='*50}")
    print(f"Running test with tool_use={tool_use}")
    print(f"Intent: {intent}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        result = scientist.think(intent=intent, tool_use=tool_use)
        end_time = time.time()
        
        return {
            "intent": intent,
            "tool_use": tool_use,
            "result": result,
            "execution_time": end_time - start_time,
            "success": True
        }
    except Exception as e:
        end_time = time.time()
        return {
            "intent": intent,
            "tool_use": tool_use,
            "result": None,
            "error": str(e),
            "execution_time": end_time - start_time,
            "success": False
        }

def run_comparison_tests():
    """运行对比测试，支持断点续跑，输出jsonl"""
    print("🔬 Starting Tool Use Comparison Test (jsonl mode)")
    print("="*60)
    
    # 生成测试用例
    test_cases = generate_test_cases()
    print(f"Generated {len(test_cases)} test cases")
    
    # 初始化科学家
    scientist = TinyScientist(model="gpt-4o")
    
    # 输出文件
    with_tools_jsonl = "./results_with_tools.jsonl"
    without_tools_jsonl = "./results_without_tools.jsonl"
    
    # 加载已完成的intent
    finished_with_tools = load_finished_intents(with_tools_jsonl)
    finished_without_tools = load_finished_intents(without_tools_jsonl)
    
    print(f"Already finished with tools: {len(finished_with_tools)}")
    print(f"Already finished without tools: {len(finished_without_tools)}")
    
    # 打开文件句柄
    with open(with_tools_jsonl, 'a', encoding='utf-8') as f_with, \
         open(without_tools_jsonl, 'a', encoding='utf-8') as f_without:
        
        for i, intent in enumerate(test_cases, 1):
            print(f"\n📋 Test Case {i}/{len(test_cases)}")
            # 先跑with tools
            if intent not in finished_with_tools:
                result_with_tools = run_single_test(scientist, intent, tool_use=True)
                f_with.write(json.dumps(result_with_tools, ensure_ascii=False) + '\n')
                f_with.flush()
                print(f"[WRITE] with_tools: {intent}")
                time.sleep(2)
            else:
                print(f"[SKIP] with_tools: {intent}")
            # 再跑without tools
            if intent not in finished_without_tools:
                result_without_tools = run_single_test(scientist, intent, tool_use=False)
                f_without.write(json.dumps(result_without_tools, ensure_ascii=False) + '\n')
                f_without.flush()
                print(f"[WRITE] without_tools: {intent}")
                time.sleep(2)
            else:
                print(f"[SKIP] without_tools: {intent}")
    print("\nAll test cases finished or skipped. Results in .jsonl files.")

if __name__ == "__main__":
    run_comparison_tests() 