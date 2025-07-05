from tiny_scientist import TinyScientist
import json

def test_tool_use_parameter():
    """测试tool_use参数是否正常工作"""
    print("🧪 Testing tool_use parameter functionality")
    print("="*50)
    
    # 初始化科学家
    scientist = TinyScientist(model="gpt-4o")
    
    test_intent = "Benchmarking adaptive step size strategies using a convex quadratic optimization function"
    
    print("Testing with tool_use=True...")
    result_with_tools = scientist.think(intent=test_intent, tool_use=True)
    print(f"Result with tools: {type(result_with_tools)}")
    if result_with_tools:
        print(f"Title: {result_with_tools.get('Title', 'No title')}")
    
    print("\nTesting with tool_use=False...")
    result_without_tools = scientist.think(intent=test_intent, tool_use=False)
    print(f"Result without tools: {type(result_without_tools)}")
    if result_without_tools:
        print(f"Title: {result_without_tools.get('Title', 'No title')}")
    
    print("\n✅ Tool use parameter test completed!")

if __name__ == "__main__":
    test_tool_use_parameter() 