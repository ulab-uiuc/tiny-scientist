# Improved Diagram Tool - Refactored with Direct utils.llm Usage

## 🔄 Refactoring Summary

The improved diagram tool has been refactored to use `utils.llm` directly, following the same pattern as other tools in `tool.py`. This provides better consistency, error handling, and cost tracking.

## 🏗️ Architecture Changes

### **Before (Using DrawerTool)**
```python
# Old approach - indirect usage
self.base_drawer = DrawerTool(model=model, temperature=temperature)
result = self.base_drawer.run(query)
```

### **After (Direct utils.llm Usage)**
```python
# New approach - direct usage like other tools
self.client, self.model_name = create_client(model)
self.cost_tracker = CostTracker()

llm_response, msg_history = get_response_from_llm(
    user_prompt,
    model=self.model_name,
    client=self.client,
    system_message=system_prompt,
    temperature=self.temperature,
    cost_tracker=self.cost_tracker,
    task_name="generate_diagram"
)
```

## 🔧 Key Improvements

### 1. **Consistent Tool Interface**
```python
def run(self, query: str) -> Dict[str, Dict[str, str]]:
    """
    Main interface method like other tools in tool.py
    """
    try:
        query_dict = json.loads(query)
        section_name = query_dict.get("section_name")
        section_content = query_dict.get("section_content")
        preferred_format = query_dict.get("format", "mermaid")
    except (json.JSONDecodeError, TypeError, AttributeError):
        raise ValueError(
            "Expected query to be a JSON string with 'section_name' and 'section_content'."
        )
    
    # Generate diagram
    result = self.generate_diagram(
        section_name=section_name,
        section_content=section_content,
        preferred_format=preferred_format,
        auto_fallback=True
    )
    
    # Format result like other tools
    results = {}
    if result["success"]:
        results["diagram"] = {
            "summary": result.get("summary", ""),
            "content": result.get("content", ""),
            "format": result.get("format", ""),
            "files": result.get("files", {})
        }
    
    self.cost_tracker.report()
    return results
```

### 2. **Direct LLM Integration**
```python
@api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
def _generate_in_format(self, section_name: str, section_content: str, 
                       format_name: str) -> Dict[str, Any]:
    """
    Generate diagram in specific format using utils.llm directly
    """
    try:
        # Create format-specific prompt
        system_prompt = self._create_format_prompt(section_name, section_content, format_name)
        user_prompt = f"Generate a {format_name} diagram for the {section_name} section."
        
        # Use utils.llm directly like other tools
        llm_response, msg_history = get_response_from_llm(
            user_prompt,
            model=self.model_name,
            client=self.client,
            system_message=system_prompt,
            temperature=self.temperature,
            cost_tracker=self.cost_tracker,
            task_name="generate_diagram"
        )
        
        # Extract and validate format-specific content
        format_content = self._extract_format_content(llm_response, format_name)
        
        if not format_content:
            return {"success": False, "error": f"Invalid {format_name} content"}
        
        # Save files
        files = self._save_diagram_files(format_content, section_name, format_name)
        
        return {
            "success": True,
            "format": format_name,
            "summary": self._extract_summary(llm_response),
            "content": format_content,
            "files": files,
            "section_name": section_name,
            "timestamp": datetime.now().isoformat(),
            "full_response": llm_response
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "format": format_name,
            "section_name": section_name,
            "timestamp": datetime.now().isoformat()
        }
```

### 3. **Enhanced Error Handling**
- Uses `@api_calling_error_exponential_backoff` decorator
- Consistent with other tools in the codebase
- Better error reporting and retry logic

### 4. **Cost Tracking Integration**
- Uses `CostTracker` for monitoring API costs
- Consistent cost reporting across all tools
- Task-specific cost tracking with `task_name="generate_diagram"`

## 📊 Benefits of Refactoring

### **1. Consistency**
- ✅ Same interface pattern as `CodeSearchTool`, `PaperSearchTool`, etc.
- ✅ Consistent error handling and retry logic
- ✅ Uniform cost tracking and reporting

### **2. Better Error Handling**
- ✅ Uses `api_calling_error_exponential_backoff` decorator
- ✅ Automatic retry with exponential backoff
- ✅ Proper error propagation

### **3. Enhanced Cost Management**
- ✅ Integrated cost tracking with `CostTracker`
- ✅ Task-specific cost monitoring
- ✅ Consistent cost reporting

### **4. Improved Maintainability**
- ✅ Direct dependency on `utils.llm` instead of `DrawerTool`
- ✅ Cleaner separation of concerns
- ✅ Easier to debug and extend

### **5. Better Performance**
- ✅ Direct LLM calls without intermediate layer
- ✅ Reduced overhead from DrawerTool wrapper
- ✅ More efficient resource usage

## 🎯 Usage Examples

### **Basic Usage (like other tools)**
```python
tool = ImprovedDiagramTool(model="gpt-4o", temperature=0.3)

query = json.dumps({
    "section_name": "Method",
    "section_content": "Data preprocessing -> Model training -> Evaluation",
    "format": "mermaid"
})

result = tool.run(query)
```

### **Direct Method Usage**
```python
tool = ImprovedDiagramTool(model="gpt-4o", temperature=0.3)

result = tool.generate_diagram(
    section_name="Method",
    section_content="Data preprocessing -> Model training -> Evaluation",
    preferred_format="mermaid",
    auto_fallback=True
)
```

### **Batch Generation**
```python
tool = ImprovedDiagramTool(model="gpt-4o", temperature=0.3)

test_cases = [
    {
        "name": "experiment_design",
        "section_name": "Experimental_Setup",
        "section_content": "ResNet-50 training on ImageNet..."
    }
]

results = tool.batch_generate(test_cases, preferred_format="mermaid")
```

## 🔄 Migration from DrawerTool

### **Key Differences:**

| Aspect | DrawerTool | ImprovedDiagramTool |
|--------|------------|-------------------|
| **LLM Integration** | Indirect via DrawerTool | Direct via utils.llm |
| **Error Handling** | Basic retry | Exponential backoff |
| **Cost Tracking** | Limited | Full integration |
| **Interface** | Custom | Consistent with other tools |
| **Performance** | Higher overhead | Lower overhead |
| **Maintainability** | Complex dependencies | Clean dependencies |

### **Advantages of New Approach:**
1. **Consistency** - Same pattern as other tools
2. **Reliability** - Better error handling and retry logic
3. **Efficiency** - Direct LLM calls without wrapper overhead
4. **Maintainability** - Cleaner code structure
5. **Cost Management** - Integrated cost tracking

## 🛠️ Technical Implementation

### **Dependencies:**
```python
from tiny_scientist.utils.llm import create_client, get_response_from_llm
from tiny_scientist.utils.cost_tracker import CostTracker
from tiny_scientist.utils.error_handler import api_calling_error_exponential_backoff
from tiny_scientist.configs import Config
```

### **Initialization:**
```python
def __init__(self, model: str = "gpt-4o", temperature: float = 0.3):
    self.model = model
    self.temperature = temperature
    
    # Initialize LLM client and cost tracker like other tools
    self.client, self.model_name = create_client(model)
    self.cost_tracker = CostTracker()
    
    # Load prompt templates using Config
    self.config = Config()
    self.prompts = self.config.prompt_template.drawer_prompt
```

### **Format-Specific Prompts:**
Each format now includes example syntax in the prompt:
- **Mermaid**: Includes flowchart example
- **DOT**: Includes digraph example  
- **PlantUML**: Includes component diagram example
- **SVG**: Includes basic SVG structure example

## 📈 Expected Impact

### **Performance Improvements:**
- **Reduced Latency**: Direct LLM calls without wrapper overhead
- **Better Success Rate**: Improved error handling and retry logic
- **Cost Efficiency**: Better cost tracking and management

### **Maintainability Improvements:**
- **Consistent Codebase**: Same patterns as other tools
- **Easier Debugging**: Direct LLM integration
- **Better Testing**: Cleaner separation of concerns

### **Reliability Improvements:**
- **Robust Error Handling**: Exponential backoff retry
- **Better Monitoring**: Integrated cost tracking
- **Consistent Behavior**: Same interface as other tools

This refactored approach provides a more robust, maintainable, and consistent diagram generation tool that integrates seamlessly with the existing codebase architecture. 