#!/usr/bin/env python3
"""
Improved Diagram Tool - Multi-format diagram generation with better stability

Supports multiple diagram formats:
- TikZ (LaTeX diagrams)
- XML (structured diagrams)
- SVG (vector graphics)
"""

import json
import os
import sys
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from importlib import resources

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Direct imports to avoid dependency issues
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tiny_scientist", "utils"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tiny_scientist"))

from llm import create_client, get_response_from_llm
from error_handler import api_calling_error_exponential_backoff
from configs import Config


class ImprovedDiagramTool:
    """
    Enhanced diagram generation tool with multiple format support
    Based on tool.py DrawerTool implementation
    """
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", temperature: float = 0.75):
        """
        Initialize the improved diagram tool
        
        Args:
            model: LLM model to use
            temperature: Generation temperature (higher for more creative output)
        """
        self.model = model
        self.temperature = temperature
        
        # Initialize LLM client like DrawerTool
        self.client, self.model_name = create_client(model)
        
        # Load prompt templates using Config like DrawerTool
        self.config = Config()
        self.prompts = self.config.prompt_template.drawer_prompt
        
        # Load few-shot examples like DrawerTool
        def escape_curly_braces(text: str) -> str:
            return re.sub(r"({|})", r"{{\1}}", text)

        def extract_pdf_text_from_resource(package: str, filename: str) -> str:
            try:
                with resources.files(package).joinpath(filename).open("rb") as f:
                    import fitz
                    doc = fitz.open(stream=f.read(), filetype="pdf")
                    extracted = [page.get_text().strip() for page in doc]
                    return "\n\n".join(extracted)
            except Exception as e:
                print(f"⚠️  Could not load {filename}: {e}")
                return ""

        method_sample_raw = extract_pdf_text_from_resource(
            "tiny_scientist.fewshot_sample", "framework.pdf"
        )
        result_sample_raw = extract_pdf_text_from_resource(
            "tiny_scientist.fewshot_sample", "result.pdf"
        )

        method_sample = escape_curly_braces(method_sample_raw)
        result_sample = escape_curly_braces(result_sample_raw)

        self.system_prompts = self.prompts.diagram_system_prompt.format(
            method_sample=method_sample,
            result_sample=result_sample,
        )
        
        # Supported formats
        self.supported_formats = {
            "tikz": {
                "description": "LaTeX TikZ diagrams",
                "advantages": ["LaTeX compatible", "Professional quality", "Academic standard"],
                "file_ext": ".tex"
            },
            "xml": {
                "description": "XML-based diagrams",
                "advantages": ["Structured format", "Easy to parse", "Extensible"],
                "file_ext": ".xml"
            },
            "svg": {
                "description": "Vector graphics",
                "advantages": ["Direct rendering", "Custom styling", "Complex diagrams"],
                "file_ext": ".svg"
            }
        }
        
        # Create output directory
        self.output_dir = Path("diagram_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
    def run(self, query: str) -> Dict[str, Dict[str, str]]:
        """
        Main interface method like DrawerTool
        
        Args:
            query: JSON string with section_name and section_content
            
        Returns:
            Dictionary with diagram results
        """
        try:
            query_dict = json.loads(query)
            section_name = query_dict.get("section_name")
            section_content = query_dict.get("section_content")
            preferred_format = query_dict.get("format", "svg")  # Default to SVG like DrawerTool
        except (json.JSONDecodeError, TypeError, AttributeError):
            raise ValueError(
                "Expected query to be a JSON string with 'section_name' and 'section_content'."
            )
        
        # Generate diagram using DrawerTool-like approach
        diagram = self.draw_diagram(
            section_name=section_name,
            section_content=section_content,
            preferred_format=preferred_format
        )
        
        # Format result like DrawerTool
        results = {}
        if diagram and diagram.get("success", False):
            results["diagram"] = {
                "summary": diagram.get("summary", ""),
                "content": diagram.get("content", ""),
                "format": diagram.get("format", ""),
                "files": diagram.get("files", {})
            }
        
        return results
        
    def draw_diagram(
        self,
        section_name: str,
        section_content: str,
        preferred_format: str = "svg",
        msg_history: Optional[List[Dict[str, Any]]] = None,
        return_msg_history: bool = False,
    ) -> Any:
        """
        Generate diagram using DrawerTool-like approach
        
        Args:
            section_name: Section name (Method, Experimental_Setup, etc.)
            section_content: Content to generate diagram from
            preferred_format: Preferred diagram format
            msg_history: Message history for conversation
            return_msg_history: Whether to return message history
            
        Returns:
            Dictionary containing diagram data and metadata
        """
        print(f"🎨 Generating {preferred_format} diagram for {section_name}...")
        
        # Use DrawerTool-like prompt generation
        section_prompt = self._get_section_prompts(section_name, section_content)
        
        # Generate diagram using DrawerTool-like approach
        diagram, updated_msg_history = self._generate_diagram(
            section_prompt, self.system_prompts, msg_history, preferred_format
        )
        
        return (diagram, updated_msg_history) if return_msg_history else diagram
    
    def _get_section_prompts(self, section_name: str, section_text: str) -> str:
        """
        Get section-specific prompt like DrawerTool
        """
        try:
            section_prompt = self.prompts.section_prompt[section_name].format(
                section_text=section_text
            )
            return section_prompt
        except KeyError:
            # Fallback for unknown sections
            return f"""
You are generating a diagram for the {section_name} section of a scientific paper.

Section content:
{section_text}

Create a clear and professional diagram that visualizes the key concepts and relationships described in this section.
"""
    
    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _generate_diagram(
        self,
        section_prompt: str,
        drawer_system_prompt: str,
        msg_history: Optional[List[Dict[str, Any]]],
        preferred_format: str = "svg"
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generate diagram using DrawerTool-like approach
        """
        # Ensure msg_history is a list
        msg_history = msg_history or []
        
        # Modify system prompt for specific format
        format_specific_prompt = self._create_format_specific_prompt(drawer_system_prompt, preferred_format)
        
        print(f"🔍 Debug: Calling LLM for {preferred_format} format...")
        
        # Generate diagram using utils.llm like DrawerTool
        llm_response, msg_history = get_response_from_llm(
            section_prompt,
            model=self.model_name,
            client=self.client,
            system_message=format_specific_prompt,
            msg_history=msg_history,
            temperature=self.temperature,
            task_name="generate_diagram"
        )
        
        print(f"🔍 Debug: LLM response length: {len(llm_response)}")
        print(f"🔍 Debug: LLM response preview: {llm_response[:200]}...")
        
        # Extract and process diagram content
        diagram = self._extract_diagram(llm_response, preferred_format)
        
        return diagram, msg_history
    
    def _create_format_specific_prompt(self, base_prompt: str, format_name: str) -> str:
        """
        Create format-specific system prompt
        """
        format_instructions = ""
        
        if format_name == "tikz":
            format_instructions = """
IMPORTANT: Generate ONLY LaTeX TikZ code, no additional text or explanations.
The response should be valid TikZ code that can be compiled directly.
Example format:
\\begin{tikzpicture}[node distance=2cm]
    \\node[rectangle, draw, fill=blue!20, minimum width=2cm] (input) {Data Input};
    \\node[rectangle, draw, fill=green!20, minimum width=2cm, right=of input] (process) {Processing};
    \\node[rectangle, draw, fill=red!20, minimum width=2cm, right=of process] (output) {Output};
    
    \\draw[->, thick] (input) -- (process);
    \\draw[->, thick] (process) -- (output);
\\end{tikzpicture}
"""
        elif format_name == "xml":
            format_instructions = """
IMPORTANT: Generate ONLY well-formed XML code, no additional text or explanations.
The response should be valid XML that can be parsed directly.
Example format:
<?xml version="1.0" encoding="UTF-8"?>
<diagram type="flowchart">
    <nodes>
        <node id="input" x="50" y="50" label="Data Input" type="rectangle"/>
        <node id="process" x="200" y="50" label="Processing" type="rectangle"/>
        <node id="output" x="350" y="50" label="Output" type="rectangle"/>
    </nodes>
    <edges>
        <edge from="input" to="process" type="arrow"/>
        <edge from="process" to="output" type="arrow"/>
    </edges>
</diagram>
"""
        elif format_name == "svg":
            format_instructions = """
IMPORTANT: Generate ONLY valid SVG code, no additional text or explanations.
The response should be complete SVG that can be rendered directly.
Example format:
<svg width="600" height="400" viewBox="0 0 600 400">
  <rect x="50" y="50" width="100" height="40" fill="lightblue" stroke="blue" stroke-width="2"/>
  <text x="100" y="75" text-anchor="middle" font-family="Arial" font-size="12">Data Input</text>
  <rect x="200" y="50" width="100" height="40" fill="lightgreen" stroke="green" stroke-width="2"/>
  <text x="250" y="75" text-anchor="middle" font-family="Arial" font-size="12">Processing</text>
  <line x1="150" y1="70" x2="200" y2="70" stroke="black" stroke-width="2" marker-end="url(#arrowhead)"/>
</svg>
"""
        
        return base_prompt + "\n\n" + format_instructions
    
    def _extract_diagram(self, response: str, format_name: str) -> Dict[str, Any]:
        """
        Extract diagram content based on format
        """
        result = {
            "success": False,
            "summary": "",
            "content": "",
            "format": format_name,
            "full_response": response,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            if format_name == "svg":
                # Use DrawerTool-like SVG extraction
                svg_match = re.search(r"<svg.*?</svg>", response, re.DOTALL)
                if svg_match:
                    svg = svg_match.group(0)
                    result["content"] = self._clean_svg(svg)
                    result["summary"] = (
                        re.sub(r"<svg.*?</svg>", "", response, flags=re.DOTALL)
                        .strip()
                        .split("\n")[0]
                    )
                    result["success"] = True
                else:
                    print("❌ Debug: No valid SVG found in response")
                    
            elif format_name == "tikz":
                # Extract TikZ code
                tikz_match = re.search(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", response, re.DOTALL)
                if tikz_match:
                    result["content"] = tikz_match.group(0).strip()
                    result["summary"] = "TikZ diagram generated successfully"
                    result["success"] = True
                else:
                    print("❌ Debug: No valid TikZ code found in response")
                    
            elif format_name == "xml":
                # Extract XML content
                xml_match = re.search(r"<\?xml.*?</diagram>", response, re.DOTALL)
                if xml_match:
                    result["content"] = xml_match.group(0).strip()
                    result["summary"] = "XML diagram generated successfully"
                    result["success"] = True
                else:
                    print("❌ Debug: No valid XML found in response")
            
            if result["success"]:
                # Save files
                files = self._save_diagram_files(result["content"], "test", format_name)
                result["files"] = files
                
        except Exception as e:
            print(f"❌ Debug: Exception in diagram extraction: {e}")
            result["error"] = str(e)
        
        return result
    
    def _clean_svg(self, svg: str) -> str:
        """
        Clean SVG like DrawerTool
        """
        # Strip any outer code block delimiters
        svg = svg.strip()
        svg = re.sub(r"^```(?:svg)?", "", svg)
        svg = re.sub(r"```$", "", svg)

        # Replace problematic ampersands
        svg = svg.replace("&", "&amp;")

        # Ensure no double XML declarations
        svg = re.sub(r"<\?xml.*?\?>", "", svg, count=1)

        # Remove extra whitespace lines
        svg = "\n".join([line for line in svg.splitlines() if line.strip()])

        return svg.strip()
    
    def _save_diagram_files(self, content: str, section_name: str, 
                           format_name: str) -> Dict[str, str]:
        """
        Save diagram files in various formats
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{section_name.lower()}_{format_name}_{timestamp}"
        
        files = {}
        
        # Save original format file
        ext = self.supported_formats[format_name]["file_ext"]
        original_path = self.output_dir / f"{base_filename}{ext}"
        with open(original_path, 'w', encoding='utf-8') as f:
            f.write(content)
        files["original"] = str(original_path)
        
        # Generate additional formats if possible
        if format_name == "tikz":
            # Convert to PDF using LaTeX compilation
            pdf_path = self._convert_tikz_to_pdf(content, base_filename)
            if pdf_path:
                files["pdf"] = pdf_path
        
        elif format_name == "xml":
            # Convert to SVG using XML processing
            svg_path = self._convert_xml_to_svg(content, base_filename)
            if svg_path:
                files["svg"] = svg_path
        
        elif format_name == "svg":
            # SVG is already in correct format
            files["svg"] = str(original_path)
        
        return files
    
    def _convert_tikz_to_pdf(self, tikz_content: str, base_filename: str) -> Optional[str]:
        """Convert TikZ to PDF using LaTeX compilation"""
        try:
            # Check if pdflatex is available
            result = subprocess.run(['which', 'pdflatex'], capture_output=True, text=True)
            if result.returncode != 0:
                print("⚠️  pdflatex not found. Install with: brew install basictex (macOS) or apt-get install texlive-full (Ubuntu)")
                return None
            
            pdf_path = self.output_dir / f"{base_filename}.pdf"
            
            # Create temporary LaTeX file
            latex_content = f"""\\documentclass{{article}}
\\usepackage{{tikz}}
\\usepackage{{positioning}}
\\usepackage[margin=1in]{{geometry}}

\\begin{{document}}
\\begin{{center}}
{tikz_content}
\\end{{center}}
\\end{{document}}"""
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False) as tmp:
                tmp.write(latex_content)
                tmp_path = tmp.name
            
            # Compile using pdflatex
            result = subprocess.run([
                'pdflatex', '-interaction=nonstopmode', '-output-directory', str(self.output_dir), tmp_path
            ], capture_output=True, text=True)
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            if result.returncode == 0 and pdf_path.exists():
                return str(pdf_path)
            else:
                print(f"⚠️  TikZ to PDF conversion failed: {result.stderr}")
            
        except Exception as e:
            print(f"⚠️  TikZ to PDF conversion failed: {e}")
        
        return None
    
    def _convert_xml_to_svg(self, xml_content: str, base_filename: str) -> Optional[str]:
        """Convert XML diagram to SVG using custom processing"""
        try:
            import xml.etree.ElementTree as ET
            
            # Parse XML content
            root = ET.fromstring(xml_content)
            
            # Create SVG content based on XML structure
            svg_content = self._xml_to_svg_converter(root)
            
            if svg_content:
                svg_path = self.output_dir / f"{base_filename}.svg"
                with open(svg_path, 'w', encoding='utf-8') as f:
                    f.write(svg_content)
                return str(svg_path)
            else:
                print("⚠️  XML to SVG conversion failed: Could not parse XML structure")
            
        except Exception as e:
            print(f"⚠️  XML to SVG conversion failed: {e}")
        
        return None
    
    def _xml_to_svg_converter(self, root) -> Optional[str]:
        """Convert XML diagram structure to SVG"""
        try:
            # Extract diagram type
            diagram_type = root.get('type', 'flowchart')
            
            # Initialize SVG
            svg_width = 600
            svg_height = 400
            svg_content = f'''<svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">
  <defs>
    <style>
      .node {{ fill: #e3f2fd; stroke: #1976d2; stroke-width: 2; }}
      .text {{ font-family: Arial; font-size: 12px; text-anchor: middle; }}
      .arrow {{ stroke: #424242; stroke-width: 2; marker-end: url(#arrowhead); }}
    </style>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#424242" />
    </marker>
  </defs>'''
            
            # Process nodes
            nodes = root.find('nodes')
            if nodes is not None:
                for node in nodes.findall('node'):
                    node_id = node.get('id', '')
                    x = int(node.get('x', 50))
                    y = int(node.get('y', 50))
                    label = node.get('label', node_id)
                    node_type = node.get('type', 'rectangle')
                    
                    if node_type == 'rectangle':
                        svg_content += f'''
  <rect x="{x}" y="{y}" width="100" height="40" class="node" rx="5"/>
  <text x="{x+50}" y="{y+25}" class="text">{label}</text>'''
                    elif node_type == 'circle':
                        svg_content += f'''
  <circle cx="{x+25}" cy="{y+20}" r="20" class="node"/>
  <text x="{x+25}" y="{y+25}" class="text">{label}</text>'''
            
            # Process edges
            edges = root.find('edges')
            if edges is not None:
                for edge in edges.findall('edge'):
                    from_node = edge.get('from', '')
                    to_node = edge.get('to', '')
                    edge_type = edge.get('type', 'arrow')
                    
                    # Simple edge drawing (you can enhance this)
                    svg_content += f'''
  <line x1="150" y1="70" x2="200" y2="70" class="arrow"/>'''
            
            svg_content += '''
</svg>'''
            
            return svg_content
            
        except Exception as e:
            print(f"⚠️  XML to SVG conversion error: {e}")
            return None


def main():
    """Test the improved diagram tool with better test cases"""
    
    print("🎨 Improved Diagram Tool Test")
    print("=" * 50)
    
    # Initialize tool
    tool = ImprovedDiagramTool(model="claude-3-5-sonnet-20241022", temperature=0.75)
    
    # Better test cases based on real academic content
    test_cases = [
        {
            "name": "deep_learning_architecture",
            "section_name": "Method",
            "section_content": """
            We propose a novel deep learning architecture that combines convolutional neural networks (CNNs) 
            with attention mechanisms. Our model consists of three main components: 1) A feature extraction 
            module using ResNet-50 backbone, 2) A multi-head attention layer that captures spatial dependencies, 
            and 3) A classification head with softmax activation. The model is trained end-to-end using 
            cross-entropy loss and Adam optimizer with learning rate 0.001.
            """
        },
        {
            "name": "experimental_evaluation",
            "section_name": "Experimental_Setup",
            "section_content": """
            We evaluate our method on three benchmark datasets: ImageNet, CIFAR-10, and Places365. 
            For each dataset, we split the data into training (80%), validation (10%), and test (10%) sets. 
            We compare against five baseline methods: ResNet-50, VGG-16, DenseNet-121, EfficientNet-B0, 
            and Vision Transformer. All experiments are conducted on NVIDIA V100 GPUs with 32GB memory. 
            Training runs for 100 epochs with batch size 64, and we report top-1 accuracy, top-5 accuracy, 
            and training time as evaluation metrics.
            """
        },
        {
            "name": "ablation_study",
            "section_name": "Results",
            "section_content": """
            Our ablation study demonstrates the effectiveness of each component. The baseline ResNet-50 
            achieves 76.2% accuracy on ImageNet. Adding attention mechanisms improves performance to 78.5%, 
            while the full model with all components reaches 81.3% accuracy. We also analyze the impact 
            of different attention heads: 4 heads (77.8%), 8 heads (79.1%), and 16 heads (81.3%). 
            The model converges faster with more attention heads, requiring 45 epochs vs 67 epochs for 
            the baseline to reach 75% accuracy.
            """
        }
    ]
    
    # Test different formats
    formats_to_test = ["svg", "tikz", "xml"]
    
    for format_name in formats_to_test:
        print(f"\n🔄 Testing {format_name.upper()} format...")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📊 Test Case {i}/{len(test_cases)}: {test_case['name']}")
            
            result = tool.draw_diagram(
                section_name=test_case["section_name"],
                section_content=test_case["section_content"],
                preferred_format=format_name
            )
            
            print(f"✅ Success: {result['success']}")
            if result['success']:
                print(f"📁 Files: {result.get('files', {})}")
                print(f"📝 Summary: {result.get('summary', '')}")
                print(f"🎯 Format: {result.get('format', '')}")
            else:
                print(f"❌ Error: {result.get('error', '')}")
            
            # Add delay to avoid rate limiting
            if i < len(test_cases):
                import time
                time.sleep(2)
    
    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    main() 