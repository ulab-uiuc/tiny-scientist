import json
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import sys

from tiny_scientist.utils.llm import create_client, get_response_from_llm
from tiny_scientist.utils.error_handler import api_calling_error_exponential_backoff
from tiny_scientist.configs import Config

@dataclass
class DiagramNode:
    id: str
    label: str
    x: float
    y: float
    width: float
    height: float
    type: str = "box"  # box, circle, diamond

@dataclass
class DiagramEdge:
    source: str
    target: str
    label: str = ""
    type: str = "arrow"  # arrow, line, dotted

class PaperDiagramGenerator:
    def __init__(self, model: str = "claude-3-7-sonnet-20250219"):
        """Initialize the diagram generator with Claude's reasoning model."""
        self.client, self.model_name = create_client(model)
        self.temperature = 0.3
        
    def analyze_text(self, text: str) -> Tuple[List[DiagramNode], List[DiagramEdge]]:
        """Use LLM to analyze text and extract diagram structure."""
        
        system_message = """You are an expert at analyzing academic text and extracting logical structures for diagram generation. 
        Your task is to identify key concepts, entities, and their relationships to create a clear diagram."""
        
        analysis_prompt = f"""Analyze the following text and extract the key components for a diagram:

Text: {text}

Please provide a structured response in the following JSON format:
{{
    "nodes": [
        {{
            "id": "unique_id",
            "label": "Node Label",
            "type": "box|circle|diamond",
            "importance": 1-5
        }}
    ],
    "edges": [
        {{
            "source": "source_id",
            "target": "target_id",
            "label": "relationship label",
            "type": "arrow|line|dotted"
        }}
    ],
    "layout_hint": "hierarchical|circular|network"
}}

Guidelines:
1. Extract main concepts as nodes
2. Identify relationships between concepts as edges
3. Use importance levels to determine node sizes
4. Choose appropriate node types (box for processes, circle for states, diamond for decisions)
5. Keep labels concise but descriptive
6. Ensure all relationships are meaningful and clear"""

        llm_response, msg_history = get_response_from_llm(
            analysis_prompt,
            model=self.model_name,
            client=self.client,
            system_message=system_message,
            msg_history=[],
            temperature=self.temperature,
            task_name="analyze_diagram_structure"
        )
        
        # Parse the JSON response
        try:
            diagram_data = json.loads(llm_response)
        except json.JSONDecodeError:
            # Extract JSON from the response if it's wrapped in text
            import re
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                diagram_data = json.loads(json_match.group())
            else:
                raise ValueError("Could not parse LLM response as JSON")
        
        # Calculate layout
        nodes, edges = self._calculate_layout(diagram_data)
        return nodes, edges
    
    def _calculate_layout(self, diagram_data: Dict) -> Tuple[List[DiagramNode], List[DiagramEdge]]:
        """Calculate optimal positions for nodes to avoid overlaps."""
        nodes = []
        edges = []
        
        # Parse nodes
        node_data = diagram_data.get('nodes', [])
        layout_hint = diagram_data.get('layout_hint', 'hierarchical')
        
        # Calculate positions based on layout hint
        if layout_hint == 'hierarchical':
            positions = self._hierarchical_layout(node_data, diagram_data.get('edges', []))
        elif layout_hint == 'circular':
            positions = self._circular_layout(node_data)
        else:
            positions = self._network_layout(node_data)
        
        # Create DiagramNode objects
        for i, node in enumerate(node_data):
            importance = node.get('importance', 3)
            width = 120 + (importance - 3) * 20
            height = 60 + (importance - 3) * 10
            
            nodes.append(DiagramNode(
                id=node['id'],
                label=node['label'],
                x=positions[i][0],
                y=positions[i][1],
                width=width,
                height=height,
                type=node.get('type', 'box')
            ))
        
        # Create DiagramEdge objects
        for edge in diagram_data.get('edges', []):
            edges.append(DiagramEdge(
                source=edge['source'],
                target=edge['target'],
                label=edge.get('label', ''),
                type=edge.get('type', 'arrow')
            ))
        
        return nodes, edges
    
    def _hierarchical_layout(self, nodes: List[Dict], edges: List[Dict]) -> List[Tuple[float, float]]:
        """Calculate hierarchical layout positions."""
        # Build adjacency list
        graph = {node['id']: [] for node in nodes}
        for edge in edges:
            if edge['source'] in graph:
                graph[edge['source']].append(edge['target'])
        
        # Find levels using BFS
        levels = {}
        visited = set()
        
        # Find root nodes (no incoming edges)
        targets = {edge['target'] for edge in edges}
        roots = [node['id'] for node in nodes if node['id'] not in targets]
        
        if not roots:
            roots = [nodes[0]['id']]
        
        # BFS to assign levels
        from collections import deque
        queue = deque([(root, 0) for root in roots])
        
        while queue:
            node_id, level = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            levels[node_id] = level
            
            for child in graph.get(node_id, []):
                if child not in visited:
                    queue.append((child, level + 1))
        
        # Calculate positions
        positions = []
        level_counts = {}
        for node in nodes:
            level = levels.get(node['id'], 0)
            level_counts[level] = level_counts.get(level, 0) + 1
        
        level_indices = {level: 0 for level in level_counts}
        
        for node in nodes:
            level = levels.get(node['id'], 0)
            index = level_indices[level]
            count = level_counts[level]
            
            x = 200 + (index - (count - 1) / 2) * 200
            y = 100 + level * 150
            
            positions.append((x, y))
            level_indices[level] += 1
        
        return positions
    
    def _circular_layout(self, nodes: List[Dict]) -> List[Tuple[float, float]]:
        """Calculate circular layout positions."""
        import math
        n = len(nodes)
        radius = max(200, n * 30)
        center_x, center_y = 400, 300
        
        positions = []
        for i in range(n):
            angle = 2 * math.pi * i / n
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            positions.append((x, y))
        
        return positions
    
    def _network_layout(self, nodes: List[Dict]) -> List[Tuple[float, float]]:
        """Calculate network layout positions using force-directed approach."""
        import random
        import math
        
        n = len(nodes)
        positions = [(random.uniform(100, 700), random.uniform(100, 500)) for _ in range(n)]
        
        # Simple force-directed layout
        for _ in range(50):
            forces = [(0, 0) for _ in range(n)]
            
            # Repulsion between nodes
            for i in range(n):
                for j in range(i + 1, n):
                    dx = positions[j][0] - positions[i][0]
                    dy = positions[j][1] - positions[i][1]
                    dist = math.sqrt(dx**2 + dy**2)
                    if dist > 0:
                        force = 10000 / (dist ** 2)
                        fx = force * dx / dist
                        fy = force * dy / dist
                        forces[i] = (forces[i][0] - fx, forces[i][1] - fy)
                        forces[j] = (forces[j][0] + fx, forces[j][1] + fy)
            
            # Apply forces
            for i in range(n):
                positions[i] = (
                    max(50, min(750, positions[i][0] + forces[i][0] * 0.01)),
                    max(50, min(550, positions[i][1] + forces[i][1] * 0.01))
                )
        
        return positions
    
    def generate_svg(self, nodes: List[DiagramNode], edges: List[DiagramEdge]) -> str:
        """Generate SVG diagram."""
        svg_width = 800
        svg_height = 600
        
        svg_content = f'''<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#333" />
        </marker>
        <marker id="arrowhead-dotted" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
        </marker>
    </defs>
    
    <style>
        .node-box {{ fill: "#e3f2fd"; stroke: "#1976d2"; stroke-width: 2; }}
        .node-circle {{ fill: "#f3e5f5"; stroke: "#7b1fa2"; stroke-width: 2; }}
        .node-diamond {{ fill: "#e8f5e9"; stroke: "#388e3c"; stroke-width: 2; }}
        .node-text {{ font-family: Arial, sans-serif; font-size: 14px; text-anchor: middle; }}
        .edge-line {{ stroke: "#333"; stroke-width: 2; fill: none; }}
        .edge-dotted {{ stroke: "#666"; stroke-width: 2; stroke-dasharray: 5,5; fill: none; }}
        .edge-label {{ font-family: Arial, sans-serif; font-size: 12px; fill: "#666"; }}
    </style>
'''
        
        # Draw edges first (so they appear behind nodes)
        for edge in edges:
            source_node = next((n for n in nodes if n.id == edge.source), None)
            target_node = next((n for n in nodes if n.id == edge.target), None)
            
            if source_node and target_node:
                # Calculate edge path
                sx, sy = source_node.x, source_node.y
                tx, ty = target_node.x, target_node.y
                
                # Simple straight line for now (can be enhanced with curves)
                line_class = "edge-dotted" if edge.type == "dotted" else "edge-line"
                marker = "url(#arrowhead-dotted)" if edge.type == "dotted" else "url(#arrowhead)"
                
                svg_content += f'''    <line x1="{sx}" y1="{sy}" x2="{tx}" y2="{ty}" 
        class="{line_class}" marker-end="{marker}" />
'''
                
                # Add edge label if exists
                if edge.label:
                    mx, my = (sx + tx) / 2, (sy + ty) / 2
                    svg_content += f'''    <text x="{mx}" y="{my - 5}" class="edge-label">{edge.label}</text>
'''
        
        # Draw nodes
        for node in nodes:
            if node.type == "box":
                svg_content += f'''    <rect x="{node.x - node.width/2}" y="{node.y - node.height/2}" 
        width="{node.width}" height="{node.height}" class="node-box" rx="5" />
'''
            elif node.type == "circle":
                radius = min(node.width, node.height) / 2
                svg_content += f'''    <circle cx="{node.x}" cy="{node.y}" r="{radius}" class="node-circle" />
'''
            elif node.type == "diamond":
                # Diamond shape
                points = f"{node.x},{node.y - node.height/2} {node.x + node.width/2},{node.y} {node.x},{node.y + node.height/2} {node.x - node.width/2},{node.y}"
                svg_content += f'''    <polygon points="{points}" class="node-diamond" />
'''
            
            # Add text label
            svg_content += f'''    <text x="{node.x}" y="{node.y + 5}" class="node-text">{node.label}</text>
'''
        
        svg_content += '</svg>'
        return svg_content
    
    def generate_tikz(self, nodes: List[DiagramNode], edges: List[DiagramEdge]) -> str:
        """Generate TikZ diagram code."""
        tikz_content = '''\\documentclass{standalone}
\\usepackage{tikz}
\\usetikzlibrary{shapes.geometric, arrows.meta, positioning}

\\tikzstyle{box} = [rectangle, rounded corners, minimum width=3cm, minimum height=1cm,
                   text centered, draw=blue!70, fill=blue!20]
\\tikzstyle{circle} = [circle, minimum width=2cm, text centered, draw=purple!70, fill=purple!20]
\\tikzstyle{diamond} = [diamond, minimum width=2.5cm, minimum height=1.5cm, text centered,
                        draw=green!70, fill=green!20]
\\tikzstyle{arrow} = [thick,->,>=stealth]
\\tikzstyle{line} = [thick]
\\tikzstyle{dotted} = [thick, dashed]

\\begin{document}
\\begin{tikzpicture}[node distance=3cm]
'''
        
        # Scale positions for TikZ
        scale = 0.02
        
        # Add nodes
        for node in nodes:
            x, y = node.x * scale, -node.y * scale  # Negative y for correct orientation
            style = node.type
            tikz_content += f'    \\node ({node.id}) at ({x:.2f},{y:.2f}) [{style}] {{{node.label}}};\n'
        
        tikz_content += '\n'
        
        # Add edges
        for edge in edges:
            style = edge.type if edge.type in ['arrow', 'line', 'dotted'] else 'arrow'
            label = f'node[above] {{{edge.label}}}' if edge.label else ''
            tikz_content += f'    \\draw [{style}] ({edge.source}) -- {label} ({edge.target});\n'
        
        tikz_content += '''\\end{tikzpicture}
\\end{document}'''
        
        return tikz_content
    
    def generate_xml(self, nodes: List[DiagramNode], edges: List[DiagramEdge]) -> str:
        """Generate XML representation of the diagram."""
        root = ET.Element('diagram')
        root.set('version', '1.0')
        root.set('generated', datetime.now().isoformat())
        
        # Add metadata
        metadata = ET.SubElement(root, 'metadata')
        ET.SubElement(metadata, 'generator').text = 'PaperDiagramGenerator'
        ET.SubElement(metadata, 'node_count').text = str(len(nodes))
        ET.SubElement(metadata, 'edge_count').text = str(len(edges))
        
        # Add nodes
        nodes_element = ET.SubElement(root, 'nodes')
        for node in nodes:
            node_elem = ET.SubElement(nodes_element, 'node')
            node_elem.set('id', node.id)
            node_elem.set('type', node.type)
            ET.SubElement(node_elem, 'label').text = node.label
            position = ET.SubElement(node_elem, 'position')
            position.set('x', str(node.x))
            position.set('y', str(node.y))
            size = ET.SubElement(node_elem, 'size')
            size.set('width', str(node.width))
            size.set('height', str(node.height))
        
        # Add edges
        edges_element = ET.SubElement(root, 'edges')
        for edge in edges:
            edge_elem = ET.SubElement(edges_element, 'edge')
            edge_elem.set('source', edge.source)
            edge_elem.set('target', edge.target)
            edge_elem.set('type', edge.type)
            if edge.label:
                ET.SubElement(edge_elem, 'label').text = edge.label
        
        # Pretty print XML
        xml_str = ET.tostring(root, encoding='unicode')
        dom = minidom.parseString(xml_str)
        return dom.toprettyxml(indent='    ')
    
    def generate_diagram(self, text: str, output_dir: str = 'diagrams') -> Dict[str, str]:
        """Generate all three diagram formats from input text."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze text and extract diagram structure
        print("Analyzing text with LLM...")
        nodes, edges = self.analyze_text(text)
        
        # Generate diagrams in all formats
        print("Generating SVG diagram...")
        svg_content = self.generate_svg(nodes, edges)
        
        print("Generating TikZ diagram...")
        tikz_content = self.generate_tikz(nodes, edges)
        
        print("Generating XML diagram...")
        xml_content = self.generate_xml(nodes, edges)
        
        # Save files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        svg_file = os.path.join(output_dir, f'diagram_{timestamp}.svg')
        with open(svg_file, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        tikz_file = os.path.join(output_dir, f'diagram_{timestamp}.tex')
        with open(tikz_file, 'w', encoding='utf-8') as f:
            f.write(tikz_content)
        
        xml_file = os.path.join(output_dir, f'diagram_{timestamp}.xml')
        with open(xml_file, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        print(f"\nDiagrams saved:")
        print(f"  SVG: {svg_file}")
        print(f"  TikZ: {tikz_file}")
        print(f"  XML: {xml_file}")
        
        return {
            'svg': svg_file,
            'tikz': tikz_file,
            'xml': xml_file,
            'nodes': len(nodes),
            'edges': len(edges)
        }

# Example usage
if __name__ == "__main__":
    # Example text about a machine learning pipeline
    sample_text = """
    The machine learning pipeline begins with Data Collection from various sources. 
    The collected data then goes through a Data Preprocessing stage where it is cleaned and normalized. 
    After preprocessing, the data is split into Training Data and Testing Data. 
    The Training Data is used to train the Machine Learning Model through an iterative Training Process. 
    During training, the model's performance is validated using a Validation Set. 
    Once trained, the model is evaluated on the Testing Data to measure its performance. 
    If the performance is satisfactory, the model proceeds to Deployment. 
    Otherwise, it goes back to Hyperparameter Tuning before retraining.
    """
    
    # Initialize generator
    generator = PaperDiagramGenerator(model="claude-3-7-sonnet-20250219")
    
    # Generate diagrams
    result = generator.generate_diagram(sample_text)
    
    print(f"\nGenerated diagram with {result['nodes']} nodes and {result['edges']} edges")