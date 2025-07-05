# DrawIO Diagram Generator

A tool to generate DrawIO XML code from text descriptions using LLM.

## Features

- Convert text descriptions to DrawIO XML code
- Support multiple diagram types (flowchart, architecture, workflow)
- Generate professional academic/technical diagrams
- Automatic validation and error handling

## Usage

### Basic Usage

```python
from drawio_diagram_generator import DrawIODiagramGenerator

# Initialize generator
generator = DrawIODiagramGenerator(model="claude-3-5-sonnet-20241022")

# Generate diagram from description
result = generator.generate_drawio_xml(
    description="Create a simple flowchart with Start -> Process -> End",
    diagram_type="flowchart"
)

if result['success']:
    print(f"Diagram saved to: {result['file_path']}")
else:
    print(f"Error: {result['error']}")
```

### Example Descriptions

#### 1. Machine Learning Pipeline
```
Create a machine learning pipeline diagram with:
1. Data Collection
2. Data Preprocessing  
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Model Deployment

Use linear flow from top to bottom with clear connections.
```

#### 2. Neural Network Architecture
```
Create a neural network diagram showing:
- Input Layer (3 nodes)
- Hidden Layer 1 (5 nodes) 
- Hidden Layer 2 (4 nodes)
- Output Layer (2 nodes)

Show connections with arrows and use different colors for each layer.
```

#### 3. Research Workflow
```
Create a research workflow with:
1. Hypothesis Formulation
2. Literature Review
3. Experimental Design
4. Data Collection
5. Data Analysis
6. Results Interpretation
7. Paper Writing
8. Publication

Show iterative nature with feedback loops.
```

## Output

The tool generates:
- DrawIO XML files (.xml)
- Saved in `diagram_outputs/` directory
- Ready to import into draw.io or diagrams.net

## File Structure

```
comparison_test/
├── drawio_diagram_generator.py    # Main generator
├── diagram_outputs/               # Generated files
│   ├── drawio_flowchart_*.xml
│   ├── drawio_architecture_*.xml
│   └── simple_example_*.xml
└── README_DrawIO.md              # This file
```

## Running Tests

```bash
cd comparison_test
python drawio_diagram_generator.py
```

This will:
1. Generate a simple example diagram
2. Test 3 different diagram types
3. Save all files to `diagram_outputs/`

## Tips for Better Results

1. **Be Specific**: Include details about shapes, colors, and connections
2. **Use Clear Language**: Describe the flow and relationships clearly
3. **Specify Layout**: Mention if you want linear, hierarchical, or network layout
4. **Include Context**: Mention if it's for academic, technical, or business use

## Example Output

The tool generates DrawIO XML that can be imported into:
- draw.io (diagrams.net)
- Any tool that supports DrawIO format
- Can be converted to PNG, SVG, PDF

## Troubleshooting

- **No XML Generated**: Check if the description is clear enough
- **Invalid XML**: The tool includes validation, but check the description
- **Rate Limiting**: Add delays between multiple generations
- **File Not Found**: Check the `diagram_outputs/` directory exists

## Integration

You can integrate this into your workflow:

```python
# Generate diagram from paper section
section_content = "Our method consists of three main steps..."
result = generator.generate_drawio_xml(
    description=section_content,
    diagram_type="methodology"
)
``` 