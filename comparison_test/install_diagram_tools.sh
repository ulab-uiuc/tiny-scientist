#!/bin/bash

# Install script for diagram conversion tools
# This script installs the necessary tools for converting diagrams to SVG

echo "🎨 Installing Diagram Conversion Tools"
echo "======================================"

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "📱 Detected macOS"
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
    echo "🔄 Installing Graphviz..."
    brew install graphviz
    
    echo "🔄 Installing PlantUML..."
    brew install plantuml
    
    echo "🔄 Installing Node.js (if not already installed)..."
    brew install node
    
    echo "🔄 Installing mermaid-cli..."
    npm install -g @mermaid-js/mermaid-cli
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "🐧 Detected Linux"
    
    # Check if apt is available
    if command -v apt &> /dev/null; then
        echo "🔄 Installing Graphviz..."
        sudo apt-get update
        sudo apt-get install -y graphviz
        
        echo "🔄 Installing PlantUML..."
        sudo apt-get install -y plantuml
        
        echo "🔄 Installing Node.js (if not already installed)..."
        curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
        sudo apt-get install -y nodejs
        
        echo "🔄 Installing mermaid-cli..."
        sudo npm install -g @mermaid-js/mermaid-cli
        
    elif command -v yum &> /dev/null; then
        echo "🔄 Installing Graphviz..."
        sudo yum install -y graphviz
        
        echo "🔄 Installing PlantUML..."
        sudo yum install -y plantuml
        
        echo "🔄 Installing Node.js (if not already installed)..."
        curl -fsSL https://rpm.nodesource.com/setup_lts.x | sudo bash -
        sudo yum install -y nodejs
        
        echo "🔄 Installing mermaid-cli..."
        sudo npm install -g @mermaid-js/mermaid-cli
        
    else
        echo "❌ Unsupported package manager. Please install manually:"
        echo "   - Graphviz: https://graphviz.org/download/"
        echo "   - PlantUML: https://plantuml.com/download"
        echo "   - mermaid-cli: npm install -g @mermaid-js/mermaid-cli"
        exit 1
    fi
    
else
    echo "❌ Unsupported OS: $OSTYPE"
    echo "Please install manually:"
    echo "   - Graphviz: https://graphviz.org/download/"
    echo "   - PlantUML: https://plantuml.com/download"
    echo "   - mermaid-cli: npm install -g @mermaid-js/mermaid-cli"
    exit 1
fi

echo ""
echo "✅ Installation completed!"
echo ""
echo "🔍 Verifying installations..."

# Check if tools are available
echo "Checking Graphviz..."
if command -v dot &> /dev/null; then
    echo "✅ Graphviz (dot) is available"
else
    echo "❌ Graphviz (dot) not found"
fi

echo "Checking PlantUML..."
if command -v plantuml &> /dev/null; then
    echo "✅ PlantUML is available"
else
    echo "❌ PlantUML not found"
fi

echo "Checking mermaid-cli..."
if command -v mmdc &> /dev/null; then
    echo "✅ mermaid-cli (mmdc) is available"
else
    echo "❌ mermaid-cli (mmdc) not found"
fi

echo ""
echo "🎉 Setup complete! You can now run the improved diagram tool."
echo ""
echo "Usage:"
echo "  python comparison_test/improved_diagram_tool.py"
echo "  python comparison_test/test_improved_tool_direct.py" 