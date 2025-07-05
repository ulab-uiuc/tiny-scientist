# Diagram Tool Troubleshooting Guide

## 🔧 Common Issues and Solutions

### **Issue: Conversion Tools Not Found**

If you see errors like:
```
⚠️  mermaid-cli not found. Install with: npm install -g @mermaid-js/mermaid-cli
⚠️  Graphviz not found. Install with: brew install graphviz (macOS) or apt-get install graphviz (Ubuntu)
⚠️  PlantUML not found. Install with: brew install plantuml (macOS) or apt-get install plantuml (Ubuntu)
```

### **Solution 1: Automatic Installation (Recommended)**

Run the installation script:

```bash
# Make the script executable
chmod +x comparison_test/install_diagram_tools.sh

# Run the installation script
./comparison_test/install_diagram_tools.sh
```

### **Solution 2: Manual Installation**

#### **For macOS:**
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Graphviz
brew install graphviz

# Install PlantUML
brew install plantuml

# Install Node.js (if not already installed)
brew install node

# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli
```

#### **For Ubuntu/Debian:**
```bash
# Update package list
sudo apt-get update

# Install Graphviz
sudo apt-get install -y graphviz

# Install PlantUML
sudo apt-get install -y plantuml

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install mermaid-cli
sudo npm install -g @mermaid-js/mermaid-cli
```

#### **For CentOS/RHEL:**
```bash
# Install Graphviz
sudo yum install -y graphviz

# Install PlantUML
sudo yum install -y plantuml

# Install Node.js
curl -fsSL https://rpm.nodesource.com/setup_lts.x | sudo bash -
sudo yum install -y nodejs

# Install mermaid-cli
sudo npm install -g @mermaid-js/mermaid-cli
```

### **Solution 3: Verify Installation**

After installation, verify that all tools are available:

```bash
# Check Graphviz
which dot
dot -V

# Check PlantUML
which plantuml
plantuml -version

# Check mermaid-cli
which mmdc
mmdc --version
```

## 🎯 Understanding the Error Messages

### **What the Errors Mean:**

1. **`No such file or directory: 'mmdc'`**
   - The mermaid-cli tool is not installed
   - This tool converts Mermaid diagrams to SVG

2. **`No such file or directory: 'dot'`**
   - The Graphviz tool is not installed
   - This tool converts DOT diagrams to SVG

3. **`No such file or directory: 'plantuml'`**
   - The PlantUML tool is not installed
   - This tool converts PlantUML diagrams to SVG

### **Impact on Functionality:**

- ✅ **Diagram Generation**: Still works perfectly
- ✅ **Format Support**: All formats (Mermaid, DOT, PlantUML, SVG) work
- ⚠️ **SVG Conversion**: Only affects automatic conversion to SVG
- ✅ **File Output**: Original format files are still saved

## 🔄 Workarounds

### **Option 1: Use Without Conversion Tools**

The tool works perfectly without conversion tools. You'll get:
- Original format files (`.mmd`, `.dot`, `.puml`, `.svg`)
- All diagram generation functionality
- Just no automatic SVG conversion

### **Option 2: Manual Conversion**

If you need SVG files, you can convert manually:

#### **Mermaid to SVG:**
```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Convert
mmdc -i input.mmd -o output.svg
```

#### **DOT to SVG:**
```bash
# Install Graphviz
brew install graphviz  # macOS
sudo apt-get install graphviz  # Ubuntu

# Convert
dot -Tsvg input.dot -o output.svg
```

#### **PlantUML to SVG:**
```bash
# Install PlantUML
brew install plantuml  # macOS
sudo apt-get install plantuml  # Ubuntu

# Convert
plantuml -tsvg input.puml
```

## 📊 Current Status

Based on your test output:
- ✅ **Mermaid Generation**: 3/3 successful (100%)
- ✅ **DOT Generation**: 3/3 successful (100%)
- ✅ **PlantUML Generation**: 3/3 successful (100%)
- ⚠️ **SVG Conversion**: 0/9 successful (0%) - due to missing tools

## 🎯 Recommendations

### **For Development/Testing:**
- The tool works fine without conversion tools
- Focus on diagram generation quality
- Install conversion tools only when needed

### **For Production:**
- Install all conversion tools for full functionality
- Use the installation script for consistency
- Verify all tools are working before deployment

### **For Academic Use:**
- Mermaid format is most stable and widely supported
- Original format files can be used directly in many platforms
- SVG conversion is optional for most use cases

## 🔧 Advanced Troubleshooting

### **If Installation Fails:**

1. **Check Node.js Installation:**
   ```bash
   node --version
   npm --version
   ```

2. **Check Package Manager:**
   ```bash
   # macOS
   brew --version
   
   # Ubuntu
   apt --version
   ```

3. **Manual Download:**
   - Graphviz: https://graphviz.org/download/
   - PlantUML: https://plantuml.com/download
   - mermaid-cli: `npm install -g @mermaid-js/mermaid-cli`

### **If Tools Install But Don't Work:**

1. **Check PATH:**
   ```bash
   echo $PATH
   which mmdc
   which dot
   which plantuml
   ```

2. **Restart Terminal:**
   - Close and reopen terminal after installation
   - Or run: `source ~/.bashrc` or `source ~/.zshrc`

3. **Check Permissions:**
   ```bash
   ls -la $(which mmdc)
   ls -la $(which dot)
   ls -la $(which plantuml)
   ```

## 📈 Success Metrics

Your current success rate is excellent:
- **Diagram Generation**: 100% success rate
- **Format Support**: All formats working
- **Error Handling**: Graceful degradation when tools missing

The missing conversion tools don't affect the core functionality - the tool is working perfectly for diagram generation! 