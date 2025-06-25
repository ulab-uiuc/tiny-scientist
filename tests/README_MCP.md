# Tiny Scientist FastMCP Server

A comprehensive Model Context Protocol (MCP) server that exposes research tools from the tiny-scientist project, enabling AI assistants to search academic papers, explore GitHub repositories, and generate research-focused prompts.

## Overview

This MCP server transforms the research capabilities of tiny-scientist into a standardized protocol that can be used by various AI clients like Claude Desktop, Cursor, and other MCP-compatible applications. It provides a bridge between AI assistants and essential research tools.

### Key Features

- **Academic Paper Search**: Query Semantic Scholar and OpenAlex for research papers
- **GitHub Code Discovery**: Search repositories and code implementations
- **Research Resources**: Access configuration settings and trending repositories
- **Intelligent Prompts**: Generate research-focused prompts for various workflows
- **Async Support**: Full asynchronous operation with progress reporting and logging

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   AI Client     │    │   FastMCP       │    │   External      │
│  (Claude, etc.) │◄──►│    Server       │◄──►│   APIs          │
│                 │    │                 │    │  (GitHub,       │
│                 │    │                 │    │   S2, OpenAlex) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Installation

### Prerequisites

1. **Python 3.10+**
2. **FastMCP package**:
   ```bash
   pip install fastmcp
   # or with poetry
   poetry add fastmcp
   ```

### Optional Dependencies

For enhanced functionality, install these packages:

```bash
# For OpenAlex paper search
pip install pyalex

# For advanced text processing (GitHub query formatting)
pip install spacy
python -m spacy download en_core_web_sm
```

### Configuration

Create a `config.toml` file in the `tiny_scientist/` directory:

```toml
[core]
# Semantic Scholar API key (optional, increases rate limits)
s2_api_key = "your_semantic_scholar_api_key"

# GitHub token (optional, increases rate limits)
github_token = "your_github_token"

# Search engine preference
engine = "semanticscholar"  # or "openalex"

# OpenAlex email (required for OpenAlex API)
# Set as environment variable: OPENALEX_MAIL_ADDRESS
```

## Available Tools

### Paper Search Tools

#### `search_papers`
Search for academic papers using Semantic Scholar or OpenAlex.

**Parameters:**
- `query` (str): Search query for academic papers
- `result_limit` (int, optional): Maximum papers to return (default: 3)

**Example:**
```python
await client.call_tool("search_papers", {
    "query": "transformer neural networks attention mechanism",
    "result_limit": 5
})
```

#### `get_paper_bibtex`
Retrieve BibTeX citation for a specific paper.

**Parameters:**
- `paper_id` (str): Paper ID from Semantic Scholar

**Example:**
```python
await client.call_tool("get_paper_bibtex", {
    "paper_id": "649def34f8be52c8b66281af98ae884c09aef38b"
})
```

### GitHub Search Tools

#### `search_github_repositories`
Search for GitHub repositories.

**Parameters:**
- `query` (str): Search query for repositories
- `result_limit` (int, optional): Maximum repositories to return (default: 10)

**Example:**
```python
await client.call_tool("search_github_repositories", {
    "query": "machine learning python tensorflow",
    "result_limit": 5
})
```

#### `search_github_code`
Search for code files in GitHub repositories.

**Parameters:**
- `query` (str): Search query for code
- `result_limit` (int, optional): Maximum code results to return (default: 10)

**Example:**
```python
await client.call_tool("search_github_code", {
    "query": "neural network implementation pytorch",
    "result_limit": 3
})
```

#### `format_github_query_from_idea`
Generate optimized GitHub search queries from research ideas.

**Parameters:**
- `idea_json` (str): JSON string containing research idea with "Title" and "Experiment" fields

**Example:**
```python
await client.call_tool("format_github_query_from_idea", {
    "idea_json": '{"Title": "Attention Mechanisms", "Experiment": "Compare different attention types"}'
})
```

## Available Resources

### `research://config/settings`
Get current server configuration and API key status.

### `research://papers/recent`
Access recently searched papers (placeholder for caching functionality).

### `research://github/trending/{language}`
Get trending repositories for a specific programming language.

**Example:**
```python
await client.read_resource("research://github/trending/python")
```

## Available Prompts

### `research_paper_search_prompt`
Generate optimized prompts for academic paper searches.

**Parameters:**
- `topic` (str): Research topic
- `focus_area` (str, optional): Specific focus area (default: "general")

### `code_implementation_search_prompt`
Generate prompts for finding code implementations.

**Parameters:**
- `algorithm` (str): Algorithm or technique to search for
- `language` (str, optional): Programming language (default: "python")

### `research_idea_development_prompt`
Generate prompts for research idea development.

**Parameters:**
- `domain` (str): Research domain
- `current_gaps` (str, optional): Known research gaps

## Usage Examples

### Basic Usage

```python
from fastmcp import Client

# Method 1: In-memory testing (fastest)
from test_mcp import mcp as tiny_scientist_server

async with Client(tiny_scientist_server) as client:
    # Search for papers
    papers = await client.call_tool("search_papers", {
        "query": "large language models",
        "result_limit": 3
    })
    
    # Search GitHub repositories
    repos = await client.call_tool("search_github_repositories", {
        "query": "transformer implementation",
        "result_limit": 5
    })
```

### Production Usage (Stdio)

```python
# Method 2: Stdio transport (production)
async with Client("test_mcp.py") as client:
    result = await client.call_tool("search_papers", {
        "query": "quantum computing algorithms"
    })
```

### Multi-Server Configuration

```python
# Method 3: Multi-server setup
config = {
    "mcpServers": {
        "tiny_scientist": {
            "command": "python",
            "args": ["path/to/test_mcp.py"],
            "env": {"OPENALEX_MAIL_ADDRESS": "your.email@example.com"}
        }
    }
}

async with Client(config) as client:
    # Use tools from multiple servers
    result = await client.call_tool("search_papers", {
        "query": "artificial intelligence"
    })
```

## Integration with AI Clients

### Claude Desktop

Add to your Claude Desktop configuration (`~/.config/claude_desktop/mcp.json`):

```json
{
  "mcpServers": {
    "tiny-scientist": {
      "command": "python",
      "args": ["/path/to/tiny-scientist/tests/test_mcp.py"],
      "env": {
        "OPENALEX_MAIL_ADDRESS": "your.email@example.com"
      }
    }
  }
}
```

### Cursor IDE

Add to your Cursor configuration (`~/.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "tiny-scientist": {
      "command": "python",
      "args": ["/path/to/tiny-scientist/tests/test_mcp.py"]
    }
  }
}
```

## Testing

Run the comprehensive test suite:

```bash
cd tiny-scientist/tests
python test_mcp_client.py
```

The test suite includes:
- ✅ In-memory connection testing
- ✅ Stdio transport testing  
- ✅ Multi-server configuration testing
- ✅ All tools, resources, and prompts testing

## Advanced Features

### Context and Logging

The server provides rich context information through the MCP Context API:

```python
@mcp.tool()
async def custom_tool(query: str, ctx: Context):
    # Log information to the client
    await ctx.info(f"Processing query: {query}")
    
    # Report progress for long operations
    await ctx.report_progress(50, 100, "Halfway complete")
    
    # Handle errors gracefully
    try:
        result = some_operation(query)
        await ctx.info("Operation completed successfully")
        return result
    except Exception as e:
        await ctx.error(f"Operation failed: {str(e)}")
        return {"error": str(e)}
```

### Custom Resource Templates

Create dynamic resources with parameters:

```python
@mcp.resource("research://custom/{category}/{subcategory}")
def get_custom_resource(category: str, subcategory: str):
    return f"Custom resource for {category}/{subcategory}"
```

### Error Handling

The server includes robust error handling with exponential backoff for API calls:

```python
@api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
def api_call_function():
    # API call with automatic retry logic
    pass
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install fastmcp requests toml pyalex spacy
   ```

2. **API Rate Limits**: Configure API keys for higher limits:
   - Semantic Scholar: Get API key from [semanticscholar.org](https://semanticscholar.org)
   - GitHub: Create personal access token

3. **Spacy Model Missing**: Install English model
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Configuration Not Found**: Ensure `config.toml` exists in `tiny_scientist/` directory

### Debug Mode

Enable detailed logging by running the server with debug output:

```python
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    mcp.run()
```

## Extension Points

### Adding New Tools

```python
@mcp.tool()
async def my_custom_tool(param: str, ctx: Context) -> Dict[str, Any]:
    """Custom research tool"""
    await ctx.info(f"Running custom tool with {param}")
    # Your implementation here
    return {"result": "success"}
```

### Adding New Resources

```python
@mcp.resource("research://my_resource/{param}")
def my_custom_resource(param: str):
    """Custom research resource"""
    return f"Resource data for {param}"
```

### Adding New Prompts

```python
@mcp.prompt()
def my_custom_prompt(topic: str) -> str:
    """Custom research prompt"""
    return f"Generate research ideas for {topic}"
```

## Performance Considerations

- **In-memory testing**: Fastest for development and unit tests
- **Stdio transport**: Standard for production deployments
- **API rate limits**: Use API keys to increase limits
- **Caching**: Consider implementing result caching for frequently accessed data
- **Async operations**: All tools support asynchronous execution

## Contributing

To extend the server:

1. Add new tools in the appropriate section
2. Update the test suite (`test_mcp_client.py`)
3. Document new features in this README
4. Follow the existing patterns for error handling and logging

## License

This project follows the same license as the tiny-scientist project.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test output for specific error messages
3. Ensure all dependencies and configurations are correct
4. Consider the MCP protocol limitations and requirements 