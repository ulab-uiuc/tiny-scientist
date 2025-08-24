"""
Simplified Docker-based Code Generation and Execution with Aider Auto-fixing

This module provides a minimal implementation for:
1. Generating Python experiment code using LLMs
2. Running code in Docker containers 
3. Auto-fixing errors using Aider inside containers
"""

import json
import os
import tempfile
from typing import Dict, Optional, Tuple

import docker
from .utils.llm import create_client, get_response_from_llm

# Add requests import for HuggingFace API
import requests
import time


class DockerCoder:
    """Simplified Docker-based code generator with Aider auto-fixing"""
    
    def __init__(self, model: str = "gpt-4o-mini", output_dir: str = "results"):
        try:
            self.client, self.model = create_client(model)
            print(f"✅ LLM client created: {model}")
        except Exception as e:
            print(f"❌ LLM client creation failed: {e}")
            print("💡 Make sure to set OPENAI_API_KEY or configure API keys")
            raise
            
        self.output_dir = os.path.abspath(output_dir)
        self.docker_client = docker.from_env()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def evaluate_model(
        self, 
        model_name: str, 
        dataset_name: str, 
        metric: str,
        max_runs: int = 3,
        max_fixes: int = 5
    ) -> Tuple[bool, str]:
        """
        Generate and run experiment with automatic error fixing
        
        Args:
            model_name: HuggingFace model identifier
            dataset_name: HuggingFace dataset identifier  
            metric: Evaluation metric (accuracy, f1, etc.)
            max_runs: Number of experiment runs
            max_fixes: Max Aider fix attempts per failed run
            
        Returns:
            Tuple of (success, message)
        """
        print(f"🎯 Evaluating {model_name} on {dataset_name} using {metric}")
        
        # Step 1: Generate experiment code using LLM
        if not self._generate_experiment_code(model_name, dataset_name, metric):
            return False, "Failed to generate experiment code"
            
        # Step 2: Create Docker container
        container = self._create_container()
        if not container:
            return False, "Failed to create Docker container"
            
        try:
            # Step 3: Run experiments with auto-fixing
            success_count = 0
            for run in range(1, max_runs + 1):
                print(f"\n🔄 Run {run}/{max_runs}")
                
                if self._run_experiment_with_fixes(container, run, max_fixes):
                    success_count += 1
                    print(f"✅ Run {run} succeeded")
                else:
                    print(f"❌ Run {run} failed after {max_fixes} fix attempts")
            
            if success_count > 0:
                return True, f"Completed {success_count}/{max_runs} successful runs"
            else:
                return False, f"All {max_runs} runs failed"
                
        finally:
            # Clean up container
            container.stop()
            container.remove()
            
    def _generate_experiment_code(self, model_name: str, dataset_name: str, metric: str) -> bool:
        """Generate Python experiment code using LLM only - no fallback"""
        
        # Require LLM client - no fallback
        if self.client is None:
            print("❌ No LLM client available. Cannot generate experiment code.")
            return False
            
        # Get model README for context
        print(f"📖 Fetching README for {model_name}...")
        model_readme = self._get_model_readme(model_name)
        
        # LLM generation only
        prompt = f"""
Generate a complete Python script to evaluate the HuggingFace model '{model_name}' 
on the '{dataset_name}' dataset using the '{metric}' metric.

MODEL README INFORMATION:
```
{model_readme}
```

GUIDELINES (be flexible and adapt as needed):

**Authentication & Environment:**
- Use HF_TOKEN from environment: hf_token = os.getenv('HF_TOKEN')
- Pass token to model loading: token=hf_token (if needed)
- DO NOT add --token command line argument - token comes from environment only
- Detect CUDA availability and use appropriate device

**Model & Dataset Loading:**
- Load the model and tokenizer appropriately for the task type
- Handle different dataset structures intelligently
- Adapt to the specific dataset format (text classification, QA, etc.)

**Data Processing Strategy:**
- Examine the dataset structure and adapt accordingly
- For text tasks: intelligently combine relevant fields (question+passage, premise+hypothesis, etc.)
- For classification: map labels correctly based on model outputs
- Process examples efficiently - you can batch if it's more appropriate
- Limit to around 100 samples for speed, but adjust if needed

**Inference & Evaluation:**
- Use the model in the most appropriate way for its task type
- Handle different output formats (logits, probabilities, text generation)
- Calculate the requested metric correctly using sklearn or appropriate libraries
- Be robust to different label formats (bool, int, string)

**Output Requirements:**
- Use argparse for --out_dir parameter ONLY (no other parameters needed)
- Get HF_TOKEN from environment variable os.getenv('HF_TOKEN'), not from command line
- Save results as JSON with the metric name and value
- Include basic metadata (total_samples, etc.)
- Print progress and final results

**Error Handling:**
- Be reasonably robust but don't over-engineer
- Let the script fail fast on major issues rather than silently skipping everything
- Print helpful error messages

**Flexibility Guidelines:**
- The model README contains the authoritative usage information - follow it
- Adapt the code structure to what makes sense for this specific model+dataset combination  
- Don't force a rigid pattern if the model/dataset has special requirements
- Use your judgment on tokenization, preprocessing, and inference approach

Generate a complete, working Python script that intelligently handles this specific model and dataset combination:

IMPORTANT: The script will be called as: python experiment.py --out_dir=DIRECTORY
Do not add any other command line parameters!
"""
        try:
            print("🤖 Generating experiment code with LLM...")
            response, _ = get_response_from_llm(
                msg=prompt,
                client=self.client,
                model=self.model,
                system_message="You are an expert Python developer. Generate complete, robust code for ML experiments. Follow the requirements exactly."
            )
            
            # Extract and save code
            code = self._extract_code(response)
            exp_path = os.path.join(self.output_dir, "experiment.py")
            with open(exp_path, "w") as f:
                f.write(code)
                
            print(f"✅ Generated experiment code: {exp_path}")
            return True
            
        except Exception as e:
            print(f"❌ LLM generation failed: {e}")
            return False
            
    def _create_container(self) -> Optional[docker.models.containers.Container]:
        """Create Docker container with Python + Aider + ML libraries"""
        try:
            # Use existing image
            image_name = "simple-coder:latest"
            
            # Environment variables
            env_vars = {}
            for key in ["HF_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
                if os.getenv(key):
                    env_vars[key] = os.getenv(key)
                    print(f"🔑 Found {key}: {os.getenv(key)[:10]}...")
                else:
                    print(f"❌ Missing {key}")
            
            print(f"📦 Environment variables to pass: {list(env_vars.keys())}")
            
            # Create container
            container = self.docker_client.containers.run(
                image=image_name,
                command="tail -f /dev/null",  # Keep running
                volumes={self.output_dir: {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace",
                environment=env_vars,
                detach=True,
                remove=False,
                mem_limit="4g"
            )
            
            # Initialize git (Aider requirement)
            container.exec_run("git init", workdir="/workspace")
            container.exec_run("git config user.email 'test@example.com'", workdir="/workspace")
            container.exec_run("git config user.name 'Test User'", workdir="/workspace")
            # Configure git to work in limited environment
            container.exec_run("git config diff.external false", workdir="/workspace")
            container.exec_run("git config core.filemode false", workdir="/workspace")
            container.exec_run("git add .", workdir="/workspace")
            container.exec_run("git commit -m 'Initial commit' --allow-empty", workdir="/workspace")
                    
            print(f"✅ Container ready: {container.id[:12]}")
            return container
            
        except Exception as e:
            print(f"❌ Container creation failed: {e}")
            return None
            
    def _run_experiment_with_fixes(self, container, run_num: int, max_fixes: int) -> bool:
        """Run experiment with automatic Aider fixes on failure"""
        
        for attempt in range(max_fixes + 1):  # +1 for initial attempt
            # Run experiment
            print(f"  🧪 Attempt {attempt + 1}")
            print(f"  📋 Running: python experiment.py --out_dir=run_{run_num}")
            
            result = container.exec_run(
                f"python experiment.py --out_dir=run_{run_num}",
                workdir="/workspace"
            )
            
            output = result.output.decode('utf-8') if result.output else ""
            print(f"  📊 Exit code: {result.exit_code}")
            print(f"  📝 Output:")
            print("=" * 50)
            print(output)
            print("=" * 50)
            
            # Check for success
            if result.exit_code == 0 and not self._has_errors(output):
                # Additional check: verify any JSON results file was created
                run_dir = os.path.join(self.output_dir, f"run_{run_num}")
                json_files = []
                if os.path.exists(run_dir):
                    json_files = [f for f in os.listdir(run_dir) if f.endswith('.json')]
                
                if json_files:
                    print(f"  ✅ Success on attempt {attempt + 1} - found JSON files: {json_files}")
                    return True
                else:
                    print(f"  ⚠️  Exit code 0 but no JSON files found in {run_dir}, treating as failure")
            else:
                print(f"  ❌ Failed on attempt {attempt + 1}")
                
            # Failed - try Aider fix if we have API keys
            if attempt < max_fixes and os.getenv("OPENAI_API_KEY"):
                print(f"  🔧 Fixing with Aider (attempt {attempt + 1}/{max_fixes})")
                
                fix_prompt = f"""
The experiment failed with this error:
{output}

Please fix the experiment.py file to resolve this error.
Make sure the code handles authentication, imports, and data processing correctly.
"""
                
                print(f"  🤖 Running Aider with prompt:")
                print(f"  {fix_prompt.strip()}")
                print(f"  {'='*60}")
                
                fix_result = container.exec_run([
                    "aider", "--yes", "--model", "gpt-4o-mini", 
                    "--no-git",
                    "--message", fix_prompt, "experiment.py"
                ], workdir="/workspace")
                
                # Export and display Aider's complete output
                aider_output = fix_result.output.decode('utf-8') if fix_result.output else ""
                print(f"  📋 Aider output (exit code: {fix_result.exit_code}):")
                print(f"  {'─'*60}")
                if aider_output.strip():
                    # Show each line with prefix for clarity
                    for line in aider_output.split('\n'):
                        print(f"  │ {line}")
                else:
                    print(f"  │ (No output from Aider)")
                print(f"  {'─'*60}")
                
                if fix_result.exit_code != 0:
                    print(f"  ❌ Aider fix failed (exit code: {fix_result.exit_code})")
                else:
                    print(f"  ✅ Aider fix completed")
            elif attempt < max_fixes:
                print(f"  ⚠️  No OPENAI_API_KEY, skipping Aider fix")
            else:
                print(f"  💀 Max fix attempts reached")
                
        return False
        
    def _has_errors(self, output: str) -> bool:
        """Check if output contains error indicators"""
        error_patterns = [
            'traceback', 'error:', 'exception:', 'failed', 
            'importerror', 'modulenotfounderror', 'syntaxerror',
            '401 client error', 'unauthorized', 'invalid credentials',
            'http error', 'connection error', 'authentication',
            'hfhubhttperror', 'oserror', 'valueerror'
        ]
        
        # Check for specific error patterns
        has_error = any(pattern in output.lower() for pattern in error_patterns)
        
        if has_error:
            return True
            
        # Only consider it an error if we see explicit failure patterns
        # Don't require specific success indicators - let file existence check handle success
        return False
         
    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response"""
        # Look for code blocks
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
                
        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
                
        # If no code blocks, return the whole response
        return response.strip()

    def _get_model_readme(self, model_name: str) -> str:
        """Fetch model README from HuggingFace Hub API"""
        try:
            # Get HF token if available
            hf_token = os.getenv('HF_TOKEN')
            headers = {}
            if hf_token:
                headers['Authorization'] = f'Bearer {hf_token}'
            
            # Try to get README from HF Hub API
            url = f"https://huggingface.co/{model_name}/resolve/main/README.md"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                readme_content = response.text
                # Limit README size to avoid token overflow
                if len(readme_content) > 3000:
                    readme_content = readme_content[:3000] + "\n... (truncated)"
                return readme_content
            else:
                return f"README not available (HTTP {response.status_code})"
                
        except Exception as e:
            return f"Failed to fetch README: {str(e)}"