#!/usr/bin/env python3
"""
Simple MCP Server Test Script

This script directly tests whether MCP server can start and respond to basic requests
"""

import asyncio
import json
import subprocess
import sys


async def test_mcp_server_direct():
    """Test MCP server directly"""
    print("🧪 Testing MCP server directly...")
    
    # Build command
    cmd = ["npx", "-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"]
    print(f"🚀 Starting command: {' '.join(cmd)}")
    
    try:
        # Start process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        print(f"✅ Process started (PID: {process.pid})")
        
        # Wait a moment for process to start
        await asyncio.sleep(2)
        
        # Check if process is still running
        if process.returncode is not None:
            stderr_output = await process.stderr.read()
            print(f"❌ Process exited with return code: {process.returncode}")
            print(f"Error output: {stderr_output.decode()}")
            return False
        
        print("🔄 Process is running, trying to send initialization request...")
        
        # Send initialization request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "TestClient",
                    "version": "1.0.0"
                }
            }
        }
        
        request_data = json.dumps(init_request) + "\n"
        print(f"📤 Sending request: {request_data.strip()}")
        
        process.stdin.write(request_data.encode())
        await process.stdin.drain()
        
        # Wait for response
        try:
            response_data = await asyncio.wait_for(
                process.stdout.readline(), 
                timeout=10.0
            )
            
            if response_data:
                response_str = response_data.decode().strip()
                print(f"📥 Received response: {response_str}")
                
                try:
                    response_json = json.loads(response_str)
                    if "result" in response_json:
                        print("✅ Initialization successful!")
                        return True
                    else:
                        print(f"⚠️ Initialization response abnormal: {response_json}")
                        return False
                except json.JSONDecodeError as e:
                    print(f"❌ Response is not valid JSON: {e}")
                    return False
            else:
                print("❌ No response received")
                return False
                
        except asyncio.TimeoutError:
            print("⏰ Timeout waiting for response")
            return False
            
    except Exception as e:
        print(f"💥 Error occurred during testing: {e}")
        return False
        
    finally:
        # Clean up process
        if 'process' in locals() and process:
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=5.0)
                print("🧹 Process cleaned up")
            except:
                process.kill()
                await process.wait()
                print("🧹 Process forcefully terminated")


async def check_npx_availability():
    """Check if npx is available"""
    print("🔍 Checking npx availability...")
    
    try:
        result = await asyncio.create_subprocess_exec(
            "npx", "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await result.communicate()
        
        if result.returncode == 0:
            version = stdout.decode().strip()
            print(f"✅ npx is available, version: {version}")
            return True
        else:
            print(f"❌ npx is not available, error: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking npx: {e}")
        return False


async def main():
    """Main function"""
    print("🚀 MCP Server Direct Test")
    print("=" * 40)
    
    # Check npx
    if not await check_npx_availability():
        print("💡 Please ensure Node.js and npm are installed")
        return 1
    
    # Test MCP server
    success = await test_mcp_server_direct()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 MCP server test successful!")
        return 0
    else:
        print("😞 MCP server test failed")
        print("💡 Possible reasons:")
        print("  - Network connection issues")
        print("  - MCP server package not properly installed")
        print("  - Server startup time too long")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error occurred during test run: {e}")
        sys.exit(1) 