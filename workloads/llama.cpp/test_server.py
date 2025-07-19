#!/usr/bin/env python3
"""
Simple test script to verify llama.cpp server is working
"""

import time
import requests
import subprocess
import sys

def test_server():
    # Start server
    server_cmd = [
        "./build/bin/llama-server",
        "-m", "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "--port", "8080",
        "--host", "0.0.0.0",
        "-c", "2048",  # context size
        "-t", "4"      # threads
    ]
    
    print("Starting llama.cpp server with TinyLlama model...")
    server_process = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start (model loading takes time)
    time.sleep(10)
    
    try:
        # Test health endpoint
        print("Testing health endpoint...")
        response = requests.get("http://localhost:8080/health", timeout=5)
        print(f"Health check response: {response.status_code}")
        
        if response.status_code == 200:
            print("✓ Server is running successfully!")
            
            # Test server info
            info_response = requests.get("http://localhost:8080/props", timeout=5)
            if info_response.status_code == 200:
                print(f"Server info: {info_response.json()}")
                
            # Test completion endpoint
            print("\nTesting completion endpoint...")
            completion_data = {
                "prompt": "Hello, how are you?",
                "max_tokens": 50,
                "temperature": 0.7
            }
            completion_response = requests.post(
                "http://localhost:8080/completion",
                json=completion_data,
                timeout=30
            )
            if completion_response.status_code == 200:
                print("✓ Completion request successful!")
                print(f"Response: {completion_response.json()['content']}")
            else:
                print(f"✗ Completion request failed: {completion_response.status_code}")
        else:
            print(f"✗ Server health check failed with status: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("✗ Failed to connect to server")
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        # Stop server
        print("\nStopping server...")
        server_process.terminate()
        server_process.wait(timeout=10)
        print("Server stopped.")

if __name__ == "__main__":
    test_server()