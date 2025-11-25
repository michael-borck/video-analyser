import asyncio
import aiohttp
import json

# !!! UPDATE THIS TO MATCH YOUR 'opencode serve' OUTPUT !!!
#!/usr/bin/env python3
"""
Test script for opencode server running on port 52180
"""

import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:52180"

def log_test(test_num: int, description: str):
    """Print test header"""
    print(f"\n{test_num}. {description}")
    print("-" * 50)

def test_config():
    """Test getting config"""
    log_test(1, "Getting config")
    try:
        response = requests.get(f"{BASE_URL}/config", timeout=5)
        response.raise_for_status()
        config = response.json()
        print(f"✓ Config retrieved")
        print(f"  Model: {config.get('model', 'N/A')}")
        return config
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def test_projects():
    """Test listing projects"""
    log_test(2, "Listing projects")
    try:
        response = requests.get(f"{BASE_URL}/projects", timeout=5)
        response.raise_for_status()
        projects = response.json()
        project_list = projects if isinstance(projects, list) else projects.get('data', [])
        print(f"✓ Found {len(project_list)} project(s)")
        for proj in project_list:
            print(f"  - {proj.get('name', 'Unknown')}")
        return project_list
    except Exception as e:
        print(f"✗ Error: {e}")
        return []

def test_sessions():
    """Test listing sessions"""
    log_test(3, "Listing sessions")
    try:
        response = requests.get(f"{BASE_URL}/sessions", timeout=5)
        response.raise_for_status()
        sessions = response.json()
        session_list = sessions if isinstance(sessions, list) else sessions.get('data', [])
        print(f"✓ Found {len(session_list)} session(s)")
        for session in session_list:
            print(f"  - {session.get('title', 'Unknown')} (ID: {session.get('id', 'N/A')})")
        return session_list
    except Exception as e:
        print(f"✗ Error: {e}")
        return []

def test_create_session():
    """Test creating a session"""
    log_test(4, "Creating a test session")
    try:
        payload = {
            "title": "Python Test Session"
        }
        response = requests.post(
            f"{BASE_URL}/sessions",
            json=payload,
            timeout=5
        )
        response.raise_for_status()
        session = response.json()
        session_id = session.get('id')
        print(f"✓ Session created")
        print(f"  ID: {session_id}")
        print(f"  Title: {session.get('title', 'N/A')}")
        return session_id
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def test_send_prompt(session_id: str):
    """Test sending a prompt to a session"""
    log_test(5, "Sending a test prompt")
    try:
        payload = {
            "model": {
                "providerID": "anthropic",
                "modelID": "claude-3-5-sonnet-20241022"
            },
            "parts": [
                {
                    "type": "text",
                    "text": "Say hello!"
                }
            ]
        }
        response = requests.post(
            f"{BASE_URL}/sessions/{session_id}/prompt",
            json=payload,
            timeout=30  # Longer timeout for AI response
        )
        response.raise_for_status()
        result = response.json()
        print(f"✓ Prompt sent and response received")
        print(f"  Response type: {result.get('role', 'N/A')}")
        return result
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def test_server_health():
    """Test basic server health"""
    log_test(0, "Testing server health")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        response.raise_for_status()
        print(f"✓ Server is running and responding")
        return True
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to server at {BASE_URL}")
        print(f"  Make sure opencode server is running: opencode serve -p 52180")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("OpenCode Server Test Suite")
    print("=" * 50)
    
    # Check server health first
    if not test_server_health():
        return
    
    # Run tests
    test_config()
    test_projects()
    test_sessions()
    
    session_id = test_create_session()
    if session_id:
        test_send_prompt(session_id)
    
    print("\n" + "=" * 50)
    print("✅ Test suite completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()

