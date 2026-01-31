#!/usr/bin/env python3
"""
Minimal Jupyter REST API Client for HalluLens Remote Development

This script provides basic programmatic access to execute commands via Jupyter server.
Designed for testing command execution on port 8887.

Usage:
    python jupyter_api_executor.py --command "echo hello world"
    python jupyter_api_executor.py --check-status
"""

import argparse
import requests
import json
import time
import sys
import websocket
import threading
from urllib.parse import urljoin

class JupyterAPIExecutor:
    """Minimal client for executing commands via Jupyter REST API."""

    def __init__(self, base_url="http://localhost:8887", token=None, password=None):
        self.base_url = base_url
        self.token = token
        self.password = password
        self.session = requests.Session()

        # Set up authentication if token is provided
        if token:
            self.session.headers.update({'Authorization': f'token {token}'})
    
    def login_with_password(self):
        """Attempt to login with password and get session cookie."""
        if not self.password:
            return False

        try:
            # Get login page to extract CSRF token
            login_response = self.session.get(urljoin(self.base_url, '/login'))

            # Extract CSRF token from the response
            import re
            csrf_match = re.search(r'name="_xsrf" value="([^"]+)"', login_response.text)

            if not csrf_match:
                print("❌ Could not find CSRF token in login page")
                return False

            csrf_token = csrf_match.group(1)
            print(f"🔑 Found CSRF token: {csrf_token[:20]}...")

            # Attempt login with password and CSRF token
            login_data = {
                'password': self.password,
                '_xsrf': csrf_token
            }
            response = self.session.post(urljoin(self.base_url, '/login'), data=login_data)

            if response.status_code == 302:
                print("✅ Successfully logged in with password (redirected)")
                return True
            elif response.status_code == 200 and 'login' not in response.url:
                print("✅ Successfully logged in with password")
                return True
            else:
                print(f"❌ Login failed with status {response.status_code}")
                print(f"   Response URL: {response.url}")
                return False
        except Exception as e:
            print(f"❌ Error during login: {e}")
            return False

    def check_server_status(self):
        """Check if Jupyter server is accessible."""
        try:
            response = self.session.get(urljoin(self.base_url, '/api/status'))
            if response.status_code == 200:
                print("✅ Jupyter server is accessible")
                return True
            elif response.status_code == 403:
                print("🔐 Jupyter server requires authentication (403 Forbidden)")

                # Try to login with password if provided
                if self.password:
                    print("🔑 Attempting login with password...")
                    if self.login_with_password():
                        # Retry status check after login
                        response = self.session.get(urljoin(self.base_url, '/api/status'))
                        if response.status_code == 200:
                            print("✅ Jupyter server is accessible after login")
                            return True
                        else:
                            print(f"❌ Still getting status {response.status_code} after login")
                            return False
                    else:
                        print("❌ Password login failed")
                        return False
                else:
                    print("   Try: python jupyter_api_executor.py --password 123")
                    print("   Or: python jupyter_api_executor.py --token YOUR_TOKEN")
                    return False
            else:
                print(f"❌ Jupyter server returned status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to Jupyter server")
            print(f"   Make sure the tunnel is active: ssh ... -L 8887:gpu-node:8887")
            return False
        except Exception as e:
            print(f"❌ Error checking server status: {e}")
            return False
    
    def list_terminals(self):
        """List active terminals."""
        try:
            response = self.session.get(urljoin(self.base_url, '/api/terminals'))
            if response.status_code == 200:
                terminals = response.json()
                print(f"📋 Found {len(terminals)} active terminals:")
                for term in terminals:
                    print(f"   Terminal {term['name']}")
                return terminals
            else:
                print(f"❌ Failed to list terminals: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Error listing terminals: {e}")
            return []
    
    def create_terminal(self):
        """Create a new terminal session."""
        try:
            response = self.session.post(urljoin(self.base_url, '/api/terminals'))
            if response.status_code == 200:
                terminal = response.json()
                print(f"✅ Created terminal: {terminal['name']}")
                return terminal
            else:
                print(f"❌ Failed to create terminal: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error creating terminal: {e}")
            return None
    
    def execute_command(self, command, terminal_name=None, timeout=10):
        """Execute a command in a terminal via WebSocket."""
        # Get or create terminal
        if terminal_name is None:
            terminals = self.list_terminals()
            if terminals:
                terminal_name = terminals[0]['name']
            else:
                terminal = self.create_terminal()
                if terminal:
                    terminal_name = terminal['name']
                else:
                    return False, "Failed to create terminal"

        print(f"🚀 Executing command in terminal {terminal_name}: {command}")

        try:
            # Build WebSocket URL for JupyterLab - try multiple formats
            ws_base = self.base_url.replace('http://', 'ws://').replace('https://', 'wss://')

            # Add session cookies to WebSocket headers
            cookies = "; ".join([f"{k}={v}" for k, v in self.session.cookies.items()])
            headers = {"Cookie": cookies} if cookies else {}

            # Try different WebSocket URL patterns for JupyterLab
            ws_patterns = [
                f"{ws_base}/api/terminals/{terminal_name}/channels",
                f"{ws_base}/terminals/{terminal_name}/websocket",
                f"{ws_base}/terminals/websocket/{terminal_name}",
                f"{ws_base}/api/terminals/{terminal_name}/websocket"
            ]

            success = False
            for ws_url in ws_patterns:
                print(f"🔗 Trying WebSocket: {ws_url}")
                try:
                    success = self._try_websocket_connection(ws_url, headers, command)
                    if success:
                        break
                except Exception as e:
                    print(f"   ❌ Failed: {str(e)[:100]}...")
                    continue

            if not success:
                return False, "All WebSocket connection attempts failed"

            return True, "Command executed successfully"

        except Exception as e:
            print(f"❌ Error executing command: {e}")
            return False, f"Execution failed: {e}"

    def _try_websocket_connection(self, ws_url, headers, command):
        """Try a specific WebSocket URL pattern."""
        output = []
        command_sent = False

        def on_message(ws, message):
            nonlocal output
            try:
                data = json.loads(message)
                if isinstance(data, list) and len(data) >= 2:
                    msg_type, content = data[0], data[1]
                    if msg_type == 'stdout':
                        output.append(content)
                        print(content, end='')
            except:
                # Handle non-JSON messages
                output.append(str(message))

        def on_open(ws):
            nonlocal command_sent
            # Send command
            message = json.dumps(['stdin', command + '\r\n'])
            ws.send(message)
            command_sent = True
            print(f"   📤 Sent command: {command}")

        def on_error(ws, error):
            raise Exception(f"WebSocket error: {error}")

        # Create WebSocket connection with authentication headers
        ws = websocket.WebSocketApp(ws_url,
                                  header=headers,
                                  on_message=on_message,
                                  on_open=on_open,
                                  on_error=on_error)

        # Run WebSocket in a thread with timeout
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()

        # Wait for command execution
        start_time = time.time()
        timeout = 10
        while time.time() - start_time < timeout:
            if command_sent and output:
                # Give some time for output to complete
                time.sleep(1)
                break
            time.sleep(0.1)

        ws.close()

        if command_sent and output:
            output_text = ''.join(output)
            print(f"\n   ✅ Success! Output length: {len(output_text)} chars")
            return True
        else:
            raise Exception("No output received or command not sent")

def main():
    parser = argparse.ArgumentParser(description='Minimal Jupyter API executor for testing')
    parser.add_argument('--command', '-c', help='Command to execute (e.g., "echo hello world")')
    parser.add_argument('--check-status', action='store_true', help='Check server status')
    parser.add_argument('--list-terminals', action='store_true', help='List active terminals')
    parser.add_argument('--url', default='http://localhost:8887', help='Jupyter server URL')
    parser.add_argument('--token', help='Jupyter authentication token')
    parser.add_argument('--password', default='123', help='Jupyter password (default: 123)')

    args = parser.parse_args()

    print("🧪 Minimal Jupyter API Executor - Testing Mode")
    print("=" * 50)

    # Initialize client
    client = JupyterAPIExecutor(args.url, args.token, args.password)
    
    # Check server status first
    if not client.check_server_status():
        print("\n💡 Troubleshooting:")
        print("   1. Ensure SSH tunnel is active: ssh ... -L 8887:gpu-node:8887")
        print("   2. Verify Jupyter server is running on remote node")
        return
    
    # Execute requested action
    if args.check_status:
        print("✅ Server status check completed")
    elif args.list_terminals:
        client.list_terminals()
    elif args.command:
        success, output = client.execute_command(args.command)
        if success:
            print(f"\n✅ Command executed successfully")
            print(f"📄 Output:\n{output}")
        else:
            print(f"\n❌ Command failed: {output}")
    else:
        # Default: test with "echo hello world"
        print("🔬 Running default test command...")
        success, output = client.execute_command("echo hello world")
        if success:
            print(f"\n✅ Test successful!")
            print(f"📄 Output:\n{output}")
        else:
            print(f"\n❌ Test failed: {output}")

if __name__ == '__main__':
    main()
