#!/usr/bin/env python3
"""
Test script for server restart protocol.

This script tests the automatic server restart functionality by:
1. Starting a server
2. Simulating a timeout
3. Triggering a restart
4. Verifying the server comes back online
"""

import sys
import time
import requests
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_with_server import ServerManager, set_server_manager
from utils.lm import restart_server, check_server_health
from loguru import logger


def test_server_restart():
    """Test the server restart functionality."""
    
    logger.info("=" * 80)
    logger.info("SERVER RESTART PROTOCOL TEST")
    logger.info("=" * 80)
    
    # Configuration
    model = "meta-llama/Llama-3.1-8B-Instruct"
    host = "0.0.0.0"
    port = 8000
    
    # Create server manager
    logger.info(f"Creating ServerManager for model: {model}")
    server_manager = ServerManager(
        model=model,
        host=host,
        port=port,
        logger_type="lmdb",
        activations_path="lmdb_data/test_restart.lmdb",
        log_file_path="test_restart_server.log"
    )
    
    try:
        # Step 1: Start server
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Starting server...")
        logger.info("=" * 80)
        server_manager.start_server()
        set_server_manager(server_manager)
        logger.success("Server started successfully")
        
        # Step 2: Verify server is healthy
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Checking server health...")
        logger.info("=" * 80)
        if check_server_health(f"http://{host}:{port}"):
            logger.success("Server is healthy")
        else:
            logger.error("Server health check failed")
            return False
        
        # Step 3: Test a simple request
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Testing simple request...")
        logger.info("=" * 80)
        try:
            response = requests.post(
                f"http://{host}:{port}/v1/completions",
                json={
                    "model": model,
                    "prompt": "Hello, world!",
                    "max_tokens": 5,
                    "temperature": 0.0
                },
                timeout=30
            )
            if response.status_code == 200:
                logger.success("Simple request succeeded")
                logger.info(f"Response: {response.json()}")
            else:
                logger.warning(f"Request returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Request failed: {e}")
        
        # Step 4: Trigger restart
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Triggering server restart...")
        logger.info("=" * 80)
        
        restart_success = restart_server(f"http://{host}:{port}/v1", wait_time=60)
        
        if restart_success:
            logger.success("Server restart succeeded")
        else:
            logger.error("Server restart failed")
            return False
        
        # Step 5: Verify server is healthy after restart
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Checking server health after restart...")
        logger.info("=" * 80)
        if check_server_health(f"http://{host}:{port}"):
            logger.success("Server is healthy after restart")
        else:
            logger.error("Server health check failed after restart")
            return False
        
        # Step 6: Test another request after restart
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Testing request after restart...")
        logger.info("=" * 80)
        try:
            response = requests.post(
                f"http://{host}:{port}/v1/completions",
                json={
                    "model": model,
                    "prompt": "Test after restart",
                    "max_tokens": 5,
                    "temperature": 0.0
                },
                timeout=30
            )
            if response.status_code == 200:
                logger.success("Request after restart succeeded")
                logger.info(f"Response: {response.json()}")
            else:
                logger.warning(f"Request returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Request after restart failed: {e}")
            return False
        
        logger.info("\n" + "=" * 80)
        logger.success("ALL TESTS PASSED!")
        logger.info("=" * 80)
        return True
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Always stop server
        logger.info("\n" + "=" * 80)
        logger.info("CLEANUP: Stopping server...")
        logger.info("=" * 80)
        server_manager.stop_server()
        logger.info("Server stopped")


def test_restart_without_server_manager():
    """Test restart when ServerManager is not available (fallback mode)."""
    
    logger.info("=" * 80)
    logger.info("TESTING RESTART WITHOUT SERVER MANAGER (FALLBACK MODE)")
    logger.info("=" * 80)
    logger.warning("This test requires a server to be running externally")
    logger.warning("Start server with: python -m activation_logging.vllm_serve --model meta-llama/Llama-3.1-8B-Instruct")
    
    host = "0.0.0.0"
    port = 8000
    
    # Check if server is running
    if not check_server_health(f"http://{host}:{port}"):
        logger.error("Server is not running. Please start it manually first.")
        return False
    
    logger.info("Server is running, attempting restart via REST endpoint...")
    
    # Try restart via REST endpoint (won't work if server is hung)
    try:
        response = requests.post(f"http://{host}:{port}/restart", timeout=5)
        logger.info(f"Restart endpoint returned: {response.status_code}")
        
        # Wait for server to restart
        logger.info("Waiting for server to restart...")
        time.sleep(10)
        
        # Check if server is back
        for i in range(30):
            if check_server_health(f"http://{host}:{port}"):
                logger.success(f"Server is back online after {i*2}s")
                return True
            time.sleep(2)
        
        logger.error("Server did not come back online")
        return False
        
    except Exception as e:
        logger.error(f"Restart failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test server restart protocol")
    parser.add_argument("--mode", choices=["full", "fallback"], default="full",
                       help="Test mode: 'full' tests with ServerManager, 'fallback' tests REST endpoint only")
    args = parser.parse_args()
    
    if args.mode == "full":
        success = test_server_restart()
    else:
        success = test_restart_without_server_manager()
    
    sys.exit(0 if success else 1)

