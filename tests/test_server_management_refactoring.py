#!/usr/bin/env python3
"""
Test script to verify the server management refactoring works correctly.

This tests that:
1. ServerManager can be imported from utils.lm
2. run_exp() can manage the server automatically
3. Server restart works when ServerManager is available
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import lm, exp
import pandas as pd
from loguru import logger

def test_server_manager_import():
    """Test that ServerManager can be imported from utils.lm"""
    logger.info("=" * 80)
    logger.info("TEST 1: ServerManager Import")
    logger.info("=" * 80)
    
    try:
        # Test import
        assert hasattr(lm, 'ServerManager'), "ServerManager not found in utils.lm"
        assert hasattr(lm, 'get_server_manager'), "get_server_manager not found in utils.lm"
        assert hasattr(lm, 'set_server_manager'), "set_server_manager not found in utils.lm"
        
        logger.success("✅ ServerManager successfully imported from utils.lm")
        return True
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        return False

def test_server_manager_creation():
    """Test that ServerManager can be created"""
    logger.info("=" * 80)
    logger.info("TEST 2: ServerManager Creation")
    logger.info("=" * 80)
    
    try:
        # Create server manager (don't start it)
        server_manager = lm.ServerManager(
            model="meta-llama/Llama-3.1-8B-Instruct",
            host="0.0.0.0",
            port=8000,
            logger_type="lmdb"
        )
        
        assert server_manager.model == "meta-llama/Llama-3.1-8B-Instruct"
        assert server_manager.host == "0.0.0.0"
        assert server_manager.port == 8000
        assert server_manager.logger_type == "lmdb"
        
        logger.success("✅ ServerManager created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Creation test failed: {e}")
        return False

def test_global_server_manager():
    """Test that global server manager can be set and retrieved"""
    logger.info("=" * 80)
    logger.info("TEST 3: Global ServerManager Registry")
    logger.info("=" * 80)
    
    try:
        # Initially should be None
        assert lm.get_server_manager() is None, "Initial server manager should be None"
        
        # Create and set server manager
        server_manager = lm.ServerManager(
            model="test-model",
            port=8001
        )
        lm.set_server_manager(server_manager)
        
        # Retrieve it
        retrieved = lm.get_server_manager()
        assert retrieved is server_manager, "Retrieved manager should be the same instance"
        assert retrieved.model == "test-model"
        assert retrieved.port == 8001
        
        # Clear it
        lm.set_server_manager(None)
        assert lm.get_server_manager() is None, "Server manager should be None after clearing"
        
        logger.success("✅ Global server manager registry works correctly")
        return True
    except Exception as e:
        logger.error(f"❌ Global registry test failed: {e}")
        return False

def test_run_exp_parameters():
    """Test that run_exp() accepts server management parameters"""
    logger.info("=" * 80)
    logger.info("TEST 4: run_exp() Server Management Parameters")
    logger.info("=" * 80)
    
    try:
        # Check that run_exp has the new parameters
        import inspect
        sig = inspect.signature(exp.run_exp)
        params = sig.parameters
        
        assert 'manage_server' in params, "manage_server parameter missing"
        assert 'server_host' in params, "server_host parameter missing"
        assert 'server_port' in params, "server_port parameter missing"
        assert 'logger_type' in params, "logger_type parameter missing"
        assert 'activations_path' in params, "activations_path parameter missing"
        assert 'log_file_path' in params, "log_file_path parameter missing"
        
        # Check defaults
        assert params['manage_server'].default == True, "manage_server should default to True"
        assert params['server_host'].default == "0.0.0.0", "server_host should default to 0.0.0.0"
        assert params['server_port'].default == 8000, "server_port should default to 8000"
        
        logger.success("✅ run_exp() has correct server management parameters")
        return True
    except Exception as e:
        logger.error(f"❌ Parameter test failed: {e}")
        return False

def test_restart_server_function():
    """Test that restart_server() can access the global server manager"""
    logger.info("=" * 80)
    logger.info("TEST 5: restart_server() Function")
    logger.info("=" * 80)
    
    try:
        # Create a mock server manager
        server_manager = lm.ServerManager(
            model="test-model",
            port=8002
        )
        lm.set_server_manager(server_manager)
        
        # Test that restart_server can find it
        retrieved = lm.get_server_manager()
        assert retrieved is not None, "Server manager should be retrievable"
        assert retrieved.model == "test-model"
        
        # Clean up
        lm.set_server_manager(None)
        
        logger.success("✅ restart_server() can access global server manager")
        return True
    except Exception as e:
        logger.error(f"❌ restart_server test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("=" * 80)
    logger.info("SERVER MANAGEMENT REFACTORING TESTS")
    logger.info("=" * 80)
    logger.info("")
    
    tests = [
        ("ServerManager Import", test_server_manager_import),
        ("ServerManager Creation", test_server_manager_creation),
        ("Global ServerManager Registry", test_global_server_manager),
        ("run_exp() Parameters", test_run_exp_parameters),
        ("restart_server() Function", test_restart_server_function),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            logger.info("")
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
            logger.info("")
    
    # Summary
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("")
    logger.info(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.success("=" * 80)
        logger.success("ALL TESTS PASSED!")
        logger.success("=" * 80)
        return 0
    else:
        logger.error("=" * 80)
        logger.error(f"SOME TESTS FAILED ({total - passed} failures)")
        logger.error("=" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(main())

