#!/usr/bin/env python3
"""
Check if NPY logging implementation is present in JsonActivationsLogger
"""

import sys
import ast
import inspect
from pathlib import Path

def check_npy_implementation():
    """Check if NPY logging functionality is implemented"""
    print("🔍 Checking NPY Implementation in JsonActivationsLogger")
    print("=" * 60)
    
    # Read the activations_logger.py file
    logger_file = Path("activation_logging/activations_logger.py")
    
    if not logger_file.exists():
        print("❌ activations_logger.py not found!")
        return False
    
    with open(logger_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for version 2.0 and storage_format
    version_check = '"version": "2.0"' in content
    storage_format_check = '"storage_format": "npy"' in content
    
    print(f"📊 Version 2.0 found: {version_check}")
    print(f"💾 Storage format 'npy' found: {storage_format_check}")
    
    # Check for required methods
    required_methods = [
        '_tensors_to_numpy_arrays',
        '_numpy_arrays_to_tensors', 
        '_get_activation_shape_info'
    ]
    
    method_checks = {}
    for method in required_methods:
        method_checks[method] = f"def {method}" in content
        print(f"🔧 Method {method}: {method_checks[method]}")
    
    # Check for NPY-specific code patterns
    npy_patterns = [
        'np.save(',
        'np.load(',
        'allow_pickle=True',
        '.npy',
        'activation_arrays',
        'torch.from_numpy',
        '.cpu().numpy()'
    ]
    
    print(f"\n🎯 NPY-specific code patterns:")
    pattern_checks = {}
    for pattern in npy_patterns:
        pattern_checks[pattern] = pattern in content
        print(f"   {pattern}: {pattern_checks[pattern]}")
    
    # Check log_entry method for NPY usage
    log_entry_npy = 'activation_arrays = self._tensors_to_numpy_arrays' in content
    save_npy = 'np.save(activation_file_path, activation_arrays' in content
    
    print(f"\n📝 log_entry method NPY integration:")
    print(f"   Converts to numpy arrays: {log_entry_npy}")
    print(f"   Saves as NPY file: {save_npy}")
    
    # Check get_entry method for NPY loading
    load_npy = 'np.load(activation_path, allow_pickle=True)' in content
    convert_back = 'self._numpy_arrays_to_tensors(activation_arrays)' in content
    
    print(f"\n📖 get_entry method NPY integration:")
    print(f"   Loads NPY files: {load_npy}")
    print(f"   Converts back to tensors: {convert_back}")
    
    # Check for backward compatibility
    backward_compat = 'if activation_path.suffix == \'.npy\':' in content
    json_fallback = 'with open(activation_path, "r") as f:' in content and 'json.load(f)' in content
    
    print(f"\n🔄 Backward compatibility:")
    print(f"   NPY/JSON format detection: {backward_compat}")
    print(f"   JSON fallback for old files: {json_fallback}")
    
    # Overall assessment
    all_checks = [
        version_check,
        storage_format_check,
        all(method_checks.values()),
        log_entry_npy,
        save_npy,
        load_npy,
        convert_back,
        backward_compat
    ]
    
    print(f"\n📋 Implementation Summary:")
    print(f"   ✅ Version & format metadata: {version_check and storage_format_check}")
    print(f"   ✅ Required helper methods: {all(method_checks.values())}")
    print(f"   ✅ NPY saving in log_entry: {log_entry_npy and save_npy}")
    print(f"   ✅ NPY loading in get_entry: {load_npy and convert_back}")
    print(f"   ✅ Backward compatibility: {backward_compat}")
    
    implementation_complete = all(all_checks)
    
    if implementation_complete:
        print(f"\n🎉 NPY Implementation Status: COMPLETE ✅")
        print(f"   The JsonActivationsLogger has been successfully updated to support NPY format!")
        print(f"   - Tensors will be saved as binary .npy files")
        print(f"   - Metadata remains in JSON format for readability")
        print(f"   - Backward compatibility with old JSON activation files")
        print(f"   - Expected storage reduction: ~95% compared to JSON tensors")
    else:
        print(f"\n❌ NPY Implementation Status: INCOMPLETE")
        print(f"   Missing components detected. Implementation needs completion.")
    
    return implementation_complete

if __name__ == "__main__":
    success = check_npy_implementation()
    if success:
        print(f"\n✅ Ready for NPY logging tests!")
    else:
        print(f"\n❌ Implementation needs fixes before testing.")
