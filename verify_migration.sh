#!/bin/bash

# HalluLens Migration Verification Script
# Verifies that migration completed successfully and all symlinks work
# Author: Augment Agent
# Date: 2024-11-14

set -e

# Configuration
SOURCE_DIR="/home/hy3134/notebook_llm/HalluLens"
TARGET_DIR="/shared/rc/llm-hd/HalluLens"
LOG_FILE="verification_$(date +%Y%m%d_%H%M%S).log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Verify a single item (supports both copy and symlink verification)
verify_item() {
    local item="$1"
    local expected_size="$2"

    log "Verifying $item..."

    # Check if target copy exists
    if [[ ! -e "$TARGET_DIR/$item" ]]; then
        error "$item not found in target directory - not copied yet"
        return 1
    fi

    # Check source status
    if [[ -L "$SOURCE_DIR/$item" ]]; then
        # It's a symlink - verify it works
        local link_target=$(readlink "$SOURCE_DIR/$item")
        if [[ "$link_target" != "$TARGET_DIR/$item" ]]; then
            error "$item symlink points to wrong location: $link_target"
            return 1
        fi

        if [[ ! -e "$SOURCE_DIR/$item" ]]; then
            error "$item symlink is broken"
            return 1
        fi

        success "$item symlink verification passed"
        return 0

    elif [[ -e "$SOURCE_DIR/$item" ]]; then
        # Original file exists - verify copy integrity
        local source_size=$(du -sb "$SOURCE_DIR/$item" | cut -f1)
        local target_size=$(du -sb "$TARGET_DIR/$item" | cut -f1)

        if [[ $target_size -eq $source_size ]]; then
            success "$item copy verification passed (both files exist, sizes match)"
            return 0
        else
            error "$item size mismatch (source: $source_size, target: $target_size)"
            return 1
        fi
    else
        error "$item not found in source directory"
        return 1
    fi
}

# Check disk space savings
check_space_savings() {
    log "Checking disk space savings..."
    
    local home_usage=$(df -BG "$SOURCE_DIR" | awk 'NR==2 {print $3}' | sed 's/G//')
    local home_avail=$(df -BG "$SOURCE_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
    local target_usage=$(df -BG "/shared/rc/llm-hd" | awk 'NR==2 {print $3}' | sed 's/G//')
    
    log "Home directory usage: ${home_usage}G (${home_avail}G available)"
    log "Target storage usage: ${target_usage}G"
    
    # Calculate percentage improvement
    local total_space=$(df -BG "$SOURCE_DIR" | awk 'NR==2 {print $2}' | sed 's/G//')
    local usage_percent=$((home_usage * 100 / total_space))
    
    log "Home directory usage: ${usage_percent}%"
    
    if [[ $usage_percent -lt 80 ]]; then
        success "Significant space savings achieved!"
    elif [[ $usage_percent -lt 90 ]]; then
        success "Good space savings achieved"
    else
        warning "Limited space savings - consider migrating more files"
    fi
}

# Test basic functionality
test_functionality() {
    log "Testing basic functionality..."
    
    cd "$SOURCE_DIR"
    
    # Test that we can list directories
    if ls -la goodwiki_json_2/ >/dev/null 2>&1; then
        success "goodwiki_json_2 directory accessible"
    else
        error "Cannot access goodwiki_json_2 directory"
    fi
    
    if ls -la goodwiki_json/ >/dev/null 2>&1; then
        success "goodwiki_json directory accessible"
    else
        error "Cannot access goodwiki_json directory"
    fi
    
    if ls -la triviiqa_json/ >/dev/null 2>&1; then
        success "triviiqa_json directory accessible"
    else
        error "Cannot access triviiqa_json directory"
    fi
    
    # Test model files if they exist
    if [[ -L "Meta-Llama-3.1-70B-Instruct-IQ3_M.gguf" ]]; then
        if [[ -r "Meta-Llama-3.1-70B-Instruct-IQ3_M.gguf" ]]; then
            success "Meta-Llama-3.1-70B model file accessible"
        else
            error "Meta-Llama-3.1-70B model file not readable"
        fi
    fi
    
    # Test Python environment if migrated
    if [[ -L "halu" ]]; then
        if [[ -d "halu/bin" ]]; then
            success "Python environment structure intact"
        else
            error "Python environment structure damaged"
        fi
    fi
}

# Generate migration report
generate_report() {
    log "=== Migration Verification Report ==="
    
    local total_items=0
    local migrated_items=0
    local failed_items=0
    
    # List of items to check
    local items=(
        "goodwiki_json_2:470G"
        "goodwiki_json:470G"
        "triviiqa_json:447G"
        "Meta-Llama-3.1-70B-Instruct-IQ3_M.gguf:30G"
        "Llama-3.3-70B-Instruct-IQ3_M.gguf:30G"
        "Meta-Llama-3-8B-Instruct-Q6_K.gguf:6.2G"
        "checkpoints:31G"
        "npyloggingtestdir_100samples:7.3G"
        "halu:9.5G"
        "data:5.5G"
    )
    
    for item_info in "${items[@]}"; do
        IFS=':' read -r item size <<< "$item_info"
        total_items=$((total_items + 1))
        
        if verify_item "$item" "$size"; then
            migrated_items=$((migrated_items + 1))
        else
            failed_items=$((failed_items + 1))
        fi
    done
    
    log "=== Summary ==="
    log "Total items checked: $total_items"
    log "Successfully migrated: $migrated_items"
    log "Failed/Not migrated: $failed_items"
    
    if [[ $failed_items -eq 0 ]]; then
        success "All items successfully migrated and verified!"
    else
        warning "$failed_items items failed verification"
    fi
    
    check_space_savings
    test_functionality
    
    log "Verification report saved to: $LOG_FILE"
}

# Main function
main() {
    log "=== HalluLens Migration Verification ==="
    
    if [[ ! -d "$SOURCE_DIR" ]]; then
        error "Source directory not found: $SOURCE_DIR"
        exit 1
    fi
    
    if [[ ! -d "$TARGET_DIR" ]]; then
        error "Target directory not found: $TARGET_DIR"
        exit 1
    fi
    
    generate_report
    
    success "Verification completed. Check $LOG_FILE for detailed results."
}

main "$@"
