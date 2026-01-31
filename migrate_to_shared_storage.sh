#!/bin/bash

# HalluLens Storage Migration Script
# Migrates files from home directory to /shared/rc/llm-hd/HalluLens
# Author: Augment Agent
# Date: 2024-11-14

set -e  # Exit on any error

# Configuration
SOURCE_DIR="/home/hy3134/notebook_llm/HalluLens"
TARGET_DIR="/shared/rc/llm-hd/HalluLens"
LOG_FILE="migration_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
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

# Check if we're in the right directory
check_environment() {
    log "Checking environment..."
    
    if [[ ! -d "$SOURCE_DIR" ]]; then
        error "Source directory $SOURCE_DIR not found!"
        exit 1
    fi
    
    if [[ ! -d "/shared/rc/llm-hd" ]]; then
        error "Target storage /shared/rc/llm-hd not mounted!"
        exit 1
    fi
    
    # Create target directory if it doesn't exist
    mkdir -p "$TARGET_DIR"
    
    success "Environment check passed"
}

# Check available space
check_space() {
    log "Checking available space..."
    
    local source_used=$(df -BG "$SOURCE_DIR" | awk 'NR==2 {print $3}' | sed 's/G//')
    local target_avail=$(df -BG "/shared/rc/llm-hd" | awk 'NR==2 {print $4}' | sed 's/G//')
    
    log "Source directory usage: ${source_used}G"
    log "Target available space: ${target_avail}G"
    
    if [[ $target_avail -lt $source_used ]]; then
        error "Insufficient space on target storage!"
        exit 1
    fi
    
    success "Space check passed"
}

# Copy a single directory/file with rsync (SAFE COPY - originals preserved)
migrate_item() {
    local item="$1"
    local size="$2"

    log "Starting COPY of $item ($size)..."

    if [[ ! -e "$SOURCE_DIR/$item" ]]; then
        warning "$item not found, skipping..."
        return 0
    fi

    # Check if already copied
    if [[ -e "$TARGET_DIR/$item" ]]; then
        warning "$item already exists in target, skipping copy..."
        return 0
    fi

    # Use rsync for reliable transfer with progress
    if rsync -avh --progress "$SOURCE_DIR/$item" "$TARGET_DIR/" 2>&1 | tee -a "$LOG_FILE"; then
        success "Successfully copied $item to target storage"

        # Verify the copy exists and has reasonable size
        if [[ -e "$TARGET_DIR/$item" ]]; then
            local source_size=$(du -sb "$SOURCE_DIR/$item" | cut -f1)
            local target_size=$(du -sb "$TARGET_DIR/$item" | cut -f1)

            if [[ $target_size -eq $source_size ]]; then
                success "Copy verification passed for $item (size: $target_size bytes)"
                log "Original file preserved at: $SOURCE_DIR/$item"
                log "Copy available at: $TARGET_DIR/$item"
            else
                error "Size mismatch for $item (source: $source_size, target: $target_size)"
                return 1
            fi
        else
            error "Copy verification failed for $item - file not found in target"
            return 1
        fi
    else
        error "Failed to copy $item"
        return 1
    fi
}

# Main migration function
main() {
    log "=== HalluLens Storage Migration Started ==="
    
    check_environment
    check_space
    
    cd "$SOURCE_DIR"
    
    # Phase 1: Large Data Files (1.4TB)
    log "=== Phase 1: Large Data Files ==="
    migrate_item "goodwiki_json_2" "470G"
    migrate_item "goodwiki_json" "470G"
    migrate_item "triviiqa_json" "447G"
    
    # Phase 2: Model Files (66GB)
    log "=== Phase 2: Model Files ==="
    migrate_item "Meta-Llama-3.1-70B-Instruct-IQ3_M.gguf" "30G"
    migrate_item "Llama-3.3-70B-Instruct-IQ3_M.gguf" "30G"
    migrate_item "Meta-Llama-3-8B-Instruct-Q6_K.gguf" "6.2G"
    
    # Phase 3: Training Data and Checkpoints (38.5GB)
    log "=== Phase 3: Training Data and Checkpoints ==="
    migrate_item "checkpoints" "31G"
    migrate_item "npyloggingtestdir_100samples" "7.3G"
    
    # Phase 4: Python Environment (9.5GB)
    log "=== Phase 4: Python Environment ==="
    migrate_item "halu" "9.5G"
    
    # Phase 5: Remaining Data (5.5GB)
    log "=== Phase 5: Remaining Data ==="
    migrate_item "data" "5.5G"
    
    log "=== COPY Operation Complete ==="
    success "All files successfully COPIED to $TARGET_DIR"
    warning "IMPORTANT: Original files are preserved in $SOURCE_DIR"
    warning "Use create_symlinks.sh to replace originals with symlinks after verification"

    # Final verification
    log "Running final verification..."
    df -h "$SOURCE_DIR"
    df -h "/shared/rc/llm-hd"

    success "Copy operation completed successfully! Check $LOG_FILE for details."
    log "Next steps:"
    log "1. Run verify_migration.sh to verify copies"
    log "2. Run create_symlinks.sh to replace originals with symlinks"
    log "3. Run cleanup_originals.sh to remove original files (AFTER verification)"
}

# Run main function
main "$@"
