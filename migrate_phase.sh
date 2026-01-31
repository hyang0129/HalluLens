#!/bin/bash

# HalluLens Phased Migration Script
# Allows running migration in individual phases for better control
# Author: Augment Agent
# Date: 2024-11-14

set -e

# Configuration
SOURCE_DIR="/home/hy3134/notebook_llm/HalluLens"
TARGET_DIR="/shared/rc/llm-hd/HalluLens"
LOG_FILE="migration_phase_$(date +%Y%m%d_%H%M%S).log"

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

# Copy a single item (SAFE COPY - originals preserved)
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
        success "$item already copied to target storage"
        return 0
    fi

    # Create target directory
    mkdir -p "$TARGET_DIR"

    # Use rsync for reliable transfer
    log "Copying $item with rsync..."
    if rsync -avh --progress "$SOURCE_DIR/$item" "$TARGET_DIR/" 2>&1 | tee -a "$LOG_FILE"; then
        success "Successfully copied $item"

        # Verify copy exists and size matches
        if [[ -e "$TARGET_DIR/$item" ]]; then
            local source_size=$(du -sb "$SOURCE_DIR/$item" | cut -f1)
            local target_size=$(du -sb "$TARGET_DIR/$item" | cut -f1)

            if [[ $target_size -eq $source_size ]]; then
                success "Copy verification passed for $item"
                log "Original preserved: $SOURCE_DIR/$item"
                log "Copy available: $TARGET_DIR/$item"
            else
                error "Size mismatch for $item"
                return 1
            fi
        else
            error "Copy verification failed for $item"
            return 1
        fi
    else
        error "Failed to copy $item"
        return 1
    fi
}

# Show usage
usage() {
    echo "Usage: $0 <phase_number>"
    echo ""
    echo "Available phases:"
    echo "  1 - Large Data Files (goodwiki_json_2, goodwiki_json, triviiqa_json)"
    echo "  2 - Model Files (Llama models)"
    echo "  3 - Training Data (checkpoints, npyloggingtestdir_100samples)"
    echo "  4 - Python Environment (halu)"
    echo "  5 - Remaining Data (data directory)"
    echo "  status - Show current migration status"
    echo ""
    exit 1
}

# Show migration status
show_status() {
    log "=== Migration Status ==="
    
    cd "$SOURCE_DIR"
    
    # Check each item
    local items=(
        "goodwiki_json_2:470G:Phase1"
        "goodwiki_json:470G:Phase1"
        "triviiqa_json:447G:Phase1"
        "Meta-Llama-3.1-70B-Instruct-IQ3_M.gguf:30G:Phase2"
        "Llama-3.3-70B-Instruct-IQ3_M.gguf:30G:Phase2"
        "Meta-Llama-3-8B-Instruct-Q6_K.gguf:6.2G:Phase2"
        "checkpoints:31G:Phase3"
        "npyloggingtestdir_100samples:7.3G:Phase3"
        "halu:9.5G:Phase4"
        "data:5.5G:Phase5"
    )
    
    for item_info in "${items[@]}"; do
        IFS=':' read -r item size phase <<< "$item_info"
        
        if [[ -e "$TARGET_DIR/$item" ]]; then
            echo -e "${GREEN}✓${NC} $phase: $item ($size) - COPIED"
        elif [[ -e "$item" ]]; then
            echo -e "${YELLOW}○${NC} $phase: $item ($size) - NOT COPIED"
        else
            echo -e "${RED}✗${NC} $phase: $item ($size) - NOT FOUND"
        fi
    done
    
    echo ""
    log "Current disk usage:"
    df -h "$SOURCE_DIR" | head -2
    df -h "/shared/rc/llm-hd" | head -2
}

# Main function
main() {
    if [[ $# -eq 0 ]]; then
        usage
    fi
    
    local phase="$1"
    
    case "$phase" in
        "1")
            log "=== Phase 1: Large Data Files ==="
            migrate_item "goodwiki_json_2" "470G"
            migrate_item "goodwiki_json" "470G"
            migrate_item "triviiqa_json" "447G"
            ;;
        "2")
            log "=== Phase 2: Model Files ==="
            migrate_item "Meta-Llama-3.1-70B-Instruct-IQ3_M.gguf" "30G"
            migrate_item "Llama-3.3-70B-Instruct-IQ3_M.gguf" "30G"
            migrate_item "Meta-Llama-3-8B-Instruct-Q6_K.gguf" "6.2G"
            ;;
        "3")
            log "=== Phase 3: Training Data and Checkpoints ==="
            migrate_item "checkpoints" "31G"
            migrate_item "npyloggingtestdir_100samples" "7.3G"
            ;;
        "4")
            log "=== Phase 4: Python Environment ==="
            migrate_item "halu" "9.5G"
            ;;
        "5")
            log "=== Phase 5: Remaining Data ==="
            migrate_item "data" "5.5G"
            ;;
        "status")
            show_status
            ;;
        *)
            error "Invalid phase: $phase"
            usage
            ;;
    esac
    
    if [[ "$phase" != "status" ]]; then
        success "Phase $phase completed successfully!"
        show_status
    fi
}

main "$@"
