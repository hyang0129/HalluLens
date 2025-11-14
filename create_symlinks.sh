#!/bin/bash

# HalluLens Symlink Creation Script
# Creates symbolic links after verifying copies are successful
# ONLY run this after copy verification is complete
# Author: Augment Agent
# Date: 2024-11-14

set -e

# Configuration
SOURCE_DIR="/home/hy3134/notebook_llm/HalluLens"
TARGET_DIR="/shared/rc/llm-hd/HalluLens"
LOG_FILE="symlink_creation_$(date +%Y%m%d_%H%M%S).log"

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

# Create symlink for a single item
create_symlink() {
    local item="$1"
    local size="$2"
    
    log "Processing $item ($size)..."
    
    # Check if source exists
    if [[ ! -e "$SOURCE_DIR/$item" ]]; then
        warning "$item not found in source, skipping..."
        return 0
    fi
    
    # Check if target copy exists
    if [[ ! -e "$TARGET_DIR/$item" ]]; then
        error "$item not found in target directory - copy first!"
        return 1
    fi
    
    # Check if already a symlink
    if [[ -L "$SOURCE_DIR/$item" ]]; then
        success "$item is already a symlink"
        return 0
    fi
    
    # Verify sizes match before creating symlink
    local source_size=$(du -sb "$SOURCE_DIR/$item" | cut -f1)
    local target_size=$(du -sb "$TARGET_DIR/$item" | cut -f1)
    
    if [[ $target_size -ne $source_size ]]; then
        error "Size mismatch for $item (source: $source_size, target: $target_size)"
        error "Cannot create symlink - copy verification failed!"
        return 1
    fi
    
    # Create backup of original (just in case)
    log "Creating backup of original $item..."
    if mv "$SOURCE_DIR/$item" "$SOURCE_DIR/${item}.backup_$(date +%Y%m%d_%H%M%S)"; then
        log "Backup created successfully"
    else
        error "Failed to create backup of $item"
        return 1
    fi
    
    # Create symbolic link
    log "Creating symbolic link for $item..."
    if ln -s "$TARGET_DIR/$item" "$SOURCE_DIR/$item"; then
        success "Symbolic link created for $item"
        
        # Test the symlink
        if [[ -e "$SOURCE_DIR/$item" ]]; then
            success "Symlink verification passed for $item"
        else
            error "Symlink verification failed for $item"
            return 1
        fi
    else
        error "Failed to create symbolic link for $item"
        return 1
    fi
}

# Show usage
usage() {
    echo "Usage: $0 [item_name|all]"
    echo ""
    echo "Creates symbolic links to replace original files with links to copied files"
    echo ""
    echo "Options:"
    echo "  all                    - Create symlinks for all copied items"
    echo "  <item_name>           - Create symlink for specific item"
    echo "  --list                - List items available for symlinking"
    echo ""
    echo "Examples:"
    echo "  $0 all                           # Create all symlinks"
    echo "  $0 goodwiki_json_2              # Create symlink for specific item"
    echo "  $0 --list                       # List available items"
    echo ""
    warning "IMPORTANT: Only run this AFTER verifying copies are successful!"
    exit 1
}

# List available items
list_items() {
    log "Items available for symlinking:"
    
    cd "$SOURCE_DIR"
    
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
        
        if [[ -L "$item" ]]; then
            echo -e "${GREEN}✓${NC} $item ($size) - Already symlinked"
        elif [[ -e "$TARGET_DIR/$item" && -e "$item" ]]; then
            echo -e "${YELLOW}○${NC} $item ($size) - Ready for symlinking"
        elif [[ -e "$TARGET_DIR/$item" ]]; then
            echo -e "${BLUE}?${NC} $item ($size) - Copied but original missing"
        else
            echo -e "${RED}✗${NC} $item ($size) - Not copied yet"
        fi
    done
}

# Main function
main() {
    if [[ $# -eq 0 ]]; then
        usage
    fi
    
    local target="$1"
    
    case "$target" in
        "--list")
            list_items
            ;;
        "all")
            log "=== Creating symlinks for all copied items ==="
            warning "This will replace original files with symbolic links!"
            read -p "Are you sure? (yes/no): " confirm
            
            if [[ "$confirm" != "yes" ]]; then
                log "Operation cancelled by user"
                exit 0
            fi
            
            create_symlink "goodwiki_json_2" "470G"
            create_symlink "goodwiki_json" "470G"
            create_symlink "triviiqa_json" "447G"
            create_symlink "Meta-Llama-3.1-70B-Instruct-IQ3_M.gguf" "30G"
            create_symlink "Llama-3.3-70B-Instruct-IQ3_M.gguf" "30G"
            create_symlink "Meta-Llama-3-8B-Instruct-Q6_K.gguf" "6.2G"
            create_symlink "checkpoints" "31G"
            create_symlink "npyloggingtestdir_100samples" "7.3G"
            create_symlink "halu" "9.5G"
            create_symlink "data" "5.5G"
            
            success "Symlink creation completed!"
            ;;
        *)
            log "=== Creating symlink for $target ==="
            create_symlink "$target" "unknown"
            ;;
    esac
    
    log "Symlink creation log saved to: $LOG_FILE"
}

main "$@"
