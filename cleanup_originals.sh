#!/bin/bash

# HalluLens Original Files Cleanup Script
# Removes backup files created during symlink creation
# ONLY run this after everything is verified to work correctly
# Author: Augment Agent
# Date: 2024-11-14

set -e

# Configuration
SOURCE_DIR="/home/hy3134/notebook_llm/HalluLens"
LOG_FILE="cleanup_$(date +%Y%m%d_%H%M%S).log"

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

# Find and list backup files
list_backups() {
    log "Searching for backup files in $SOURCE_DIR..."
    
    cd "$SOURCE_DIR"
    
    local backup_files=($(find . -name "*.backup_*" -type f -o -name "*.backup_*" -type d))
    
    if [[ ${#backup_files[@]} -eq 0 ]]; then
        log "No backup files found"
        return 0
    fi
    
    log "Found ${#backup_files[@]} backup files/directories:"
    
    local total_size=0
    for backup in "${backup_files[@]}"; do
        local size=$(du -sh "$backup" | cut -f1)
        local size_bytes=$(du -sb "$backup" | cut -f1)
        total_size=$((total_size + size_bytes))
        echo -e "  ${YELLOW}$backup${NC} ($size)"
    done
    
    local total_size_human=$(echo $total_size | awk '{
        if ($1 > 1024^3) printf "%.1fG", $1/1024^3
        else if ($1 > 1024^2) printf "%.1fM", $1/1024^2
        else if ($1 > 1024) printf "%.1fK", $1/1024
        else printf "%dB", $1
    }')
    
    log "Total backup size: $total_size_human"
    
    return ${#backup_files[@]}
}

# Remove backup files
cleanup_backups() {
    log "Starting cleanup of backup files..."
    
    cd "$SOURCE_DIR"
    
    local backup_files=($(find . -name "*.backup_*" -type f -o -name "*.backup_*" -type d))
    
    if [[ ${#backup_files[@]} -eq 0 ]]; then
        success "No backup files to clean up"
        return 0
    fi
    
    local removed_count=0
    local failed_count=0
    
    for backup in "${backup_files[@]}"; do
        log "Removing $backup..."
        
        if rm -rf "$backup"; then
            success "Removed $backup"
            removed_count=$((removed_count + 1))
        else
            error "Failed to remove $backup"
            failed_count=$((failed_count + 1))
        fi
    done
    
    log "Cleanup summary:"
    log "  Removed: $removed_count files/directories"
    log "  Failed: $failed_count files/directories"
    
    if [[ $failed_count -eq 0 ]]; then
        success "All backup files cleaned up successfully"
    else
        warning "$failed_count backup files could not be removed"
    fi
}

# Verify symlinks are working before cleanup
verify_symlinks() {
    log "Verifying symlinks are working before cleanup..."
    
    cd "$SOURCE_DIR"
    
    local items=(
        "goodwiki_json_2"
        "goodwiki_json"
        "triviiqa_json"
        "Meta-Llama-3.1-70B-Instruct-IQ3_M.gguf"
        "Llama-3.3-70B-Instruct-IQ3_M.gguf"
        "Meta-Llama-3-8B-Instruct-Q6_K.gguf"
        "checkpoints"
        "npyloggingtestdir_100samples"
        "halu"
        "data"
    )
    
    local working_symlinks=0
    local total_symlinks=0
    
    for item in "${items[@]}"; do
        if [[ -L "$item" ]]; then
            total_symlinks=$((total_symlinks + 1))
            
            if [[ -e "$item" ]]; then
                working_symlinks=$((working_symlinks + 1))
                log "✓ $item symlink working"
            else
                error "✗ $item symlink broken"
            fi
        fi
    done
    
    if [[ $total_symlinks -eq 0 ]]; then
        warning "No symlinks found - cleanup may not be needed"
        return 1
    fi
    
    if [[ $working_symlinks -eq $total_symlinks ]]; then
        success "All $total_symlinks symlinks are working correctly"
        return 0
    else
        error "Only $working_symlinks out of $total_symlinks symlinks are working"
        error "Do not proceed with cleanup until all symlinks are fixed!"
        return 1
    fi
}

# Show usage
usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  list     - List backup files that would be removed"
    echo "  cleanup  - Remove backup files (after verification)"
    echo "  verify   - Verify symlinks are working"
    echo ""
    echo "Examples:"
    echo "  $0 list      # Show what would be cleaned up"
    echo "  $0 verify    # Check symlinks are working"
    echo "  $0 cleanup   # Remove backup files"
    echo ""
    warning "IMPORTANT: Only run cleanup after verifying everything works!"
    exit 1
}

# Main function
main() {
    if [[ $# -eq 0 ]]; then
        usage
    fi
    
    local command="$1"
    
    case "$command" in
        "list")
            list_backups
            ;;
        "verify")
            verify_symlinks
            ;;
        "cleanup")
            log "=== Starting backup cleanup process ==="
            
            # First verify symlinks
            if ! verify_symlinks; then
                error "Symlink verification failed - aborting cleanup"
                exit 1
            fi
            
            # List what will be removed
            if ! list_backups; then
                success "No backup files to clean up"
                exit 0
            fi
            
            # Confirm with user
            warning "This will permanently delete backup files!"
            warning "Make sure all symlinks are working correctly first!"
            read -p "Are you absolutely sure? (yes/no): " confirm
            
            if [[ "$confirm" != "yes" ]]; then
                log "Cleanup cancelled by user"
                exit 0
            fi
            
            # Perform cleanup
            cleanup_backups
            
            # Final verification
            log "Running final disk space check..."
            df -h "$SOURCE_DIR"
            
            success "Cleanup completed! Check $LOG_FILE for details."
            ;;
        *)
            error "Invalid command: $command"
            usage
            ;;
    esac
}

main "$@"
