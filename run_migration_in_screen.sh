#!/bin/bash

# HalluLens Migration Runner with Screen Session
# Runs migration in a detachable screen session to prevent interruption
# Author: Augment Agent
# Date: 2024-11-14

# Configuration
SESSION_NAME="hallulens_migration"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Show usage
usage() {
    echo "Usage: $0 <command> [phase]"
    echo ""
    echo "Commands:"
    echo "  start [phase]  - Start migration (full migration or specific phase)"
    echo "  attach         - Attach to existing migration session"
    echo "  status         - Show migration status"
    echo "  stop           - Stop migration session"
    echo "  logs           - Show recent migration logs"
    echo ""
    echo "Examples:"
    echo "  $0 start        # Start full migration"
    echo "  $0 start 1      # Start phase 1 only"
    echo "  $0 attach       # Attach to running session"
    echo "  $0 status       # Check migration status"
    echo ""
    exit 1
}

# Check if screen session exists
session_exists() {
    screen -list | grep -q "$SESSION_NAME"
}

# Start migration in screen session
start_migration() {
    local phase="$1"
    
    if session_exists; then
        warning "Migration session already exists. Use 'attach' to connect or 'stop' to terminate."
        return 1
    fi
    
    log "Starting migration in screen session: $SESSION_NAME"
    
    if [[ -n "$phase" ]]; then
        log "Running phase $phase migration..."
        screen -dmS "$SESSION_NAME" bash -c "cd '$SCRIPT_DIR' && bash migrate_phase.sh '$phase'; echo 'Migration completed. Press any key to exit.'; read"
    else
        log "Running full migration..."
        screen -dmS "$SESSION_NAME" bash -c "cd '$SCRIPT_DIR' && bash migrate_to_shared_storage.sh; echo 'Migration completed. Press any key to exit.'; read"
    fi
    
    success "Migration started in screen session. Use '$0 attach' to monitor progress."
    log "You can safely disconnect from SSH - the migration will continue running."
}

# Attach to existing session
attach_session() {
    if ! session_exists; then
        error "No migration session found. Use 'start' to begin migration."
        return 1
    fi
    
    log "Attaching to migration session..."
    log "Use Ctrl+A, D to detach from session without stopping migration"
    screen -r "$SESSION_NAME"
}

# Stop migration session
stop_session() {
    if ! session_exists; then
        warning "No migration session found."
        return 0
    fi
    
    log "Stopping migration session..."
    screen -S "$SESSION_NAME" -X quit
    success "Migration session stopped."
}

# Show migration status
show_status() {
    log "Checking migration status..."
    
    if session_exists; then
        success "Migration session is running: $SESSION_NAME"
        log "Use '$0 attach' to connect to the session"
    else
        log "No active migration session found."
    fi
    
    # Show file status if migrate_phase.sh exists
    if [[ -f "$SCRIPT_DIR/migrate_phase.sh" ]]; then
        log "Current file migration status:"
        bash "$SCRIPT_DIR/migrate_phase.sh" status
    fi
}

# Show recent logs
show_logs() {
    log "Recent migration logs:"
    
    # Find most recent log file
    local latest_log=$(ls -t migration_*.log 2>/dev/null | head -1)
    
    if [[ -n "$latest_log" && -f "$latest_log" ]]; then
        log "Showing last 50 lines from $latest_log:"
        echo "----------------------------------------"
        tail -50 "$latest_log"
        echo "----------------------------------------"
    else
        warning "No migration log files found."
    fi
}

# Main function
main() {
    if [[ $# -eq 0 ]]; then
        usage
    fi
    
    local command="$1"
    local phase="$2"
    
    case "$command" in
        "start")
            start_migration "$phase"
            ;;
        "attach")
            attach_session
            ;;
        "status")
            show_status
            ;;
        "stop")
            stop_session
            ;;
        "logs")
            show_logs
            ;;
        *)
            error "Invalid command: $command"
            usage
            ;;
    esac
}

# Check if screen is installed
if ! command -v screen &> /dev/null; then
    error "Screen is not installed. Please install it first:"
    error "  sudo yum install screen  # On RHEL/CentOS"
    error "  sudo apt install screen  # On Ubuntu/Debian"
    exit 1
fi

main "$@"
