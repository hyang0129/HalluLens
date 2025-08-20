#!/bin/bash

# Remote GPU Connection Script for HalluLens Development
# This script automatically finds and connects to the GPU node running job "nb8887new"

# Configuration
SSH_USER="hy3134"
SSH_HOST="sporcsubmit.rc.rit.edu"
SSH_KEY="C:\Users\HongM\OneDrive\Documents\ssh\.ssh\rit_rc_key"
SSH_PASSPHRASE="123"
JOB_NAME="nb8887new"
PROJECT_DIR="notebook_llm/HalluLens"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if we're on Windows (Git Bash/MINGW)
is_windows() {
    [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]
}

echo -e "${BLUE}🚀 HalluLens Remote GPU Connection Script${NC}"
echo -e "${BLUE}===========================================${NC}"

# Function to setup SSH agent with automatic passphrase
setup_ssh_agent() {
    echo -e "${YELLOW}🔐 Setting up SSH agent with automatic passphrase...${NC}"

    # Check if ssh-agent is already running
    if [ -z "$SSH_AUTH_SOCK" ]; then
        echo -e "${BLUE}🚀 Starting SSH agent...${NC}"
        eval $(ssh-agent -s) > /dev/null
    fi

    # Check if key is already loaded
    if ! ssh-add -l 2>/dev/null | grep -q "$(basename "$SSH_KEY")"; then
        echo -e "${BLUE}🔑 Adding SSH key to agent...${NC}"

        # Use expect if available, otherwise use sshpass or manual input
        if command -v expect &> /dev/null; then
            expect -c "
                spawn ssh-add \"$SSH_KEY\"
                expect \"Enter passphrase for*\" {
                    send \"$SSH_PASSPHRASE\r\"
                    exp_continue
                }
                expect eof
            " > /dev/null 2>&1
        elif command -v sshpass &> /dev/null; then
            echo "$SSH_PASSPHRASE" | SSH_ASKPASS_REQUIRE=force SSH_ASKPASS="echo $SSH_PASSPHRASE" ssh-add "$SSH_KEY" > /dev/null 2>&1
        else
            # Fallback: create a temporary script for SSH_ASKPASS
            TEMP_ASKPASS=$(mktemp)
            echo "#!/bin/bash" > "$TEMP_ASKPASS"
            echo "echo '$SSH_PASSPHRASE'" >> "$TEMP_ASKPASS"
            chmod +x "$TEMP_ASKPASS"

            SSH_ASKPASS="$TEMP_ASKPASS" SSH_ASKPASS_REQUIRE=force ssh-add "$SSH_KEY" > /dev/null 2>&1
            rm -f "$TEMP_ASKPASS"
        fi

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ SSH key added successfully${NC}"
        else
            echo -e "${YELLOW}⚠️  SSH key addition may have failed, but continuing...${NC}"
        fi
    else
        echo -e "${GREEN}✅ SSH key already loaded in agent${NC}"
    fi
}

# Function to execute SSH command and get output
ssh_execute() {
    local command="$1"
    if is_windows; then
        # On Windows/Git Bash, use the Windows-style path
        ssh -i "$SSH_KEY" "$SSH_USER@$SSH_HOST" "$command" 2>/dev/null
    else
        # On Linux/Mac, convert to Unix-style path
        local unix_key_path=$(echo "$SSH_KEY" | sed 's|C:|/c|' | sed 's|\\|/|g')
        ssh -i "$unix_key_path" "$SSH_USER@$SSH_HOST" "$command" 2>/dev/null
    fi
}

# Function to connect interactively
ssh_connect() {
    local jobid="$1"
    echo -e "${GREEN}🔗 Connecting to GPU node via job $jobid...${NC}"

    if is_windows; then
        ssh -i "$SSH_KEY" "$SSH_USER@$SSH_HOST" -t "srun --jobid=$jobid --pty bash -c 'cd $PROJECT_DIR && source halu/bin/activate && exec bash'"
    else
        local unix_key_path=$(echo "$SSH_KEY" | sed 's|C:|/c|' | sed 's|\\|/|g')
        ssh -i "$unix_key_path" "$SSH_USER@$SSH_HOST" -t "srun --jobid=$jobid --pty bash -c 'cd $PROJECT_DIR && source halu/bin/activate && exec bash'"
    fi
}

# Setup SSH agent with automatic passphrase
setup_ssh_agent

echo -e "${YELLOW}🔍 Connecting to login node and searching for job '$JOB_NAME'...${NC}"

# Get job information
JOB_INFO=$(ssh_execute "squeue -u $SSH_USER --name=$JOB_NAME --format='%i,%N,%T,%M' --noheader")

if [ -z "$JOB_INFO" ]; then
    echo -e "${RED}❌ Error: No running job found with name '$JOB_NAME'${NC}"
    echo -e "${YELLOW}💡 Available jobs for user $SSH_USER:${NC}"
    ssh_execute "squeue -u $SSH_USER --format='%i,%j,%N,%T,%M' --noheader" | while IFS=',' read -r jobid jobname node state time; do
        echo -e "   Job ID: $jobid, Name: $jobname, Node: $node, State: $state, Time: $time"
    done
    echo ""
    echo -e "${YELLOW}🔧 To submit a new GPU job, you can use:${NC}"
    echo -e "   sbatch --partition=tier3 --gres=gpu:1 --time=24:00:00 --job-name=$JOB_NAME your_script.sh"
    exit 1
fi

# Parse job information
IFS=',' read -r JOBID NODE STATE TIME <<< "$JOB_INFO"

echo -e "${GREEN}✅ Found job '$JOB_NAME':${NC}"
echo -e "   Job ID: ${BLUE}$JOBID${NC}"
echo -e "   Node: ${BLUE}$NODE${NC}"
echo -e "   State: ${BLUE}$STATE${NC}"
echo -e "   Runtime: ${BLUE}$TIME${NC}"

# Check if job is running
if [ "$STATE" != "RUNNING" ] && [ "$STATE" != "R" ]; then
    echo -e "${RED}❌ Error: Job is not in RUNNING state (current state: $STATE)${NC}"
    echo -e "${YELLOW}💡 Please wait for the job to start or submit a new one.${NC}"
    exit 1
fi

echo -e "${GREEN}🎯 Job is running! Connecting to GPU node $NODE...${NC}"
echo -e "${BLUE}📁 Will automatically navigate to: $PROJECT_DIR${NC}"
echo ""

# Connect to the GPU node
ssh_connect "$JOBID"

# If we get here, the connection was closed
echo ""
echo -e "${YELLOW}🔌 Connection to GPU node closed.${NC}"
echo -e "${GREEN}✅ Session ended successfully.${NC}"
