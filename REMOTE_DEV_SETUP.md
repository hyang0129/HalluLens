# Remote Development Environment Setup for HalluLens

This document provides comprehensive instructions for setting up and using the remote development environment for the HalluLens project.

## 🚀 Quick Start (TL;DR)
**Recommended approach:** Use the automated `connect_gpu.sh` script which handles SSH agent setup automatically, eliminating the need to repeatedly enter your SSH key passphrase.

```bash
# One command to connect to GPU node with automatic SSH agent setup
./connect_gpu.sh
```

This script will:
- ✅ Set up SSH agent and handle passphrase automatically
- ✅ Find your running GPU job dynamically
- ✅ Connect directly to the GPU node
- ✅ Navigate to HalluLens directory automatically

## 🖥️ Remote Server Information

**Login Server:** `sporcsubmit.rc.rit.edu`
**Username:** `hy3134`
**SSH Key:** `C:\Users\HongM\OneDrive\Documents\ssh\.ssh\rit_rc_key`
**Key Passphrase:** `123`

### Server Architecture
- **Login Node:** `sporcsubmit.rc.rit.edu` - No GPU, used for basic tasks, file management
- **GPU Node:** `skl-a-52` - Has GPU access, used for ML training/inference

## 🔐 SSH Connection Commands

### 🚀 Automated GPU Connection (Recommended - Uses SSH Agent)
```bash
# Preferred method - automatically handles SSH agent and passphrase
./connect_gpu.sh

# Or on Windows
connect_gpu.bat
```
**✅ Benefits:**
- Automatically sets up SSH agent to avoid repeated passphrase prompts
- Finds the correct GPU node dynamically
- Navigates directly to HalluLens directory
- Handles different node allocations seamlessly
- Works on Linux, Mac, and Windows (Git Bash)

### SSH Agent Setup (Manual - if needed)
```bash
# Start SSH agent (if not already running)
eval $(ssh-agent -s)

# Add key to agent (enter passphrase once)
ssh-add "C:\Users\HongM\OneDrive\Documents\ssh\.ssh\rit_rc_key"

# Now SSH commands won't prompt for passphrase
ssh hy3134@sporcsubmit.rc.rit.edu
```
**Note:** Once added to SSH agent, the key remains loaded until agent is stopped or system reboot.

### Manual SSH Connection Commands (Fallback)

#### Basic SSH Connection (Login Node - No GPU)
```bash
ssh hy3134@sporcsubmit.rc.rit.edu -i "C:\Users\HongM\OneDrive\Documents\ssh\.ssh\rit_rc_key"
```
**Use for:** File management, basic Python scripts, environment setup, non-GPU tasks

#### SSH with Port Forwarding to GPU Node
```bash
ssh hy3134@sporcsubmit.rc.rit.edu -i "C:\Users\HongM\OneDrive\Documents\ssh\.ssh\rit_rc_key" -L 8887:skl-a-52:8887
```
**Use for:** Jupyter notebooks on GPU node, vLLM servers, GPU-accelerated training

#### SSH with Multiple Port Forwards (GPU Node + API Access)
```bash
ssh hy3134@sporcsubmit.rc.rit.edu -i "C:\Users\HongM\OneDrive\Documents\ssh\.ssh\rit_rc_key" -L 8887:skl-a-52:8887 -L 8000:skl-a-52:8000
```
**Use for:** Full access to GPU node services (Jupyter + vLLM/API servers)

#### Direct Access to GPU Node via Slurm Job
```bash
# First, check your running jobs
squeue -u hy3134

# Access GPU node through existing Slurm job
srun --jobid=20710294 --pty bash
```
**Note:** You must have an active Slurm job allocation to access GPU nodes. Direct SSH to compute nodes is not allowed.

## 🤖 Automated Connection Scripts

### `connect_gpu.sh` (Linux/Mac/Git Bash)
Automatically finds and connects to the GPU node running job "nb8887new":

```bash
./connect_gpu.sh
```

**Features:**
- ✅ Automatically finds the correct GPU node
- ✅ Handles different node allocations
- ✅ Navigates to HalluLens directory
- ✅ Provides colored output and error handling
- ✅ Works on Linux, Mac, and Windows (Git Bash)

### `connect_gpu.bat` (Windows)
Windows batch file wrapper for the bash script:

```cmd
connect_gpu.bat
```

**Requirements:**
- Git for Windows (includes Git Bash)
- Same SSH key setup as manual connection

### `test_gpu_connection.sh` (Environment Verification)
Comprehensive test script to verify the GPU environment:

```bash
./test_gpu_connection.sh
```

**Tests performed:**
- ✅ Hostname verification (GPU node)
- ✅ GPU availability (nvidia-smi)
- ✅ Python environment
- ✅ HalluLens directory structure
- ✅ Python package imports (PyTorch, CUDA, etc.)
- ✅ Storage space check

## 📁 Directory Structure

### Remote VM Paths
- **Home Directory:** `/home/hy3134/` (shared across login and GPU nodes)
- **HalluLens Repository:** `/home/hy3134/notebook_llm/HalluLens/`
- **Storage Usage:** 2.2T used, 859G available (73% usage)

### Node-Specific Information
- **Login Node (`sporcsubmit.rc.rit.edu`):** CPU-only, file management, basic development
- **GPU Node (`skl-a-52`):** GPU-enabled, ML training/inference, vLLM servers

### Local Paths
- **Local Repository:** `C:\Users\HongM\projects\HalluLens`
- **SSH Key Location:** `C:\Users\HongM\OneDrive\Documents\ssh\.ssh\rit_rc_key`

## 🐍 Python Environment

### Current Setup
- **Python Version:** Python 3.9.21
- **Python Path:** `/usr/bin/python`
- **Conda Available:** Yes (check with `conda env list`)

### Key Dependencies (from requirements.txt)
```
# general
jsonlines
pandas

# dataset
requests
datasets
gdown

# lm inference
vllm
openai

# longwiki
segtok
sentence-transformers==3.3.1

# refusal
ratelimit
retry
```

## 🚀 Quick Start Guide

### 🎯 Automated GPU Connection (Strongly Recommended)
```bash
# Run the automated connection script with SSH agent
./connect_gpu.sh

# Or on Windows
connect_gpu.bat
```
**✅ This script automatically:**
- Sets up SSH agent and handles the passphrase (no repeated prompts!)
- Finds the running job named "nb8887new" dynamically
- Connects directly to the correct GPU node
- Navigates to the HalluLens directory automatically
- Provides colored output and comprehensive error handling

**🔑 SSH Agent Benefits:**
- Enter passphrase only once per session
- All subsequent SSH connections are seamless
- Works across multiple terminal windows
- Significantly improves development workflow

### Option 2: Manual SSH Agent Setup + Connection
```bash
# 1. Setup SSH agent (one-time per session)
eval $(ssh-agent -s)
ssh-add "C:\Users\HongM\OneDrive\Documents\ssh\.ssh\rit_rc_key"  # Enter passphrase once

# 2. Connect to login node (no passphrase prompt)
ssh hy3134@sporcsubmit.rc.rit.edu

# 3. Find your GPU job
squeue -u hy3134 --name=nb8887new

# 4. Connect to GPU node
srun --jobid=<JOBID> --pty bash

# 5. Navigate to project
cd notebook_llm/HalluLens
```

### Option 3: Manual Connection (Legacy - Not Recommended)
```bash
# 1. Connect to login node (will prompt for passphrase)
ssh hy3134@sporcsubmit.rc.rit.edu -i "C:\Users\HongM\OneDrive\Documents\ssh\.ssh\rit_rc_key"

# 2. Find your GPU job
squeue -u hy3134 --name=nb8887new

# 3. Connect to GPU node
srun --jobid=<JOBID> --pty bash

# 4. Navigate to project
cd notebook_llm/HalluLens
```
**⚠️ Note:** This method requires entering the passphrase for every SSH connection.

### 3. Verify Environment
```bash
# Quick verification
hostname  # Should show skl-a-XX.rc.rit.edu
nvidia-smi  # Should show NVIDIA A100 GPU
python --version  # Should show Python 3.9.21

# Or run comprehensive test
./test_gpu_connection.sh
```

### 4. Install/Update Dependencies (if needed)
```bash
pip install -r requirements.txt
```

## 🔧 Development Workflow

### For AI Assistant Remote Development

#### CPU-Only Tasks (Login Node)
1. **Connect via SSH:** `ssh hy3134@sporcsubmit.rc.rit.edu -i "..."`
2. **Navigate to project directory:** `cd notebook_llm/HalluLens`
3. **Use for:** File management, code editing, dependency installation, basic Python scripts
4. **Limitations:** No GPU access, no ML model training/inference

#### GPU-Required Tasks (Need GPU Node Access)
1. **Connect to login node first:** `ssh hy3134@sporcsubmit.rc.rit.edu -i "..."`
2. **Check active Slurm jobs:** `squeue -u hy3134`
3. **Access GPU node via Slurm:** `srun --jobid=20710294 --pty bash`
4. **Use for:** vLLM servers, model training, GPU-accelerated inference
5. **GPU Available:** NVIDIA A100-PCIE-40GB (40GB VRAM), CUDA 12.4

### For Jupyter Notebook Development (GPU Node)
1. **Connect with port forwarding:** `-L 8887:skl-a-52:8887`
2. **Start Jupyter on GPU node:** `jupyter notebook --port=8887 --no-browser`
3. **Access locally:** Open `http://localhost:8887` in your browser
4. **Benefits:** Full GPU access for ML experiments

### For vLLM Server Access (GPU Node)
1. **Connect with port forwarding:** `-L 8000:skl-a-52:8000`
2. **Start vLLM server on GPU node:** Follow vLLM documentation
3. **Access API locally:** Use `http://localhost:8000` for API calls
4. **Note:** vLLM requires GPU, so must run on `skl-a-52`

## 📊 Current Remote Environment Status

### Existing Data and Models
- **Large Language Models:** 
  - `Llama-3.3-70B-Instruct-IQ3_M.gguf` (31GB)
  - `Meta-Llama-3.1-70B-Instruct-IQ3_M.gguf` (31GB)
  - `Meta-Llama-3-8B-Instruct-Q6_K.gguf` (6.6GB)

### Data Directories
- **LMDB Data:** `lmdb_data/` (504GB)
- **Checkpoints:** `checkpoints/` (12GB)
- **Output:** `output/` (314MB)
- **Data:** `data/` (2.8GB)

### Active Log Files
- Multiple server and training log files indicating active development
- Recent activity timestamps show ongoing work

## 🎯 Slurm Job Management

### GPU Job Configuration
- **Job Name:** `nb8887new` (standardized name for GPU jobs)
- **Partition:** `tier3`
- **Node:** Dynamic allocation (e.g., `skl-a-52`, `skl-a-XX`)
- **GPU:** NVIDIA A100-PCIE-40GB
- **Access Method:** Via automated connection script

### Key Slurm Commands
```bash
# Check your running jobs
squeue -u hy3134

# Find specific job by name
squeue -u hy3134 --name=nb8887new

# Check job details
scontrol show job <JOBID>

# Access GPU node through job (manual method)
srun --jobid=<JOBID> --pty bash

# Submit a new GPU job (if needed)
sbatch --partition=tier3 --gres=gpu:1 --time=24:00:00 --job-name=nb8887new your_script.sh

# Cancel a job
scancel <JOBID>
```

### Job Allocation Details
- **Time Limit:** 20 days (20-00:00:00)
- **GPU Access:** NVIDIA A100-PCIE-40GB
- **Memory:** Shared node resources
- **Storage:** Same home directory (`/home/hy3134/`) across all nodes

## ⚙️ Configuration Files

### LM Inference Configuration
- **File:** `utils/lm.py`
- **Current Server:** `CUSTOM_SERVER = "0.0.0.0"`
- **Default Port:** `8000`
- **Supported Models:** Llama 3.1/3.3, Mistral variants

### Model Endpoints
```python
model_map = {
    'meta-llama/Llama-3.1-405B-Instruct-FP8': {'name': 'llama3.1_405B', 'server_urls': ["http://0.0.0.0:8000/v1"]},
    'meta-llama/Llama-3.3-70B-Instruct': {'name': 'llama3.3_70B', 'server_urls': ["http://0.0.0.0:8000/v1"]},
    # ... other models
}
```

## � SSH Agent Best Practices

### Why Use SSH Agent?
SSH agent eliminates the need to repeatedly enter your SSH key passphrase during a session:
- **One-time setup:** Enter passphrase once per terminal session
- **Seamless connections:** All SSH commands work without prompts
- **Multiple connections:** Works across multiple terminal windows
- **Script compatibility:** Enables automated scripts and AI assistant connections

### SSH Agent Workflow
```bash
# 1. Start SSH agent (automatic in most modern terminals)
eval $(ssh-agent -s)

# 2. Add your key (enter passphrase once)
ssh-add "C:\Users\HongM\OneDrive\Documents\ssh\.ssh\rit_rc_key"

# 3. Verify key is loaded
ssh-add -l

# 4. Now all SSH connections are seamless
ssh hy3134@sporcsubmit.rc.rit.edu  # No passphrase prompt!
```

### Automated Setup with connect_gpu.sh
The `connect_gpu.sh` script automatically handles SSH agent setup:
- Detects if SSH agent is running
- Starts agent if needed
- Adds the key with automatic passphrase handling
- Uses multiple fallback methods (expect, sshpass, SSH_ASKPASS)
- Provides clear status messages

### SSH Agent Persistence
```bash
# For persistent SSH agent across terminal sessions (optional)
# Add to ~/.bashrc (Linux/Mac) or ~/.bash_profile (Mac)
if [ -z "$SSH_AUTH_SOCK" ]; then
    eval $(ssh-agent -s) > /dev/null 2>&1
    ssh-add "C:\Users\HongM\OneDrive\Documents\ssh\.ssh\rit_rc_key" > /dev/null 2>&1
fi
```

## �🔍 Troubleshooting

### Common Issues

#### SSH Agent Not Working / Repeated Passphrase Prompts
```bash
# Check if SSH agent is running
echo $SSH_AUTH_SOCK

# If empty, start SSH agent
eval $(ssh-agent -s)

# Add key to agent
ssh-add "C:\Users\HongM\OneDrive\Documents\ssh\.ssh\rit_rc_key"

# Verify key is loaded
ssh-add -l
```
**Solution:** Use the automated `connect_gpu.sh` script which handles SSH agent setup automatically.

#### SSH Agent Key Not Persisting
```bash
# On Windows (Git Bash), add to ~/.bashrc for persistence
echo 'eval $(ssh-agent -s) > /dev/null 2>&1' >> ~/.bashrc
echo 'ssh-add "C:\Users\HongM\OneDrive\Documents\ssh\.ssh\rit_rc_key" > /dev/null 2>&1' >> ~/.bashrc
```
**Note:** You'll still need to enter the passphrase once per terminal session.

#### Port Already in Use
```
bind [127.0.0.1]:8887: Address already in use
channel_setup_fwd_listener_tcpip: cannot listen to port: 8887
```
**Solution:** Use a different port or kill the process using the port

#### SSH Key Passphrase Issues
- **Passphrase:** `123`
- **Key Location:** `C:\Users\HongM\OneDrive\Documents\ssh\.ssh\rit_rc_key`
- **Recommended:** Use SSH agent to avoid repeated prompts

#### Storage Space
- **Current Usage:** 73% (2.2T used, 859G available)
- **Monitor with:** `df -h /home/hy3134`

### Useful Commands

#### Check System Resources
```bash
# Storage usage (works on both nodes)
df -h /home/hy3134

# Memory usage
free -h

# CPU usage
htop

# GPU usage (only available on skl-a-52)
nvidia-smi

# Check which node you're on
hostname
```

#### Process Management
```bash
# List running processes
ps aux | grep python

# Kill process by PID
kill <PID>

# Check port usage
netstat -tulpn | grep :8000
```

## 📝 Notes

### Development Best Practices
1. **Always work in the remote HalluLens directory:** `/home/hy3134/notebook_llm/HalluLens/`
2. **Choose the right node:**
   - **Login node:** File management, code editing, basic scripts
   - **GPU node (`skl-a-52`):** ML training, vLLM servers, GPU inference
3. **Use version control:** The remote repo has `.git` directory - commit changes regularly
4. **Monitor storage:** Large models and data files can fill up space quickly
5. **Use appropriate Python environment:** Check if conda environments are needed for specific tasks

### File Synchronization
- **Manual sync:** Use `scp` or `rsync` to sync files between local and remote
- **Git-based sync:** Commit locally, pull remotely (or vice versa)
- **Direct editing:** Use SSH + terminal editors for quick changes

### Security Notes
- **SSH key is password-protected** with passphrase `123`
- **RIT RC environment** has monitoring and usage policies
- **Keep credentials secure** and don't share in public repositories

## 🤖 AI Assistant Remote Development

### How AI Assistant Connects
The AI assistant can connect to the remote VM using the same SSH credentials and execute commands remotely. **With SSH agent setup, the AI assistant can connect seamlessly without passphrase prompts.** This enables:

- **Remote code execution:** Run Python scripts, tests, and benchmarks
- **Environment setup:** Install dependencies, configure settings
- **File editing:** Modify code files using terminal editors
- **System monitoring:** Check resources, processes, and logs
- **Development assistance:** Debug issues, run experiments, analyze results

### AI Assistant Workflow
1. **Establish SSH connection** using the provided credentials
2. **Navigate to project directory:** `/home/hy3134/notebook_llm/HalluLens/`
3. **Execute development tasks** as requested by the user
4. **Provide feedback** on results, errors, and recommendations

### Capabilities
- ✅ **Run HalluLens benchmarks and evaluations**
- ✅ **Install and manage Python dependencies**
- ✅ **Execute training and inference scripts**
- ✅ **Monitor system resources and processes**
- ✅ **Debug and troubleshoot issues**
- ✅ **Analyze logs and output files**
- ❌ **Access web interfaces** (requires port forwarding)
- ❌ **Modify files requiring GUI** (use terminal editors instead)

## 🔄 Synchronization Strategies

### Option 1: Git-Based Sync (Recommended)
```bash
# On local machine
git add .
git commit -m "Local changes"
git push origin main

# On remote VM
cd /home/hy3134/notebook_llm/HalluLens/
git pull origin main
```

### Option 2: SCP File Transfer
```bash
# Upload file to remote
scp -i "C:\Users\HongM\OneDrive\Documents\ssh\.ssh\rit_rc_key" local_file.py hy3134@sporcsubmit.rc.rit.edu:/home/hy3134/notebook_llm/HalluLens/

# Download file from remote
scp -i "C:\Users\HongM\OneDrive\Documents\ssh\.ssh\rit_rc_key" hy3134@sporcsubmit.rc.rit.edu:/home/hy3134/notebook_llm/HalluLens/remote_file.py ./
```

### Option 3: Rsync Synchronization
```bash
# Sync entire directory (upload)
rsync -avz -e "ssh -i 'C:\Users\HongM\OneDrive\Documents\ssh\.ssh\rit_rc_key'" ./local_dir/ hy3134@sporcsubmit.rc.rit.edu:/home/hy3134/notebook_llm/HalluLens/

# Sync entire directory (download)
rsync -avz -e "ssh -i 'C:\Users\HongM\OneDrive\Documents\ssh\.ssh\rit_rc_key'" hy3134@sporcsubmit.rc.rit.edu:/home/hy3134/notebook_llm/HalluLens/ ./local_dir/
```

## 🧪 Testing and Validation

### Environment Validation Commands
```bash
# Test SSH connection
ssh hy3134@sporcsubmit.rc.rit.edu -i "C:\Users\HongM\OneDrive\Documents\ssh\.ssh\rit_rc_key" "echo 'Connection successful'"

# Test Python environment
ssh hy3134@sporcsubmit.rc.rit.edu -i "C:\Users\HongM\OneDrive\Documents\ssh\.ssh\rit_rc_key" "cd notebook_llm/HalluLens && python --version"

# Test HalluLens dependencies
ssh hy3134@sporcsubmit.rc.rit.edu -i "C:\Users\HongM\OneDrive\Documents\ssh\.ssh\rit_rc_key" "cd notebook_llm/HalluLens && python -c 'import vllm, openai, pandas; print(\"Dependencies OK\")'"
```

### Quick HalluLens Test
```bash
# Test basic HalluLens functionality
cd /home/hy3134/notebook_llm/HalluLens/
python -c "from utils.lm import generate; print('LM utils imported successfully')"
```

## 📋 Environment Checklist

### Initial Setup Verification
- [ ] SSH connection established successfully
- [ ] Can navigate to `/home/hy3134/notebook_llm/HalluLens/`
- [ ] Python 3.9.21 is available
- [ ] Required dependencies are installed
- [ ] Can import HalluLens modules
- [ ] Storage space is adequate (>10GB free recommended)

### Before Running Experiments
- [ ] Latest code is synced from local/git
- [ ] Required data files are present in `data/` directory
- [ ] Model files are available (if using local models)
- [ ] vLLM server is running (if needed)
- [ ] Sufficient storage space for outputs
- [ ] No conflicting processes running

### After Experiments
- [ ] Results are saved in appropriate directories
- [ ] Log files are reviewed for errors
- [ ] Important outputs are backed up/synced
- [ ] Temporary files are cleaned up if needed
- [ ] System resources are released

---

**Last Updated:** August 20, 2025 - Added SSH Agent best practices and automated connection emphasis
**Environment:** RIT Research Computing Cluster
**Project:** HalluLens - LLM Hallucination Benchmark
**Status:** ✅ Remote Development Environment Ready
