#!/bin/bash
# Quick script to check files on remote server

SSH_USER="hy3134"
SSH_HOST="sporcsubmit.rc.rit.edu"
SSH_KEY="C:\Users\HongM\OneDrive\Documents\ssh\.ssh\rit_rc_key"

# Check if generation file exists
echo "Checking for generation file..."
ssh -i "$SSH_KEY" "$SSH_USER@$SSH_HOST" "ls -lh /shared/rc/llm-hd/HalluLens/goodwiki_jsonv2/generation.jsonl 2>&1"

echo ""
echo "Checking directory contents..."
ssh -i "$SSH_KEY" "$SSH_USER@$SSH_HOST" "ls -lh /shared/rc/llm-hd/HalluLens/goodwiki_jsonv2/ 2>&1"

echo ""
echo "Counting lines in generation file (if exists)..."
ssh -i "$SSH_KEY" "$SSH_USER@$SSH_HOST" "wc -l /shared/rc/llm-hd/HalluLens/goodwiki_jsonv2/generation.jsonl 2>&1"

