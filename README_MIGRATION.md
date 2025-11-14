# HalluLens Storage Migration Guide

This directory contains scripts to migrate HalluLens files from the home directory to the new shared storage at `/shared/rc/llm-hd`.

## Overview

**Problem**: Home directory is 94% full (2.9T used, 195G available)
**Solution**: Migrate ~1.6TB of data to new 10TB shared storage
**Approach**: SAFE COPY + verification + symbolic links (originals preserved until verified)

## Migration Scripts

### 1. `migrate_to_shared_storage.sh`
**Full automated COPY script (SAFE)**
- Copies all files in 5 phases (originals preserved)
- Uses rsync for reliable transfers
- Verifies copy integrity
- Includes progress tracking and logging

**Usage:**
```bash
bash migrate_to_shared_storage.sh
```

### 2. `migrate_phase.sh`
**Phased COPY script for better control (SAFE)**
- Copy individual phases separately
- Check copy status
- Resume interrupted copies
- Originals always preserved

**Usage:**
```bash
# Copy specific phase
bash migrate_phase.sh 1    # Large data files (1.4TB)
bash migrate_phase.sh 2    # Model files (66GB)
bash migrate_phase.sh 3    # Training data (38.5GB)
bash migrate_phase.sh 4    # Python environment (9.5GB)
bash migrate_phase.sh 5    # Remaining data (5.5GB)

# Check status
bash migrate_phase.sh status
```

### 3. `run_migration_in_screen.sh`
**Screen session manager for uninterrupted migration**
- Runs migration in detachable screen session
- Prevents SSH disconnection issues
- Monitor progress remotely

**Usage:**
```bash
# Start full migration in screen
bash run_migration_in_screen.sh start

# Start specific phase in screen
bash run_migration_in_screen.sh start 1

# Attach to running session
bash run_migration_in_screen.sh attach

# Check status
bash run_migration_in_screen.sh status

# Stop migration
bash run_migration_in_screen.sh stop
```

### 4. `verify_migration.sh`
**Copy and symlink verification**
- Verifies copy integrity
- Tests file accessibility
- Supports both copy and symlink verification
- Generates detailed report

**Usage:**
```bash
bash verify_migration.sh
```

### 5. `create_symlinks.sh`
**Symlink creation (after copy verification)**
- Creates symbolic links to replace originals
- Verifies copy integrity before symlinking
- Creates backups of originals
- Individual or batch processing

**Usage:**
```bash
# Create all symlinks
bash create_symlinks.sh all

# Create specific symlink
bash create_symlinks.sh goodwiki_json_2

# List available items
bash create_symlinks.sh --list
```

### 6. `cleanup_originals.sh`
**Cleanup backup files (final step)**
- Removes backup files created during symlinking
- Verifies symlinks work before cleanup
- Recovers disk space
- Safety checks included

**Usage:**
```bash
# List backup files
bash cleanup_originals.sh list

# Verify symlinks
bash cleanup_originals.sh verify

# Clean up backups
bash cleanup_originals.sh cleanup
```

## Migration Phases

### Phase 1: Large Data Files (1.4TB)
- `goodwiki_json_2` (470G)
- `goodwiki_json` (470G)
- `triviiqa_json` (447G)

### Phase 2: Model Files (66GB)
- `Meta-Llama-3.1-70B-Instruct-IQ3_M.gguf` (30G)
- `Llama-3.3-70B-Instruct-IQ3_M.gguf` (30G)
- `Meta-Llama-3-8B-Instruct-Q6_K.gguf` (6.2G)

### Phase 3: Training Data (38.5GB)
- `checkpoints` (31G)
- `npyloggingtestdir_100samples` (7.3G)

### Phase 4: Python Environment (9.5GB)
- `halu` virtual environment

### Phase 5: Remaining Data (5.5GB)
- `data` directory

## Recommended Safe Workflow

### Option 1: Full Automated Copy (Safest)
```bash
# 1. Upload scripts to remote server
# 2. Connect to remote environment
bash connect_gpu.sh

# 3. Run full copy in screen session
bash run_migration_in_screen.sh start

# 4. Monitor progress (can disconnect/reconnect)
bash run_migration_in_screen.sh attach

# 5. Verify copies
bash verify_migration.sh

# 6. Create symlinks (after verification)
bash create_symlinks.sh all

# 7. Verify symlinks work
bash verify_migration.sh

# 8. Clean up backup files (optional)
bash cleanup_originals.sh cleanup
```

### Option 2: Phased Copy (More Control)
```bash
# 1. Connect to remote environment
bash connect_gpu.sh

# 2. Start with largest files first
bash run_migration_in_screen.sh start 1

# 3. Verify phase 1
bash verify_migration.sh

# 4. Create symlinks for phase 1
bash create_symlinks.sh goodwiki_json_2
bash create_symlinks.sh goodwiki_json
bash create_symlinks.sh triviiqa_json

# 5. Continue with remaining phases
bash run_migration_in_screen.sh start 2
# ... repeat verify + symlink steps

# 6. Final cleanup
bash cleanup_originals.sh cleanup
```

## Safety Features

- **Rsync**: Reliable file transfers with progress tracking
- **Verification**: Checks file integrity before creating symlinks
- **Logging**: Detailed logs for troubleshooting
- **Resumable**: Can restart interrupted transfers
- **Space checks**: Verifies sufficient space before starting
- **Symbolic links**: Maintains compatibility with existing scripts

## Important Notes

1. **Backup**: Ensure important data is backed up before migration
2. **Screen sessions**: Use screen to prevent SSH disconnection issues
3. **Monitoring**: Check logs regularly during migration
4. **Verification**: Always run verification after migration
5. **Gradual approach**: Consider running phases separately for large datasets

## Troubleshooting

### If migration fails:
1. Check log files for error details
2. Verify network connectivity
3. Check available disk space
4. Resume with phase-specific script

### If symlinks break:
1. Run verification script to identify issues
2. Manually recreate broken symlinks:
   ```bash
   ln -s /shared/rc/llm-hd/HalluLens/filename filename
   ```

### If space issues occur:
1. Check disk usage: `df -h`
2. Clean up temporary files
3. Consider migrating in smaller batches

## Expected Results

After successful migration:
- Home directory usage: ~50-60% (down from 94%)
- Available space: ~1.2-1.4TB (up from 195GB)
- All existing scripts continue to work via symlinks
- Data accessible from new high-capacity storage
