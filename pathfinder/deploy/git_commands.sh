#!/bin/bash
#
# Pathfinder Git Commands Quick Reference
# Copy-paste these commands as needed
#

# =============================================================================
# INITIAL SETUP (First Time)
# =============================================================================

# Clone the repository
git clone https://github.com/your-org/ai-security.git
cd ai-security

# Checkout pathfinder branch
git checkout pathfinder

# Or create a new pathfinder branch from main
git checkout main
git checkout -b pathfinder

# =============================================================================
# UPDATE PATHFINDER (Get Latest)
# =============================================================================

# Fetch all changes from remote
git fetch origin

# Switch to pathfinder branch
git checkout pathfinder

# Pull latest changes
git pull origin pathfinder

# Alternative: reset to match remote exactly
git fetch origin
git reset --hard origin/pathfinder

# =============================================================================
# WORKING WITH PATHFINDER CODE
# =============================================================================

# Check current branch
git branch

# Check status
git status

# View recent commits
git log --oneline -10

# View changes in a file
git diff pathfinder/security/pathfinder_scanner.py

# =============================================================================
# MAKING CHANGES
# =============================================================================

# Stage specific files
git add pathfinder/deploy/secure_vllm_deploy.sh

# Stage all changes
git add .

# Commit with message
git commit -m "feat(pathfinder): add secure vLLM deployment script"

# Push to remote
git push origin pathfinder

# =============================================================================
# SYNC WITH MAIN
# =============================================================================

# Get latest from main
git checkout main
git pull origin main

# Switch back to pathfinder and merge main
git checkout pathfinder
git merge main

# Or rebase pathfinder on top of main
git checkout pathfinder
git rebase main

# =============================================================================
# DEPLOY TO PRODUCTION
# =============================================================================

# Tag a release
git tag -a v1.0.0 -m "Pathfinder v1.0.0 - Secure vLLM deployment"
git push origin v1.0.0

# Merge pathfinder into main
git checkout main
git merge pathfinder
git push origin main

# =============================================================================
# TROUBLESHOOTING
# =============================================================================

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Discard all local changes
git checkout -- .

# Stash changes temporarily
git stash
git stash pop  # restore them later

# View stashed changes
git stash list

# =============================================================================
# CLONE PATHFINDER INTO CONTAINER MOUNT
# =============================================================================

# For use with secure_vllm_deploy.sh
cd /home/compat/models/vLLM
git clone https://github.com/your-org/ai-security.git pathfinder
cd pathfinder
git checkout pathfinder

# Update later
cd /home/compat/models/vLLM/pathfinder
git fetch origin
git pull origin pathfinder
