#!/bin/bash

# Git setup script for RAG Lens project

echo "Setting up Git repository for RAG Lens project..."

# Configure git (if needed)
if ! git config user.name >/dev/null 2>&1; then
    git config --global user.name "cute-spring"
    git config --global user.email "cute-spring@example.com"
    echo "Git user configured as cute-spring"
fi

# Initialize repository if needed
if [ ! -d ".git" ]; then
    git init
    echo "Git repository initialized"
fi

# Add all files
git add .

# Initial commit
git commit -m "Initial commit: RAG Pipeline Testing & Tuning Tool

- Complete RAG pipeline testing tool with Streamlit interface
- Step-by-step pipeline control and simulation
- Test case management system with multiple source support
- API integration guide with comprehensive documentation
- Enhanced test suite with 156 test cases across 12 categories
- Authentication patterns, error handling, and monitoring templates
- Health check endpoints and performance monitoring
- Complete integration examples and interface contracts

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

echo "Initial commit created successfully!"

# Show status
echo "Current Git status:"
git status

echo ""
echo "Repository ready for GitHub push!"
echo "To push to GitHub:"
echo "1. Create repository at https://github.com/cute-spring"
echo "2. Run: git remote add origin https://github.com/cute-spring/rag-lens.git"
echo "3. Run: git push -u origin main"