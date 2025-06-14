# NSFK? (Not Safe For Kids?) Project Guide for Claude Code

## Project Overview
Building an AI agent that analyzes YouTube videos to help busy parents determine if content is appropriate for their children. The tool provides:
- Quick content summary (1-2 lines)
- Key keywords found
- Safety rating (0-100 score)

## Project Repository
- **GitHub URL**: https://github.com/OzasaHiro/NSFK
- **Main Branch**: `main`
- **Language**: English (all code and documentation)

## Session Workflow

### Start of Session
```bash
cd /Users/hiroaki/Desktop/Development/ClaudeCode/NSFK
git pull origin main
```

### End of Session
1. Run tests if applicable
2. Check git status: `git status`
3. Add changes: `git add .`
4. Commit with descriptive message
5. Push to GitHub: `git push origin main`

## Project Structure
```
NSFK/
├── .env                    # Environment variables (Gemini API key)
├── .gitignore             # Git ignore file
├── requirements.txt       # Python dependencies
├── youtube_downloader.py  # YouTube video download functionality
├── venv/                  # Python virtual environment
└── chrome_extension/      # Chrome extension (to be created)
```

## Technical Stack
- **LLM**: Gemini (API key in .env)
- **Video Analysis**: 
  - yt-dlp for downloading
  - PySceneDetect for scene detection
  - Whisper for audio transcription
- **Deployment**: Chrome Extension

## Development Guidelines
1. This is the production version based on the prototype in Prototype_Implementation_Plan.md
2. Focus on Chrome Extension implementation
3. Main features:
   - YouTube video URL input
   - Video and audio content analysis
   - YouTube comments analysis (future)
   - Reddit/Wiki data integration (future)
   - Safety scoring algorithm
   - User-friendly interface

## Key Features to Implement
1. Chrome Extension that works on YouTube pages
2. Video content analysis using Gemini
3. Audio transcription and analysis
4. Safety scoring system (0-100)
5. Quick summary generation
6. Keyword extraction

## Important Notes
- Do NOT upload Plan.md, Prototype_Implementation_Plan.md, Reference.md, or SESSION_RESTART_GUIDE.md to GitHub
- Use Gemini API instead of OpenAI (as specified in the plan)
- Ensure all code and comments are in English
- Follow the prototype as reference but build the production version

## Testing Commands
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests (to be created)
python test_[feature].py
```

## Commit Message Template
```bash
git commit -m "$(cat <<'EOF'
Brief description of changes

- Specific change 1
- Specific change 2

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```