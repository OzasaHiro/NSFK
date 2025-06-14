#!/usr/bin/env python3
"""
NSFK API Startup Script
Simple script to start the NSFK API server with proper configuration
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    print("Checking requirements...")
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found!")
        print("Please create a .env file with your GEMINI_API_KEY:")
        print("  echo 'GEMINI_API_KEY=your_api_key_here' > .env")
        return False
    
    # Check if GEMINI_API_KEY is set
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in .env file!")
        print("Please add your Gemini API key to the .env file")
        return False
    
    # Check if key looks valid (basic check)
    if len(api_key) < 20:
        print("⚠️  GEMINI_API_KEY looks suspicious (too short)")
        print("Please verify your API key is correct")
    
    print("✅ Environment configuration looks good")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
        ("aiohttp", "aiohttp"),
        ("python-dotenv", "dotenv")
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them with:")
        print("  pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def start_server(host="127.0.0.1", port=8000, reload=True):
    """Start the FastAPI server"""
    print(f"Starting NSFK API server on {host}:{port}")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        import uvicorn
        uvicorn.run(
            "api:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

def main():
    """Main function"""
    print("NSFK API Server Startup")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    if not check_dependencies():
        sys.exit(1)
    
    # Get command line arguments
    host = "127.0.0.1"
    port = 8000
    reload = True
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--production":
            reload = False
            print("Starting in production mode (no auto-reload)")
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python start_api.py              # Development mode")
            print("  python start_api.py --production # Production mode")
            print("  python start_api.py --help       # Show this help")
            return
    
    print(f"\nServer will be available at:")
    print(f"  Local: http://{host}:{port}")
    print(f"  API Documentation: http://{host}:{port}/docs")
    print(f"  Health Check: http://{host}:{port}/health")
    print()
    
    # Start server
    start_server(host, port, reload)

if __name__ == "__main__":
    main()