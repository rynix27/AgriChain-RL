"""
server/app.py -- AgriChain-RL server entry point

Required by OpenEnv multi-mode deployment validator.
Starts the AgriChain-RL FastAPI server on port 7860.
"""

import uvicorn
import sys
import os

# Add parent directory to path so agrichain.py is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """
    Main entry point for AgriChain-RL server.
    Required by OpenEnv multi-mode deployment spec.
    """
    from agrichain import app

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )


if __name__ == "__main__":
    main()
