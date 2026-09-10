import sys
from pathlib import Path

# Add current script's folder to Python's import search path
sys.path.append(str(Path(__file__).resolve().parent))

from reply_buffer import ReplayBuffer

print("ok")