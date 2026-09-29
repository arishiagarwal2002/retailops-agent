"""Create an API key. The plaintext is printed once; only its hash is stored.

    python -m scripts.create_api_key demo-ui
"""
import sys

from app.core.auth import create_api_key

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "default"
    print(create_api_key(name))
