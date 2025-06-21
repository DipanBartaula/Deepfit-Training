# huggingface_login.py
"""
Simple utility to login to Hugging Face Hub.
Usage:
  - Set the environment variable HUGGINGFACE_TOKEN in .env, or
  - Run this script and input the token interactively.
Requires: huggingface-hub package
"""
import os
from huggingface_hub import login

def hf_login(token: str = None):
    """
    Log in to Hugging Face Hub.
    If token is None, tries environment variable HUGGINGFACE_TOKEN, else prompts user input.
    """
    if token is None:
        token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        try:
            token = input("Enter your Hugging Face access token: ").strip()
        except KeyboardInterrupt:
            print("\n[huggingface_login] No token provided; aborting.")
            return
    if not token:
        print("[huggingface_login] Empty token; aborting.")
        return
    login(token)
    print("[huggingface_login] Successfully logged in to Hugging Face.")

if __name__ == "__main__":
    hf_login()
