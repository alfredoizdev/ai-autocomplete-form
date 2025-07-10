"""Test if HF_TOKEN is set and working"""
import os
from huggingface_hub import HfApi

token = os.environ.get('HF_TOKEN')
if token:
    print(f"✅ HF_TOKEN found (length: {len(token)})")
    try:
        api = HfApi(token=token)
        user = api.whoami()
        print(f"✅ Successfully authenticated as: {user['name']}")
        
        # Save token for huggingface_hub to use
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("✅ Token saved for this session")
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
else:
    print("❌ HF_TOKEN environment variable not set")
    print("Please run: export HF_TOKEN='your_token_here'")