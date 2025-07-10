"""
Login to Hugging Face Hub
"""

from huggingface_hub import login

print("=== Hugging Face Login ===")
print("\nTo get your token:")
print("1. Go to: https://huggingface.co/settings/tokens")
print("2. Create a new token or copy an existing one")
print("3. Make sure you have 'read' permissions for accessing models")
print()

token = input("Please enter your Hugging Face token: ").strip()

if token:
    try:
        login(token=token)
        print("\n✅ Successfully logged in to Hugging Face!")
        
        # Verify login
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami()
        print(f"Logged in as: {user_info['name']}")
        
    except Exception as e:
        print(f"\n❌ Login failed: {e}")
        print("\nPlease make sure:")
        print("1. Your token is valid")
        print("2. You have accepted the Gemma license")
        print("3. Your internet connection is working")
else:
    print("\n❌ No token provided")