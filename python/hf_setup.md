# Hugging Face Setup Instructions

## Login to Hugging Face

You need to login to Hugging Face to download the Gemma model. Here's how:

### Option 1: Using Environment Variable (Recommended)

1. Get your token from: https://huggingface.co/settings/tokens
2. Create a new token with "read" permissions
3. Set the token as an environment variable:

```bash
export HF_TOKEN="your_token_here"
```

### Option 2: Using Python

Run this in your terminal:

```bash
cd python
source venv/bin/activate
python
```

Then in Python:

```python
from huggingface_hub import login
login()
# Enter your token when prompted
```

### Option 3: Create .env file

Create a `.env` file in the python directory:

```
HF_TOKEN=your_token_here
```

## After Login

Once logged in, you can proceed with downloading the model:

```bash
python mlx_training/download_model.py
```