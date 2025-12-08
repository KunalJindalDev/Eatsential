from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: No API Key found in .env")
else:
    print(f"Using Key: {api_key[:10]}...")
    client = genai.Client(api_key=api_key)
    
    print("\nAvailable Models:")
    try:
        # Pager object - iterate to print
        for model in client.models.list():
            print(f" - {model.name}")
    except Exception as e:
        print(f"Error listing models: {e}")