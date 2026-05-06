import os
import requests
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Read API key from environment
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={API_KEY}"

payload = {
    "contents": [
        {
            "parts": [
                {
                    "text": "Hello Gemini, reply with one short sentence."
                }
            ]
        }
    ]
}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)

data = response.json()

print("\nFull Response:\n")
print(data)

# Print generated text only
try:
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    print("\nGemini Response:")
    print(text)
except Exception:
    print("\nCould not extract text from response.")

