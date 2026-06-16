import os
import google.generativeai as genai

# Test our native parsing logic before falling back to python-dotenv
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if line.startswith('GEMINI_API_KEY='):
                os.environ['GEMINI_API_KEY'] = line.strip().split('=', 1)[1].strip('"\'')

api_key = os.environ.get("GEMINI_API_KEY")

print(f"Loaded API Key: {api_key[:10]}... (length: {len(api_key) if api_key else 0})")

if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3.1-pro-preview')
        print("Calling Gemini 3.1 Pro Preview...")
        response = model.generate_content("Reply with the exact text: 'Pipes are working!'")
        print(f"Gemini Response: {response.text.strip()}")
    except Exception as e:
        print(f"Failed to call Gemini: {e}")
else:
    print("API Key not found!")
