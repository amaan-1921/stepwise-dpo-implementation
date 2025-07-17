# test_api_key.py
import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

try:
    models = openai.models.list()
    print("✅ API Key is working. Models available:")
    for model in models.data[:5]:
        print("-", model.id)
except Exception as e:
    print("❌ Error:", e)
