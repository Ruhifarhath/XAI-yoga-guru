# utils/groq_rag.py
import requests

GROQ_API_KEY = ""
MODEL = "llama3-70b-8192" 

def answer_with_groq(query: str, context: str) -> str:
    prompt = f"""
You are a yoga expert. Use the following context to answer the user's question clearly and informatively.

Context:
{context}

Question:
{query}

Answer:"""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 500
    }

   
    try:
        response = requests.post(url, headers=headers, json=data)

        # Check for errors
        if response.status_code != 200:
            return f"❌ Groq API Error: {response.status_code} - {response.text}"

        json_data = response.json()

        if 'choices' not in json_data or len(json_data['choices']) == 0:
            return "❌ Groq API Error: No response choices returned."

        return json_data['choices'][0]['message']['content'].strip()

    except Exception as e:
        return f"❌ Groq API Error: {e}"
