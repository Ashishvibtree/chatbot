from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# Initialize OpenRouter Client
# Replace with your actual OpenRouter API key or set it as an environment variable
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_API_KEY,
)

# Pydantic models for request validation
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    system_prompt: str
    messages: List[Message]

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves the HTML frontend."""
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Handles chat requests from the frontend."""
    try:
        # 1. Prepare messages array (System Prompt + Chat History)
        api_messages = [{"role": "system", "content": request.system_prompt}]
        
        # Add the conversation history
        for msg in request.messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        # 2. Call OpenRouter API
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:8000", # Replace with your site URL
                "X-OpenRouter-Title": "FastAPI Custom Bot", # Replace with your site name
            },
            # model="mistralai/ministral-3b-2512",
            model="meta-llama/llama-3.2-3b-instruct",
            messages=api_messages
        )
        
        # 3. Return the AI's response
        bot_reply = completion.choices[0].message.content
        return {"reply": bot_reply}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
