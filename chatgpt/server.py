import uvicorn
from fastapi import FastAPI 
from openai import OpenAI
from pydantic import BaseModel


app = FastAPI()

class Payload(BaseModel):
    api_key: str
    prompt: str
    model_type: str


def send_prompt(prompt, key, model_type):
    client = OpenAI(api_key=key)
    response = client.chat.completions.create(
        model=model_type,
        messages=[{"role": "user", "content": prompt}]
    )
    response = response.choices[0].message.content.strip()
    response = f"{response}"
    return response


@app.post("/openai_api")
async def model(payload: Payload):
    try:
        api_key = payload.api_key
        prompt = payload.prompt
        model_type = payload.model_type
        response = send_prompt(prompt, api_key, model_type)
        res = {"response": response}
    except Exception as e:
        res = {"response": f"error: {e}"}
    return res


uvicorn.run(app, host='0.0.0.0', port=8000)
