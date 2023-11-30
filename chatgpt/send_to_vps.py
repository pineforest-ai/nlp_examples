import requests


VPS_IP = ""

API_KEY = ""

prompt = "Расскажи анекдот про Штирлица."
res = requests.post(
    f"http://{VPS_IP}:8000/openai_api",
    json={"api_key": API_KEY, "prompt": prompt, "model_type": "gpt-3.5-turbo"}
)
print(res.status_code)
if res.status_code == 200:
    print(res.json()["response"])
