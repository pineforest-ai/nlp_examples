from openai import OpenAI


OPENAI_KEY = ""

client = OpenAI(api_key=OPENAI_KEY)

def send_prompt(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    response = response.choices[0].message.content.strip()
    response = f"{response}"
    return response


prompt = "Расскажи анекдот про Штирлица."

response = send_prompt(prompt)
print(response)
