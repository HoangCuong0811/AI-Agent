import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from prompts import SYSTEM_PROMPT

load_dotenv('.env')
if not load_dotenv:
    raise FileNotFoundError('Cannot found .env file')

client = genai.Client()
MODEL = "gemini-3.5-flash-lite"
def get_weather(location: str):
    return f"The weather at {location} is cloudy with high humidity"


config = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[get_weather]
)

chat = client.chats.create(
    model=MODEL,
    config=config
) 

response = chat.send_message("What is the weather in Hanoi right now?")

print(response.text)



