"""60-second demo: the official OpenAI SDK against Switchboard (only base_url changes).

    switchboard serve --with-worker &        # lite mode: no Docker, mock provider, no keys
    pip install openai && python examples/openai_demo.py
"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="anything-in-lite-mode")

for model in ("gpt-4o-mini", "llama-3.3-70b-versatile", "gemini-2.5-flash", "auto"):
    raw = client.chat.completions.with_raw_response.create(
        model=model, temperature=0, messages=[{"role": "user", "content": "What is the capital of France?"}])
    r = raw.parse()
    h = raw.headers
    print(f"{model:26s} -> served by {h['x-provider']}/{h['x-model']}  cache={h['x-cache']}  "
          f"saved=${h['x-cost-saved-usd']}  {r.choices[0].message.content[:30]!r}")

# again: exact-cache hit
raw = client.chat.completions.with_raw_response.create(
    model="auto", temperature=0, messages=[{"role": "user", "content": "What is the capital of France?"}])
print("repeat:", raw.headers["x-cache"])

# streaming
for chunk in client.chat.completions.create(model="auto", stream=True, messages=[{"role": "user", "content": "Say hi"}]):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
