# Client Integration

## Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/api/v1",
    api_key="not-needed",
)

# Load the preset first:
# POST http://localhost:8000/api/models/gemma/load
response = client.chat.completions.create(
    model="gemma",
    messages=[{"role": "user", "content": "Explain gravity in one sentence."}],
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
print()
```

## curl

```bash
curl http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma",
    "messages": [{"role": "user", "content": "Why is the sky blue?"}],
    "stream": true
  }'
```
