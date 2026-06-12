# Client Integration

## Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/api/v1",
    api_key="not-needed",
)

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

If `gemma` is not already active, the server loads it automatically before proxying the request.

If you want to pre-warm a model explicitly, you can still call `POST /api/models/{name}/load` first.

If a proxy request returns `503` because runtime communication failed, call `load` again before retrying.
