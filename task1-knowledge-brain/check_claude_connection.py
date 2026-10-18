import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(".env"))

api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
model = os.getenv("ANTHROPIC_MODEL", "").strip()

if not api_key or not model:
    print("ERROR: API key or model setting is missing.")
else:
    print("Testing model:", model)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=30,
            messages=[
                {"role": "user", "content": "Reply with OK only."}
            ],
        )

        print("SUCCESS: Claude responded.")
        for block in response.content:
            if block.type == "text":
                print("Response:", block.text)

    except anthropic.APIStatusError as exc:
        print("API status:", exc.status_code)
        print("Error type:", exc.type)
        print("Error message:", exc.message[:1000])

    except Exception as exc:
        print("Error type:", type(exc).__name__)
        print("Error message:", str(exc)[:500])