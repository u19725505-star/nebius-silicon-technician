"""Thin Nebius Token Factory client (OpenAI-compatible)."""
import os
import sys
from openai import OpenAI

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"


def client():
    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        sys.exit("Set NEBIUS_API_KEY first (see README).")
    return OpenAI(base_url=BASE_URL, api_key=key)


def list_nvidia_models():
    for m in client().models.list().data:
        if "nvidia" in m.id.lower():
            print(m.id)


def ask(system, user, model=None, max_tokens=300, temperature=0.2):
    """Returns (text, usage). Keep max_tokens small to protect credits."""
    model = model or os.environ.get("NEBIUS_MODEL")
    if not model:
        sys.exit("Set NEBIUS_MODEL first (run: python -m silicon_tech.llm models).")
    r = client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return r.choices[0].message.content, r.usage


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "models":
        list_nvidia_models()
    else:
        text, usage = ask(
            "You are a data center technician. Be brief.",
            "A CPU core shows IPC 0.45 and cache miss rate 36%. One-sentence diagnosis?",
            max_tokens=200,
        )
        print(text)
        print("tokens:", usage.prompt_tokens, "in,", usage.completion_tokens, "out")