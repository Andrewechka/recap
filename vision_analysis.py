import json
import time

# Безопасные лимиты
MAX_TOKENS = 300
MODEL_ID = "gpt-4o"
DETAIL = "low"


def _vision_call(client, messages):
    """
    Единый безопасный вызов vision
    """
    return client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
    )


def detect_important_pages(
    profile_reference,
    chapter_reference,
    pages,
    client,
    prompt,
    instructions,
):
    """
    Возвращает список важных страниц.
    Работает ПО ОДНОЙ странице за раз → не бьёт TPM.
    """

    important_pages = []
    total_tokens = 0

    for idx, page in enumerate(pages):
        messages = [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{page}",
                            "detail": DETAIL,
                        },
                    },
                ],
            },
        ]

        try:
            response = _vision_call(client, messages)
        except Exception as e:
            print(f"⚠️ Vision error on page {idx}: {e}")
            time.sleep(2)
            continue

        total_tokens += response.usage.total_tokens

        try:
            data = json.loads(response.choices[0].message.content)
            if data.get("important", False):
                important_pages.append(
                    {"image_index": idx, "type": data.get("type", "chapter")}
                )
        except Exception:
            pass

        time.sleep(0.5)  # защита от TPM

    return {
        "total_tokens": total_tokens,
        "parsed_response": important_pages,
    }


def get_important_panels(
    profile_reference,
    panels,
    client,
    prompt,
    instructions,
):
    """
    Определяет важные панели — по одной панели за вызов
    """

    important = []
    total_tokens = 0

    for idx, panel in enumerate(panels):
        messages = [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{panel}",
                            "detail": DETAIL,
                        },
                    },
                ],
            },
        ]

        try:
            response = _vision_call(client, messages)
        except Exception as e:
            print(f"⚠️ Vision error on panel {idx}: {e}")
            time.sleep(1)
            continue

        total_tokens += response.usage.total_tokens

        try:
            data = json.loads(response.choices[0].message.content)
            if data.get("important", False):
                important.append(idx)
        except Exception:
            pass

        time.sleep(0.3)

    return {
        "total_tokens": total_tokens,
        "parsed_response": important,
    }
