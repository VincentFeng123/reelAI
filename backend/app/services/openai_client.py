from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI


def build_openai_client(*, api_key: str, timeout: float, enabled: bool) -> "OpenAI | None":
    if not enabled:
        return None

    from openai import OpenAI

    return OpenAI(api_key=api_key, timeout=timeout)
