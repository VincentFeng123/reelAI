from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI


def build_openai_client(*, api_key: str, timeout: float, enabled: bool) -> "OpenAI | None":
    # OpenAI integration is permanently disabled.
    return None
