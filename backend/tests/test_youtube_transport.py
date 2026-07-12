from pathlib import Path

import pytest
from curl_cffi import requests

from backend.app.services import youtube


@pytest.mark.parametrize("target", youtube._IMPERSONATE_TARGETS)
def test_youtube_impersonation_target_is_supported(target: str) -> None:
    session = requests.Session(impersonate=target)
    try:
        response = session.get(Path(__file__).resolve().as_uri())
    finally:
        session.close()

    assert response.status_code == 0
