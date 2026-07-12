import threading

from backend.app.clip_engine.singleflight import singleflight


def test_singleflight_coalesces_identical_concurrent_work() -> None:
    cache: dict[str, str] = {}
    call_count = 0
    call_count_lock = threading.Lock()
    provider_started = threading.Event()
    provider_release = threading.Event()
    barrier = threading.Barrier(2)
    results: list[str] = []

    def worker() -> None:
        nonlocal call_count
        barrier.wait()
        with singleflight("same-key"):
            if "same-key" not in cache:
                with call_count_lock:
                    call_count += 1
                provider_started.set()
                assert provider_release.wait(timeout=2)
                cache["same-key"] = "result"
            results.append(cache["same-key"])

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    assert provider_started.wait(timeout=2)
    provider_release.set()
    for thread in threads:
        thread.join(timeout=2)

    assert not any(thread.is_alive() for thread in threads)
    assert call_count == 1
    assert results == ["result", "result"]
