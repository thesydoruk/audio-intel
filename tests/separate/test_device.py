from audio_intel.separate.models.base import resolve_device


def test_bare_cuda_gets_an_index() -> None:
    # SpeechBrain picks the card from the part after ":" and warns on plain "cuda".
    assert resolve_device("cuda") == "cuda:0"
    assert resolve_device("cuda:1") == "cuda:1"
    assert resolve_device("cpu") == "cpu"


def test_auto_resolves_to_an_indexed_device_or_cpu() -> None:
    assert resolve_device("auto") in ("cuda:0", "cpu")
