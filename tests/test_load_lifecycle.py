"""Checkpoint identity and cold admission, without weights or model execution."""
import importlib
import threading
import time
import pytest


@pytest.mark.parametrize('replace_during_load', [False, True])
def test_checkpoint_identity_binds_bytes_before_construction(tmp_path, monkeypatch, replace_during_load):
    module = importlib.import_module('cnn_chinese_hw.recognizer.recognizer')
    from iso_tools.inference.artifacts import local_identity
    path = tmp_path / 'model.pt'
    path.write_bytes(b'old checkpoint')
    expected = local_identity(path, 'torch').revision
    class Model:
        def to(self, _): return self
        def eval(self): pass
        def load_state_dict(self, _):
            if replace_during_load:
                path.write_bytes(b'new checkpoint bytes')
    load_options = []
    def load(*args, **kwargs):
        load_options.append(kwargs)
        return {'classes': [65], 'data_cfg': {}, 'model_cfg': {}, 'model_state': {}}
    monkeypatch.setattr(module.torch, 'load', load)
    monkeypatch.setattr(module, 'build_model', lambda _: Model())
    if replace_during_load:
        with pytest.raises(RuntimeError, match='changed during model construction'):
            module._LoadedModel(path, 'cpu')
    else:
        loaded = module._LoadedModel(path, 'cpu')
        assert loaded.inference_identity.revision == expected
        path.write_bytes(b'replacement after completed load')
        assert loaded.inference_identity.revision == expected
    assert load_options[0]['weights_only'] is True


def test_concurrent_cold_wrapper_access_loads_once_and_failure_is_retryable(monkeypatch):
    module = importlib.import_module('cnn_chinese_hw.client_server.HWServer')
    entered, release = threading.Event(), threading.Event()
    calls, outputs = [], []
    class Recognizer:
        def __init__(self, **kwargs):
            calls.append(self)
            entered.set()
            assert release.wait(2)
    monkeypatch.setattr(module, 'HandwritingRecognizer', Recognizer)
    server = module.HWServer()
    threads = [threading.Thread(target=lambda: outputs.append(server.recognizer)) for _ in range(8)]
    for thread in threads: thread.start()
    assert entered.wait(2)
    release.set()
    for thread in threads: thread.join(2)
    assert not any(thread.is_alive() for thread in threads)
    assert len(calls) == 1 and len(outputs) == 8
    assert all(value is calls[0] for value in outputs)
    def fail(**kwargs): raise RuntimeError('temporary load error')
    monkeypatch.setattr(module, 'HandwritingRecognizer', fail)
    server = module.HWServer()
    with pytest.raises(RuntimeError): _ = server.recognizer
    monkeypatch.setattr(module, 'HandwritingRecognizer', Recognizer)
    assert server.recognizer is calls[-1]
