"""Tests: LlamaCppControlPlane HTTP 客户端 + LlamaCppClient 外观 + 工具函数。"""

from __future__ import annotations

import pytest
from host.config import Settings

# ===========================================================================
# 工具函数
# ===========================================================================


class TestMmprojHelpers:
    def test_is_mmproj_model_id(self):
        from host.client.llama_cpp_client import is_mmproj_model_id

        assert is_mmproj_model_id("Qwen-mmproj-BF16") is True
        assert is_mmproj_model_id("Qwen3.5-4B-Q4_K_M") is False
        assert is_mmproj_model_id("") is False

    def test_mmproj_path_from_preset(self):
        from host.client.llama_cpp_client import mmproj_path_from_preset

        ini = "[model]\nmodel = /models/main.gguf\nmmproj = /models/mmproj.gguf\n"
        assert mmproj_path_from_preset(ini) == "/models/mmproj.gguf"

    def test_mmproj_path_from_preset_short_key(self):
        from host.client.llama_cpp_client import mmproj_path_from_preset

        ini = "m = /models/mmproj.gguf\n"
        assert mmproj_path_from_preset(ini) == "/models/mmproj.gguf"

    def test_mmproj_path_from_preset_no_match(self):
        from host.client.llama_cpp_client import mmproj_path_from_preset

        assert mmproj_path_from_preset("other = value") is None
        assert mmproj_path_from_preset("") is None

    def test_model_path_from_preset(self):
        from host.client.llama_cpp_client import model_path_from_preset

        ini = "[model]\nmodel = /models/main.gguf\n"
        assert model_path_from_preset(ini) == "/models/main.gguf"

    def test_mmproj_path_from_args(self):
        from host.client.llama_cpp_client import mmproj_path_from_args

        args = ["--some-flag", "--mmproj", "/path/to/mmproj.gguf", "-c", "4096"]
        assert mmproj_path_from_args(args) == "/path/to/mmproj.gguf"

    def test_mmproj_path_from_args_short_flag(self):
        from host.client.llama_cpp_client import mmproj_path_from_args

        args = ["-mm", "/path/mmproj.gguf"]
        assert mmproj_path_from_args(args) == "/path/mmproj.gguf"

    def test_mmproj_path_from_args_no_match(self):
        from host.client.llama_cpp_client import mmproj_path_from_args

        assert mmproj_path_from_args([]) is None
        assert mmproj_path_from_args(["--other"]) is None

    def test_model_path_dir(self):
        from host.client.llama_cpp_client import model_path_dir

        assert model_path_dir("/models/dir/model.gguf") == "/models/dir"
        assert model_path_dir("") == ""
        assert model_path_dir("model.gguf") == ""
        assert model_path_dir("/models/dir/sub/model.gguf") == "/models/dir/sub"

    def test_pick_best_mmproj_ranks_by_format(self):
        from host.client.llama_cpp_client import ModelInfo, pick_best_mmproj

        models = [
            ModelInfo(id="mmproj-f32", status="unloaded"),
            ModelInfo(id="mmproj-bf16", status="unloaded"),
            ModelInfo(id="mmproj-f16", status="unloaded"),
        ]
        best = pick_best_mmproj(models)
        assert best.id == "mmproj-bf16"

    def test_pick_best_mmproj_falls_back_to_alphabetical(self):
        from host.client.llama_cpp_client import ModelInfo, pick_best_mmproj

        models = [
            ModelInfo(id="z-mmproj", status="unloaded"),
            ModelInfo(id="a-mmproj", status="unloaded"),
        ]
        best = pick_best_mmproj(models)
        assert best.id == "a-mmproj"

    def test_parse_modalities_from_props(self):
        from host.client.llama_cpp_client import parse_modalities_from_props

        assert parse_modalities_from_props({"input": ["text", "image"]}) == ["text", "image"]
        assert parse_modalities_from_props({"vision": True}) == ["text", "image"]
        assert parse_modalities_from_props({"vision": False}) == ["text"]
        assert parse_modalities_from_props({"vision": True, "audio": True}) == [
            "text",
            "image",
            "audio",
        ]
        assert parse_modalities_from_props([]) == []
        assert parse_modalities_from_props({}) == []
        assert parse_modalities_from_props(["text"]) == ["text"]

    def test_model_has_mmproj_configured_from_preset(self):
        from host.client.llama_cpp_client import ModelInfo, model_has_mmproj_configured

        model = ModelInfo(
            id="Qwen-VL",
            status="unloaded",
            preset_ini="mmproj = /models/mmproj.gguf\n",
        )
        assert model_has_mmproj_configured(model) is True

    def test_model_has_mmproj_configured_false(self):
        from host.client.llama_cpp_client import ModelInfo, model_has_mmproj_configured

        model = ModelInfo(id="Qwen-VL", status="unloaded")
        assert model_has_mmproj_configured(model) is False

    def test_find_mmproj_companion_prefers_same_directory(self):
        from host.client.llama_cpp_client import (
            ModelInfo,
            find_mmproj_companion,
        )

        main = ModelInfo(
            id="Qwen3-VL-4B-Instruct-Q4_K_M",
            status="unloaded",
            path="/models/Qwen3-VL-4B-Instruct-Q4_K_M/model.gguf",
        )
        models = [
            main,
            ModelInfo(
                id="Qwen3-VL-4B-mmproj-BF16",
                status="unloaded",
                path="/models/Qwen3-VL-4B-Instruct-Q4_K_M/mmproj-BF16.gguf",
            ),
            ModelInfo(
                id="other-mmproj-BF16",
                status="unloaded",
                path="/models/other/mmproj-BF16.gguf",
            ),
        ]
        companion = find_mmproj_companion(main, models)
        assert companion is not None
        assert companion.id == "Qwen3-VL-4B-mmproj-BF16"

    def test_find_mmproj_companion_no_mmproj_models(self):
        from host.client.llama_cpp_client import ModelInfo, find_mmproj_companion

        main = ModelInfo(id="Qwen3-VL-4B", status="unloaded", path="/models/main.gguf")
        companion = find_mmproj_companion(main, [main])
        assert companion is None

    def test_resolved_model_path_fallback(self):
        from host.client.llama_cpp_client import ModelInfo, resolved_model_path

        # uses path directly
        m1 = ModelInfo(id="m1", status="unloaded", path="/models/m1.gguf")
        assert resolved_model_path(m1) == "/models/m1.gguf"

        # falls back to preset
        m2 = ModelInfo(id="m2", status="unloaded", preset_ini="model = /models/m2.gguf")
        assert resolved_model_path(m2) == "/models/m2.gguf"

        # empty fallback
        m3 = ModelInfo(id="m3", status="unloaded")
        assert resolved_model_path(m3) == ""


# ===========================================================================
# LlamaCppControlPlane — HTTP 客户端
# ===========================================================================


def _make_llama_settings(**llama_cpp_kwargs):
    return Settings(
        target={
            "inference_base_url": "http://edge:8080/v1",
            "model": "Qwen3.5-4B-Q4_K_M",
            "llama_cpp": {"mode": "router", **llama_cpp_kwargs},
        }
    )


class TestControlPlaneHealth:
    def test_control_plane_health_true(self, monkeypatch):
        from host.client.llama_cpp_client import LlamaCppControlPlane

        class _FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"status": "ok"}

        class _FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url, params=None):
                return _FakeResp()

        monkeypatch.setattr("httpx.Client", lambda **kw: _FakeClient())
        ctrl = LlamaCppControlPlane(_make_llama_settings())
        assert ctrl.health() is True

    def test_control_plane_health_false_on_error(self, monkeypatch):
        from host.client.llama_cpp_client import LlamaCppControlPlane

        class _FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url, params=None):
                raise ConnectionError("refused")

        monkeypatch.setattr("httpx.Client", lambda **kw: _FakeClient())
        ctrl = LlamaCppControlPlane(_make_llama_settings())
        assert ctrl.health() is False


class TestControlPlaneListModels:
    def test_list_models_basic(self, monkeypatch):
        from host.client.llama_cpp_client import LlamaCppControlPlane

        payload = {
            "data": [
                {
                    "id": "Qwen3.5-4B-Q4_K_M",
                    "in_cache": True,
                    "path": "/models/Qwen3.5-4B-Q4_K_M.gguf",
                    "status": {"value": "loaded", "args": ["-c", "131072"]},
                }
            ]
        }

        class _FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return payload

        class _FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url, params=None):
                return _FakeResp()

        monkeypatch.setattr("httpx.Client", lambda **kw: _FakeClient())
        ctrl = LlamaCppControlPlane(_make_llama_settings())
        models = ctrl.list_models()
        assert len(models) == 1
        assert models[0].id == "Qwen3.5-4B-Q4_K_M"
        assert models[0].status == "loaded"
        assert models[0].in_cache is True

    def test_list_models_defensive_parsing(self, monkeypatch):
        """验证防御性解析：status 不是 dict、preset_ini 不是 str 等情况。"""
        from host.client.llama_cpp_client import LlamaCppControlPlane

        payload = {
            "data": [
                {
                    "id": "model-1",
                    "status": "not-a-dict",
                    "architecture": None,
                },
                {
                    "id": "model-2",
                    "status": {"value": "loaded"},
                    "architecture": {"input_modalities": ["image"]},
                },
            ]
        }

        class _FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return payload

        class _FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url, params=None):
                return _FakeResp()

        monkeypatch.setattr("httpx.Client", lambda **kw: _FakeClient())
        ctrl = LlamaCppControlPlane(_make_llama_settings())
        models = ctrl.list_models()
        assert len(models) == 2
        assert models[0].status == "unknown"  # status is not a dict
        assert models[1].input_modalities == ["image"]


class TestControlPlaneLoadModel:
    def test_load_model_success(self, monkeypatch):
        from host.client.llama_cpp_client import LlamaCppControlPlane

        posted = []

        class _FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"success": True}

        class _FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def post(self, url, json=None):
                posted.append({"url": url, "json": json})
                return _FakeResp()

        monkeypatch.setattr("httpx.Client", lambda **kw: _FakeClient())
        ctrl = LlamaCppControlPlane(_make_llama_settings())
        result = ctrl.load_model("Qwen3.5-4B-Q4_K_M")
        assert result.success is True
        assert any("/models/load" in p["url"] for p in posted)

    def test_load_model_failure(self, monkeypatch):
        from host.client.llama_cpp_client import LlamaCppControlPlane

        class _FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"success": False, "message": "model file missing"}

        class _FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def post(self, url, json=None):
                return _FakeResp()

        monkeypatch.setattr("httpx.Client", lambda **kw: _FakeClient())
        ctrl = LlamaCppControlPlane(_make_llama_settings())
        result = ctrl.load_model("missing-model")
        assert result.success is False
        assert "model file missing" in result.message

    def test_load_model_handles_http_error(self, monkeypatch):
        """load_model 应捕获 httpx 异常并返回 failure。"""
        from host.client.llama_cpp_client import LlamaCppControlPlane

        class _FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def post(self, url, json=None):
                raise RuntimeError("connection refused")

        monkeypatch.setattr("httpx.Client", lambda **kw: _FakeClient())
        ctrl = LlamaCppControlPlane(_make_llama_settings())
        result = ctrl.load_model("unknown")
        assert result.success is False
        assert "connection refused" in result.message


class TestControlPlaneWaitUntilLoaded:
    def test_wait_until_loaded_success(self, monkeypatch):
        from host.client.llama_cpp_client import LlamaCppControlPlane

        calls = [0]

        class _FakeControl:
            def get_model_status(self, model_id):
                calls[0] += 1
                # First two calls return "loading" (not terminal), third returns "loaded"
                return "loading" if calls[0] < 3 else "loaded"

        ctrl = LlamaCppControlPlane(_make_llama_settings())
        ctrl.get_model_status = _FakeControl().get_model_status  # type: ignore

        assert ctrl.wait_until_loaded("mymodel", timeout_s=10, poll_interval_s=0.01) is True
        assert calls[0] >= 3

    def test_wait_until_loaded_timeout(self, monkeypatch):
        from host.client.llama_cpp_client import LlamaCppControlPlane

        class _FakeControl:
            def get_model_status(self, model_id):
                return "loading"

        ctrl = LlamaCppControlPlane(_make_llama_settings())
        ctrl.get_model_status = _FakeControl().get_model_status  # type: ignore

        assert ctrl.wait_until_loaded("mymodel", timeout_s=0.05, poll_interval_s=0.02) is False

    def test_wait_until_loaded_terminal_state_returns_false(self, monkeypatch):
        """非 transcient 状态（如 unloaded）应直接返回 False。"""
        from host.client.llama_cpp_client import LlamaCppControlPlane

        ctrl = LlamaCppControlPlane(_make_llama_settings())

        def fake_status(model_id):
            return "unloaded"

        ctrl.get_model_status = fake_status  # type: ignore
        assert ctrl.wait_until_loaded("mymodel", timeout_s=10, poll_interval_s=0.01) is False


class TestControlPlaneGetProps:
    def test_get_props(self, monkeypatch):
        from host.client.llama_cpp_client import LlamaCppControlPlane

        payload = {
            "total_slots": 4,
            "model_path": "/models/Qwen.gguf",
            "modalities": {"input": ["text", "image"]},
            "is_sleeping": False,
            "build_info": "llama.cpp build 1234",
        }

        class _FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return payload

        class _FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url, params=None):
                return _FakeResp()

        monkeypatch.setattr("httpx.Client", lambda **kw: _FakeClient())
        ctrl = LlamaCppControlPlane(_make_llama_settings())
        props = ctrl.get_props("Qwen3.5-4B-Q4_K_M")
        assert props.total_slots == 4
        assert props.model_path == "/models/Qwen.gguf"
        assert "image" in props.modalities
        assert props.is_sleeping is False


class TestControlPlaneUnloadModel:
    def test_unload_model_success(self, monkeypatch):
        from host.client.llama_cpp_client import LlamaCppControlPlane

        class _FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"success": True}

        class _FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def post(self, url, json=None):
                return _FakeResp()

        monkeypatch.setattr("httpx.Client", lambda **kw: _FakeClient())
        ctrl = LlamaCppControlPlane(_make_llama_settings())
        assert ctrl.unload_model("my-model") is True

    def test_unload_model_propagates_exception(self, monkeypatch):
        """unload_model 不捕获异常（与 load_model 不同）。"""
        from host.client.llama_cpp_client import LlamaCppControlPlane

        class _FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def post(self, url, json=None):
                raise RuntimeError("server error")

        monkeypatch.setattr("httpx.Client", lambda **kw: _FakeClient())
        ctrl = LlamaCppControlPlane(_make_llama_settings())
        with pytest.raises(RuntimeError, match="server error"):
            ctrl.unload_model("my-model")


# ===========================================================================
# LlamaCppClient — 统一外观
# ===========================================================================


class TestClientResolveModelId:
    def test_resolve_model_id_uses_alias(self):
        from host.client.llama_cpp_client import LlamaCppClient

        settings = Settings(
            target={
                "inference_base_url": "http://edge:8080/v1",
                "model": "Qwen3.5-4B-Q4_K_M",
                "llama_cpp": {
                    "model_aliases": {"Qwen3.5-4B-Q4_K_M": "Qwen/Qwen3.5-4B-GGUF:Q4_K_M"}
                },
            }
        )
        client = LlamaCppClient(settings)
        resolved = client.resolve_model_id("Qwen3.5-4B-Q4_K_M")
        assert resolved == "Qwen/Qwen3.5-4B-GGUF:Q4_K_M"

    def test_resolve_model_id_no_alias(self):
        from host.client.llama_cpp_client import LlamaCppClient

        settings = Settings(
            target={
                "inference_base_url": "http://edge:8080/v1",
                "model": "Qwen3.5-4B-Q4_K_M",
            }
        )
        client = LlamaCppClient(settings)
        assert client.resolve_model_id("custom-model") == "custom-model"


class TestClientRequireModelInCatalog:
    def test_require_model_in_catalog_raises_with_available(self, monkeypatch):
        from host.client.llama_cpp_client import LlamaCppClient, ModelInfo

        class _FakeControl:
            def list_models(self, *, reload=False):
                return [
                    ModelInfo(id="other-model", status="unloaded"),
                    ModelInfo(id="Qwen3.5-4B-mmproj-BF16", status="unloaded"),
                ]

        client = LlamaCppClient(_make_llama_settings())
        client.control = _FakeControl()  # type: ignore

        with pytest.raises(RuntimeError) as excinfo:
            client.require_model_in_catalog("Qwen3.5-4B-Q4_K_M")
        err = str(excinfo.value)
        assert "Qwen3.5-4B-Q4_K_M" in err
        assert "other-model" in err

    def test_require_model_in_catalog_no_available(self, monkeypatch):
        from host.client.llama_cpp_client import LlamaCppClient

        class _FakeControl:
            def list_models(self, *, reload=False):
                return []

        client = LlamaCppClient(_make_llama_settings())
        client.control = _FakeControl()  # type: ignore

        with pytest.raises(RuntimeError) as excinfo:
            client.require_model_in_catalog("missing-model")
        assert "no loadable models found" in str(excinfo.value)


class TestClientEnsureModelLoaded:
    def test_ensure_model_loaded_returns_when_already_loaded(self, monkeypatch):
        from host.client.llama_cpp_client import LlamaCppClient, ModelInfo

        class _FakeControl:
            def list_models(self, *, reload=False):
                return [ModelInfo(id="Qwen3.5-4B-Q4_K_M", status="loaded")]

        client = LlamaCppClient(_make_llama_settings())
        client.control = _FakeControl()  # type: ignore
        assert client.ensure_model_loaded("Qwen3.5-4B-Q4_K_M") == "Qwen3.5-4B-Q4_K_M"

    def test_ensure_model_loaded_rejects_unpaired_mmproj(self, monkeypatch):
        from host.client.llama_cpp_client import LlamaCppClient, ModelInfo

        class _FakeControl:
            def list_models(self, *, reload=False):
                return [
                    ModelInfo(
                        id="Qwen3-VL-4B-Instruct-Q4_K_M",
                        status="unloaded",
                        path="/models/Qwen3-VL-4B-Instruct-Q4_K_M/model.gguf",
                    ),
                    ModelInfo(
                        id="Qwen3-VL-4B-mmproj-BF16",
                        status="unloaded",
                        path="/models/Qwen3-VL-4B-Instruct-Q4_K_M/mmproj-BF16.gguf",
                    ),
                ]

        client = LlamaCppClient(_make_llama_settings())
        client.control = _FakeControl()  # type: ignore

        with pytest.raises(RuntimeError) as excinfo:
            client.ensure_model_loaded("Qwen3-VL-4B-Instruct-Q4_K_M")
        assert "mmproj" in str(excinfo.value).lower()

    def test_ensure_model_loaded_regression_loading_to_unloaded_triggers_active_load(
        self, monkeypatch
    ):
        """核心状态机：模型从 'loading' 回归到 'unloaded'，触发主动加载。"""
        from host.client.llama_cpp_client import LlamaCppClient, LoadModelResult, ModelInfo

        wait_calls = [0]
        state = {"status": "loading", "load_called": False}

        class _FakeControl:
            def list_models(self, *, reload=False):
                return [ModelInfo(id="Qwen3.5-4B-Q4_K_M", status=state["status"])]

            def get_model_status(self, model_id):
                return state["status"]

            def wait_until_loaded(self, model_id, *, timeout_s=300.0, poll_interval_s=2.0):
                wait_calls[0] += 1
                if wait_calls[0] == 1:
                    # First call: simulate loading -> unloaded regression
                    state["status"] = "unloaded"
                    return False
                # Second call: after active load, return success
                return True

            def load_model(self, model_id):
                state["load_called"] = True
                state["status"] = "loaded"
                return LoadModelResult(success=True, message="")

        client = LlamaCppClient(_make_llama_settings())
        client.control = _FakeControl()  # type: ignore

        result = client.ensure_model_loaded("Qwen3.5-4B-Q4_K_M")
        assert result == "Qwen3.5-4B-Q4_K_M"
        assert state["load_called"] is True

    def test_ensure_model_loaded_wait_timeout_raises(self, monkeypatch):
        """轮询超时后正确抛出 RuntimeError。"""
        from host.client.llama_cpp_client import LlamaCppClient, ModelInfo

        class _FakeControl:
            def list_models(self, *, reload=False):
                return [ModelInfo(id="Qwen3.5-4B-Q4_K_M", status="loading")]

            def get_model_status(self, model_id):
                return "loading"

            def wait_until_loaded(self, model_id, *, timeout_s=300.0, poll_interval_s=2.0):
                return False

        client = LlamaCppClient(_make_llama_settings())
        client.control = _FakeControl()  # type: ignore

        with pytest.raises(RuntimeError) as excinfo:
            client.ensure_model_loaded("Qwen3.5-4B-Q4_K_M")
        assert "timed out" in str(excinfo.value)

    def test_ensure_model_loaded_autoload_fallback_on_load_failure(self, monkeypatch):
        """load_model 返回失败后尝试 autoload_fallback。"""
        from host.client.llama_cpp_client import LlamaCppClient, LoadModelResult, ModelInfo

        state = {"load_called": 0}

        class _FakeControl:
            def list_models(self, *, reload=False):
                return [ModelInfo(id="Qwen3.5-4B-Q4_K_M", status="unloaded")]

            def get_model_status(self, model_id):
                return "loaded"  # autoload 成功了

            def load_model(self, model_id):
                state["load_called"] += 1
                return LoadModelResult(success=False, message="model file missing")

        settings = _make_llama_settings(lifecycle={"autoload_fallback": True})
        client = LlamaCppClient(settings)
        client.control = _FakeControl()  # type: ignore

        result = client.ensure_model_loaded("Qwen3.5-4B-Q4_K_M")
        assert result == "Qwen3.5-4B-Q4_K_M"
        assert state["load_called"] == 1

    def test_ensure_model_loaded_autoload_fallback_still_fails(self, monkeypatch):
        """autoload_fallback 仍失败时抛出异常。"""
        from host.client.llama_cpp_client import LlamaCppClient, LoadModelResult, ModelInfo

        class _FakeControl:
            def list_models(self, *, reload=False):
                return [ModelInfo(id="Qwen3.5-4B-Q4_K_M", status="unloaded")]

            def get_model_status(self, model_id):
                return "unloaded"

            def load_model(self, model_id):
                return LoadModelResult(success=False, message="model file missing")

        settings = _make_llama_settings(lifecycle={"autoload_fallback": True})
        client = LlamaCppClient(settings)
        client.control = _FakeControl()  # type: ignore

        with pytest.raises(RuntimeError) as excinfo:
            client.ensure_model_loaded("Qwen3.5-4B-Q4_K_M")
        assert "model file missing" in str(excinfo.value)

    def test_ensure_model_loaded_verifies_image_modality_for_vlm(self, monkeypatch):
        from host.client.llama_cpp_client import LlamaCppClient, ModelInfo, ServerProps

        class _FakeControl:
            def list_models(self, *, reload=False):
                return [
                    ModelInfo(
                        id="Qwen3-VL-4B-Instruct-Q4_K_M",
                        status="loaded",
                        preset_ini="mmproj = /models/mmproj.gguf\n",
                    ),
                ]

            def get_props(self, model_id=None):
                return ServerProps(modalities=["text"])

        client = LlamaCppClient(_make_llama_settings())
        client.control = _FakeControl()  # type: ignore

        with pytest.raises(RuntimeError) as excinfo:
            client.ensure_model_loaded("Qwen3-VL-4B-Instruct-Q4_K_M")
        assert "vision support" in str(excinfo.value)


class TestClientReadinessSnapshot:
    def test_readiness_snapshot_all_ok(self, monkeypatch):
        from host.client.llama_cpp_client import LlamaCppClient, ModelInfo, ServerProps

        class _FakeControl:
            def health(self):
                return True

            def list_models(self):
                return [ModelInfo(id="Qwen3.5-4B-Q4_K_M", status="loaded")]

            def get_props(self, model_id=None):
                return ServerProps(total_slots=2, modalities=["text"])

        client = LlamaCppClient(_make_llama_settings())
        client.control = _FakeControl()  # type: ignore

        snap = client.readiness_snapshot("Qwen3.5-4B-Q4_K_M")
        assert snap["healthy"] is True
        assert snap["model_found"] is True
        assert snap["model_status"] == "loaded"
        assert snap["props_ok"] is True
        assert snap["total_slots"] == 2

    def test_readiness_snapshot_individual_probe_failures(self, monkeypatch):
        """每个探测独立 try/except，一个失败不影响其他。"""
        from host.client.llama_cpp_client import LlamaCppClient

        call_count = {"health": 0, "list_models": 0, "get_props": 0}

        class _FakeControl:
            def health(self):
                call_count["health"] += 1
                raise RuntimeError("health down")

            def list_models(self):
                call_count["list_models"] += 1
                raise RuntimeError("list down")

            def get_props(self, model_id=None):
                call_count["props"] += 1
                raise RuntimeError("props down")

        client = LlamaCppClient(_make_llama_settings())
        client.control = _FakeControl()  # type: ignore

        snap = client.readiness_snapshot("Qwen3.5-4B-Q4_K_M")
        # All probes should still have been attempted
        assert call_count["health"] == 1
        assert call_count["list_models"] == 1
        # All default values
        assert snap["healthy"] is False
        assert snap["model_found"] is False
        assert snap["model_status"] == "unknown"
        assert snap["props_ok"] is False

    def test_readiness_snapshot_with_llama_cpp_disabled(self, monkeypatch):
        """llama_cpp 为 None 时不执行 props 检查。"""
        from host.client.llama_cpp_client import LlamaCppClient, ModelInfo

        class _FakeControl:
            def health(self):
                return True

            def list_models(self):
                return [ModelInfo(id="Qwen3.5-4B-Q4_K_M", status="loaded")]

            def get_props(self, model_id=None):
                raise RuntimeError("should not be called")

        settings = Settings(
            target={
                "inference_base_url": "http://edge:8080/v1",
                "model": "Qwen3.5-4B-Q4_K_M",
            }
        )
        client = LlamaCppClient(settings)
        client.control = _FakeControl()  # type: ignore

        snap = client.readiness_snapshot("Qwen3.5-4B-Q4_K_M")
        assert snap["healthy"] is True
        assert snap["model_found"] is True
        # props_ok should be False since cfg is None; we skip props
        assert snap["props_ok"] is False
