"""llama.cpp server control-plane client and unified facade."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Literal

import httpx
from loguru import logger
from openai import OpenAI

from host.config import Settings
from host.config.settings import LlamaCppConfig

ModelStatus = Literal["loaded", "loading", "unloaded", "sleeping", "unknown"]

_MMPROJ_PRESET_KEYS = ("mmproj", "m")
_MMPROJ_ARG_FLAGS = ("--mmproj", "-mm")
_MMPROJ_QUANT_SUFFIX = re.compile(r"-Q\d+_[A-Za-z0-9_]+$")
_MMPROJ_FORMAT_RANK = ("bf16", "f16", "f32")


def is_mmproj_model_id(model_id: str) -> bool:
    return bool(model_id) and "mmproj" in model_id.lower()


def mmproj_path_from_preset(preset_ini: str) -> str | None:
    for line in preset_ini.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("["):
            continue
        key, sep, value = stripped.partition("=")
        if sep and key.strip().lower() in _MMPROJ_PRESET_KEYS:
            path = value.strip()
            if path:
                return path
    return None


def model_path_from_preset(preset_ini: str) -> str | None:
    for line in preset_ini.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("["):
            continue
        key, sep, value = stripped.partition("=")
        if sep and key.strip().lower() in ("model", "m"):
            path = value.strip()
            if path:
                return path
    return None


def parse_modalities_from_props(mod_obj: Any) -> list[str]:
    """Normalize llama.cpp /props modalities to input modality names."""
    if isinstance(mod_obj, list):
        return [str(value) for value in mod_obj if value]
    if not isinstance(mod_obj, dict) or not mod_obj:
        return []
    raw_input = mod_obj.get("input")
    if isinstance(raw_input, list):
        return [str(value) for value in raw_input if value]
    result: list[str] = []
    if mod_obj.get("vision"):
        result.append("image")
    if mod_obj.get("audio"):
        result.append("audio")
    if "vision" in mod_obj or "audio" in mod_obj or result:
        return ["text", *result]
    return []


def mmproj_path_from_args(args: list[str]) -> str | None:
    for index, arg in enumerate(args):
        if arg in _MMPROJ_ARG_FLAGS and index + 1 < len(args):
            path = args[index + 1].strip()
            if path:
                return path
    return None


def model_path_dir(path: str) -> str:
    if not path:
        return ""
    parent = PurePosixPath(path.replace("\\", "/")).parent
    return "" if str(parent) in (".", "") else str(parent)


def pick_best_mmproj(candidates: list["ModelInfo"]) -> "ModelInfo":
    def rank(model: ModelInfo) -> tuple[int, str]:
        lowered = model.id.lower()
        for index, tag in enumerate(_MMPROJ_FORMAT_RANK):
            if tag in lowered:
                return index, model.id
        return len(_MMPROJ_FORMAT_RANK), model.id

    return min(candidates, key=rank)


def resolved_model_path(model: "ModelInfo") -> str:
    return model.path or model_path_from_preset(model.preset_ini) or ""


def resolved_mmproj_path(model: "ModelInfo") -> str:
    return (
        model.path
        or mmproj_path_from_preset(model.preset_ini)
        or mmproj_path_from_args(model.args)
        or ""
    )


def model_has_mmproj_configured(model: "ModelInfo") -> bool:
    return bool(mmproj_path_from_preset(model.preset_ini) or mmproj_path_from_args(model.args))


def find_mmproj_companion(main: "ModelInfo", models: list["ModelInfo"]) -> "ModelInfo | None":
    mmproj_models = [model for model in models if is_mmproj_model_id(model.id)]
    if not mmproj_models:
        return None

    main_dir = model_path_dir(resolved_model_path(main))
    if main_dir:
        same_dir = [
            model
            for model in mmproj_models
            if model_path_dir(resolved_mmproj_path(model)) == main_dir
        ]
        if same_dir:
            return pick_best_mmproj(same_dir)

    stem = _MMPROJ_QUANT_SUFFIX.sub("", main.id)
    if not stem:
        return None
    stem_key = stem.lower().replace("-", "")
    if len(stem_key) < 4:
        return None

    # Require the full stem to appear in the mmproj id — truncating to a short
    # prefix risks pairing models that share a family but differ in scale
    # (e.g. Qwen2.5-VL-3B vs Qwen2.5-VL-7B both start with 'qwen2.5vl').
    by_name = [model for model in mmproj_models if stem_key in model.id.lower().replace("-", "")]
    if len(by_name) == 1:
        return by_name[0]
    if by_name:
        return pick_best_mmproj(by_name)
    return None


# ---------------------------------------------------------------------------
# Lightweight response types — keep llama.cpp JSON off the upper layers.
# ---------------------------------------------------------------------------


@dataclass
class ModelInfo:
    id: str
    status: ModelStatus
    in_cache: bool = False
    path: str = ""
    args: list[str] = field(default_factory=list)
    preset_ini: str = ""
    input_modalities: list[str] = field(default_factory=list)


@dataclass
class ServerProps:
    total_slots: int = 0
    model_path: str = ""
    modalities: list[str] = field(default_factory=list)
    is_sleeping: bool = False
    build_info: str = ""


@dataclass
class LoadModelResult:
    success: bool
    message: str = ""


# ---------------------------------------------------------------------------
# LlamaCppControlPlane — native llama.cpp REST API (root paths, not /v1)
# ---------------------------------------------------------------------------


class LlamaCppControlPlane:
    """HTTP client for llama.cpp server's native control-plane endpoints.

    Uses root-path endpoints (/health, /models, /props) which are
    distinct from the OpenAI-compatible /v1/* data-plane paths.
    """

    def __init__(self, settings: Settings) -> None:
        self._base = settings.target.inference_root_url()
        cfg: LlamaCppConfig | None = settings.target.llama_cpp
        fallback_key = settings.target.resolved_inference_api_key()
        raw_key = cfg.resolved_api_key(fallback_key) if cfg else fallback_key
        headers: dict[str, str] = {}
        if raw_key and raw_key != "EMPTY":
            headers["Authorization"] = f"Bearer {raw_key}"
        self._headers = headers
        self._timeout = settings.target.monitor_timeout_s

    def _url(self, path: str) -> str:
        return f"{self._base}{path}"

    def _get(self, path: str, **params: Any) -> Any:
        with httpx.Client(timeout=self._timeout, headers=self._headers) as c:
            resp = c.get(self._url(path), params=params or None)
            resp.raise_for_status()
            return resp.json()

    def _post(self, path: str, json: dict[str, Any] | None = None) -> Any:
        with httpx.Client(timeout=self._timeout, headers=self._headers) as c:
            resp = c.post(self._url(path), json=json or {})
            resp.raise_for_status()
            return resp.json()

    # --- health ---

    def health(self) -> bool:
        try:
            self._get("/health")
            return True
        except Exception:
            return False

    # --- model management (router mode) ---

    def list_models(self, *, reload: bool = False) -> list[ModelInfo]:
        params: dict[str, Any] = {}
        if reload:
            params["reload"] = 1
        data = self._get("/models", **params)
        results: list[ModelInfo] = []
        for item in data.get("data", []):
            status_obj = item.get("status", {})
            status_val: str = (
                status_obj.get("value", "unknown") if isinstance(status_obj, dict) else "unknown"
            )
            preset_ini = status_obj.get("preset", "") if isinstance(status_obj, dict) else ""
            if not isinstance(preset_ini, str):
                preset_ini = ""
            architecture = item.get("architecture", {})
            input_modalities: list[str] = []
            if isinstance(architecture, dict):
                raw_modalities = architecture.get("input_modalities", [])
                if isinstance(raw_modalities, list):
                    input_modalities = [str(value) for value in raw_modalities]
            path = str(item.get("path", "") or "")
            if not path:
                path = model_path_from_preset(preset_ini) or ""
            results.append(
                ModelInfo(
                    id=item.get("id", ""),
                    status=status_val,  # type: ignore[arg-type]
                    in_cache=item.get("in_cache", False),
                    path=path,
                    args=status_obj.get("args", []) if isinstance(status_obj, dict) else [],
                    preset_ini=preset_ini,
                    input_modalities=input_modalities,
                )
            )
        return results

    def get_model_status(self, model_id: str) -> ModelStatus:
        try:
            models = self.list_models()
        except Exception:
            return "unknown"
        for m in models:
            if m.id == model_id:
                return m.status
        return "unknown"

    def load_model(self, model_id: str) -> LoadModelResult:
        try:
            result = self._post("/models/load", {"model": model_id})
        except Exception as exc:
            return LoadModelResult(success=False, message=str(exc))
        success = bool(result.get("success", False))
        message = str(result.get("message") or result.get("error") or ("" if success else result))
        return LoadModelResult(success=success, message=message)

    def unload_model(self, model_id: str) -> bool:
        result = self._post("/models/unload", {"model": model_id})
        return bool(result.get("success", False))

    def wait_until_loaded(
        self,
        model_id: str,
        *,
        timeout_s: float = 300.0,
        poll_interval_s: float = 2.0,
    ) -> bool:
        """Poll list_models until model reaches 'loaded' or timeout."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status = self.get_model_status(model_id)
            if status == "loaded":
                return True
            if status not in ("loading", "unknown"):
                return False
            time.sleep(poll_interval_s)
        return False

    # --- props ---

    def get_props(self, model_id: str | None = None) -> ServerProps:
        params: dict[str, Any] = {}
        if model_id:
            params["model"] = model_id
        data = self._get("/props", **params)
        modalities = parse_modalities_from_props(data.get("modalities"))
        return ServerProps(
            total_slots=data.get("total_slots", 0),
            model_path=data.get("model_path", ""),
            modalities=modalities,
            is_sleeping=data.get("is_sleeping", False),
            build_info=data.get("build_info", ""),
        )


# ---------------------------------------------------------------------------
# LlamaCppClient — unified facade (data plane + control plane)
# ---------------------------------------------------------------------------


class LlamaCppClient:
    """Unified Target client for llama.cpp router mode.

    Combines the OpenAI-compatible data plane with llama.cpp's native
    control plane (/health, /models, /props).

    HostController and TargetProvisioner depend only on this facade.
    """

    def __init__(self, settings: Settings) -> None:
        self._inference_client = OpenAI(
            base_url=settings.target.inference_base_url.rstrip("/"),
            api_key=settings.target.resolved_inference_api_key(),
        )
        self.control = LlamaCppControlPlane(settings)
        self._settings = settings

    @property
    def model(self) -> str:
        return self._settings.target.model

    def health_check(self) -> bool:
        """Check that the OpenAI-compatible data plane is reachable."""
        try:
            self._inference_client.models.list()
            return True
        except Exception:
            return False

    def resolve_model_id(self, name: str | None = None) -> str:
        """Resolve a configured model name or alias to the router model id."""
        cfg = self._settings.target.llama_cpp
        target = name or self._settings.target.model
        return cfg.resolve_model_id(target) if cfg else target

    def list_models(self, *, reload: bool = False) -> list[ModelInfo]:
        """List models on the Target llama-server router."""
        return self.control.list_models(reload=reload)

    def model_status(self, model_id: str | None = None) -> ModelStatus:
        """Get the status of a model on the Target router."""
        return self.control.get_model_status(self.resolve_model_id(model_id))

    @staticmethod
    def _is_loadable_model(model_id: str) -> bool:
        return bool(model_id) and not is_mmproj_model_id(model_id)

    def _resolve_mmproj_hint(self, hint: str, models: list[ModelInfo]) -> str:
        if hint.endswith(".gguf") or "/" in hint or "\\" in hint:
            return hint
        match = next((model for model in models if model.id == hint), None)
        if match is not None:
            return resolved_mmproj_path(match) or hint
        return hint

    def resolve_mmproj_path(
        self,
        main: ModelInfo,
        models: list[ModelInfo] | None = None,
    ) -> str | None:
        """Return the mmproj path that should accompany a VLM load, if any."""
        if models is None:
            try:
                models = self.control.list_models()
            except Exception:
                models = []

        cfg = self._settings.target.llama_cpp
        if cfg:
            for key in (main.id, self.resolve_model_id(main.id)):
                hint = cfg.model_mmproj.get(key)
                if hint:
                    return self._resolve_mmproj_hint(hint, models)

        if model_has_mmproj_configured(main):
            return mmproj_path_from_preset(main.preset_ini) or mmproj_path_from_args(main.args)

        companion = find_mmproj_companion(main, models)
        if companion is not None:
            return resolved_mmproj_path(companion) or None
        return None

    def _ensure_mmproj_paired(
        self,
        *,
        requested: str,
        match: ModelInfo,
        models: list[ModelInfo],
    ) -> str | None:
        mmproj_path = self.resolve_mmproj_path(match, models)
        if mmproj_path and not model_has_mmproj_configured(match):
            raise RuntimeError(
                f"failed to load model {requested!r} (router id {match.id!r}): "
                f"multimodal projector {mmproj_path!r} must be paired with this model "
                "before load; place the GGUF and mmproj files in the same subdirectory "
                "on the Target (mmproj filename must contain 'mmproj') and run "
                "'jishubench target models --reload', or add mmproj= to the Target "
                f"presets.ini (status={match.status})"
            )
        return mmproj_path

    def _model_supports_image(
        self,
        resolved_id: str,
        *,
        models: list[ModelInfo] | None = None,
    ) -> bool:
        try:
            props = self.control.get_props(resolved_id)
            if "image" in props.modalities:
                return True
        except Exception:
            pass
        if models is None:
            try:
                models = self.control.list_models()
            except Exception:
                models = []
        match = next((model for model in models if model.id == resolved_id), None)
        return bool(match and "image" in match.input_modalities)

    def _verify_mmproj_loaded(
        self,
        *,
        requested: str,
        resolved_id: str,
        mmproj_path: str,
        models: list[ModelInfo] | None = None,
    ) -> None:
        if self._model_supports_image(resolved_id, models=models):
            return
        raise RuntimeError(
            f"failed to load model {requested!r} (router id {resolved_id!r}): "
            f"model loaded without vision support (expected mmproj {mmproj_path!r}); "
            "check Target mmproj pairing"
            " (status=loaded)"
        )

    def _usable_model_ids(self, models: list[ModelInfo] | None = None) -> list[str]:
        if models is None:
            try:
                models = self.control.list_models()
            except Exception:
                return []
        return sorted(m.id for m in models if self._is_loadable_model(m.id))

    def find_model(
        self,
        model_id: str | None = None,
        *,
        rescan: bool = False,
    ) -> ModelInfo | None:
        resolved = self.resolve_model_id(model_id)
        try:
            models = self.control.list_models(reload=rescan)
        except Exception:
            return None
        return next((m for m in models if m.id == resolved), None)

    def require_model_in_catalog(
        self,
        model_id: str | None = None,
        *,
        rescan: bool = True,
    ) -> ModelInfo:
        """Return catalog entry or raise RuntimeError."""
        requested = model_id or self.model
        resolved = self.resolve_model_id(model_id)
        match = self.find_model(model_id, rescan=False)
        if match is None and rescan:
            match = self.find_model(model_id, rescan=True)
        if match is None:
            _available = self._usable_model_ids()
            _parts = [
                f"model {requested!r} (router id {resolved!r}) is not on the Target",
            ]
            if _available:
                _preview = ", ".join(_available[:8])
                _suffix = " …" if len(_available) > 8 else ""
                _parts.append(f"available models: {_preview}{_suffix}")
            else:
                _parts.append("no loadable models found on Target")
            _parts.append("list models: jishubench target models --reload")
            raise RuntimeError("; ".join(_parts))
        return match

    # --- lifecycle ---

    def ensure_model_loaded(self, model_id: str | None = None) -> str:
        """Ensure the requested model is loaded; return resolved router id.

        Raises RuntimeError on failure.
        """
        requested = model_id or self.model
        cfg = self._settings.target.llama_cpp
        match = self.require_model_in_catalog(model_id, rescan=True)
        resolved = match.id
        models = self.control.list_models()
        mmproj_path = self._ensure_mmproj_paired(
            requested=requested,
            match=match,
            models=models,
        )
        timeout = cfg.lifecycle.wait_load_timeout_s if cfg else 300.0
        poll = cfg.lifecycle.poll_interval_s if cfg else 2.0

        def _verify_and_return() -> str:
            if mmproj_path:
                self._verify_mmproj_loaded(
                    requested=requested,
                    resolved_id=resolved,
                    mmproj_path=mmproj_path,
                    models=models,
                )
            return resolved

        if match.status == "loaded":
            return _verify_and_return()

        # If the router is already loading, wait first. If the wait exits early
        # because the state moved to unloaded/sleeping, fall through to an
        # active load attempt below.
        if match.status == "loading":
            if self.control.wait_until_loaded(resolved, timeout_s=timeout, poll_interval_s=poll):
                return _verify_and_return()
            interim_status = self.control.get_model_status(resolved)
            if interim_status == "loaded":
                return _verify_and_return()
            if interim_status in ("loading", "unknown"):
                raise RuntimeError(
                    f"failed to load model {requested!r} (router id {resolved!r}): "
                    f"timed out after {timeout}s while loading"
                    f" (status={interim_status})"
                )
            # state regressed (unloaded/sleeping/etc.) — try active load below

        if mmproj_path:
            logger.info("loading {} with mmproj {}", resolved, mmproj_path)
        else:
            logger.info("loading {}", resolved)
        load_result = self.control.load_model(resolved)
        if not load_result.success:
            if cfg and cfg.lifecycle.autoload_fallback:
                # Router may still autoload on first /v1 request; verify if possible.
                status = self.control.get_model_status(resolved)
                if status == "loaded":
                    return _verify_and_return()
            detail = load_result.message or "POST /models/load returned success=false"
            raise RuntimeError(
                f"failed to load model {requested!r} (router id {resolved!r}): {detail}"
                f" (status={match.status})"
            )

        if not self.control.wait_until_loaded(resolved, timeout_s=timeout, poll_interval_s=poll):
            status = self.control.get_model_status(resolved)
            raise RuntimeError(
                f"failed to load model {requested!r} (router id {resolved!r}): "
                f"load accepted but model did not reach loaded within {timeout}s"
                f" (status={status})"
            )

        final_status = self.control.get_model_status(resolved)
        if final_status != "loaded":
            raise RuntimeError(
                f"failed to load model {requested!r} (router id {resolved!r}): "
                f"load finished but model status is unexpected"
                f" (status={final_status})"
            )
        return _verify_and_return()

    def unload_model(self, model_id: str | None = None) -> bool:
        resolved = self.resolve_model_id(model_id)
        try:
            return self.control.unload_model(resolved)
        except Exception:
            return False

    # --- readiness snapshot (used by HostController.preflight) ---

    def readiness_snapshot(self, model_id: str | None = None) -> dict[str, Any]:
        resolved = self.resolve_model_id(model_id)
        result: dict[str, Any] = {
            "healthy": False,
            "model_found": False,
            "model_status": "unknown",
            "props_ok": False,
            "modalities": [],
        }
        try:
            result["healthy"] = self.control.health()
        except Exception:
            pass
        match: ModelInfo | None = None
        try:
            models = self.control.list_models()
            match = next((m for m in models if m.id == resolved), None)
            if match:
                result["model_found"] = True
                result["model_status"] = match.status
        except Exception:
            pass
        if match and match.status == "loaded":
            try:
                props = self.control.get_props(resolved)
                result["props_ok"] = True
                result["modalities"] = props.modalities or match.input_modalities
                result["total_slots"] = props.total_slots
            except Exception:
                pass
        return result
