"""Wake word detection with openWakeWord.

Runs locally on every 80 ms frame -- this is the only model on the Pi, and the
reason nothing leaves the house until Faethon is actually addressed.

The pretrained models land in openWakeWord's own resources directory inside the
venv. That means a fresh machine downloads them once on first run, which is the
behaviour we want: nothing extra to copy around.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import openwakeword
from openwakeword.model import Model
from openwakeword.utils import download_models

log = logging.getLogger(__name__)


def _ensure_model(name: str) -> str:
    """Resolve a config wake-word value to something Model() accepts.

    Accepts a filesystem path, a pretrained key ("hey_jarvis"), or a versioned
    filename stem ("hey_jarvis_v0.1").
    """
    if Path(name).exists():
        return name

    stem_to_key = {
        Path(meta["download_url"]).stem: key
        for key, meta in openwakeword.MODELS.items()
    }
    key = stem_to_key.get(name, name)
    if key not in openwakeword.MODELS:
        raise ValueError(
            f"Unknown wake word model {name!r}. "
            f"Pretrained options: {sorted(openwakeword.MODELS)}. "
            f"For a custom model, give the path to a .tflite or .onnx file."
        )

    path = Path(openwakeword.MODELS[key]["model_path"])
    if not path.exists():
        log.info("downloading wake word model %s (first run only)", key)
        download_models(model_names=[path.stem])
    return str(path)


class WakeWordDetector:
    """Feeds frames to openWakeWord and reports crossings of the threshold.

    A refractory period after each detection stops one spoken wake word from
    firing several times as its score decays.
    """

    def __init__(
        self,
        model_name: str,
        threshold: float = 0.5,
        refractory_sec: float = 2.0,
    ) -> None:
        model_path = _ensure_model(model_name)
        self.threshold = threshold
        self.refractory_sec = refractory_sec
        self._last_fire = 0.0

        # openWakeWord picks its runtime from this, not from the file, so a
        # mismatch fails at load. Custom-trained models are commonly exported
        # as .onnx while the stock pretrained ones ship as .tflite.
        framework = "onnx" if model_path.lower().endswith(".onnx") else "tflite"

        log.info("loading wake word model: %s (%s)", model_path, framework)
        self._model = Model(
            wakeword_models=[model_path], inference_framework=framework
        )
        # openWakeWord keys scores by model name, not by the path we passed.
        self.labels = list(self._model.models.keys())
        log.info("wake word ready: %s (threshold %.2f)", self.labels, threshold)

    def reset(self) -> None:
        """Clear the model's internal audio buffer.

        Call after Faethon finishes speaking: otherwise the tail of its own reply
        is still sitting in the feature buffer and can trigger a detection.
        """
        self._model.reset()
        self._last_fire = time.monotonic()

    def process(self, frame: bytes) -> float | None:
        """Feed one frame. Returns the score if the wake word fired, else None."""
        samples = np.frombuffer(frame, dtype=np.int16)
        scores = self._model.predict(samples)

        best = max(scores.values()) if scores else 0.0
        if best < self.threshold:
            return None

        now = time.monotonic()
        if now - self._last_fire < self.refractory_sec:
            return None

        self._last_fire = now
        log.info("wake word detected (score %.3f)", best)
        return best
