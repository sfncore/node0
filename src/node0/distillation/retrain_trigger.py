# Copyright 2025 Pluralis Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Retraining trigger for node0 distillation.

Monitors acceptance rate from the petals monitoring system and triggers a full
retraining cycle when a plateau is detected. After retraining completes, the
new base draft checkpoint is registered and LoRA adapters are retrained.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from hivemind.utils.logging import get_logger


logger = get_logger(__name__)


class RetrainState(str, Enum):
    IDLE = "idle"
    PLATEAU_DETECTED = "plateau_detected"
    RETRAINING_BASE = "retraining_base"
    UPDATING_REGISTRY = "updating_registry"
    RETRAINING_LORA = "retraining_lora"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PlateauSignal:
    """Signal emitted by acceptance rate monitoring when a plateau is detected."""

    acceptance_rate: float
    window_size: int
    variance: float
    timestamp: float = field(default_factory=time.time)

    @property
    def is_plateau(self) -> bool:
        return self.variance < _PLATEAU_VARIANCE_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "acceptance_rate": self.acceptance_rate,
            "window_size": self.window_size,
            "variance": self.variance,
            "timestamp": self.timestamp,
        }


_PLATEAU_VARIANCE_THRESHOLD = 0.001
_DEFAULT_MODELS_DIR = Path.home() / "gt" / ".models"
_DEFAULT_REJECTION_DATA_DIR = Path.home() / "gt" / ".corpus"


@dataclass
class RetrainConfig:
    """Configuration for the retraining trigger."""

    models_dir: Path = field(default_factory=lambda: _DEFAULT_MODELS_DIR)
    rejection_data_dir: Path = field(default_factory=lambda: _DEFAULT_REJECTION_DATA_DIR)
    min_rejection_samples: int = 100
    plateau_cooldown_seconds: float = 3600.0
    checkpoint_prefix: str = "base-draft"


class RetrainTrigger:
    """
    Triggers a full retraining cycle when acceptance rate plateau is detected.

    The retraining cycle:
    1. Receive plateau signal from monitoring (cross-rig, from petals)
    2. Collect latest rejection data
    3. Run base model distillation with rejection data
    4. Register new checkpoint in draft registry
    5. Retrain LoRA adapters on the new base
    """

    def __init__(
        self,
        config: RetrainConfig | None = None,
        distill_fn: Callable[[Path, Path, Path], Path] | None = None,
        lora_retrain_fn: Callable[[Path], None] | None = None,
    ) -> None:
        self.config = config or RetrainConfig()
        self._state = RetrainState.IDLE
        self._last_trigger_time: float = 0.0
        self._distill_fn = distill_fn or _default_distill
        self._lora_retrain_fn = lora_retrain_fn or _default_lora_retrain

    @property
    def state(self) -> RetrainState:
        return self._state

    def on_plateau_signal(self, signal: PlateauSignal) -> bool:
        """
        Handle a plateau signal from acceptance rate monitoring.

        Returns True if a retraining cycle was triggered, False if skipped.
        """
        if self._state not in (RetrainState.IDLE, RetrainState.COMPLETED, RetrainState.FAILED):
            logger.warning(
                "Ignoring plateau signal: retraining already in progress (state=%s)",
                self._state,
            )
            return False

        if not signal.is_plateau:
            logger.debug(
                "Signal variance %.4f above threshold %.4f, not a plateau",
                signal.variance,
                _PLATEAU_VARIANCE_THRESHOLD,
            )
            return False

        elapsed = time.time() - self._last_trigger_time
        if elapsed < self.config.plateau_cooldown_seconds:
            logger.info(
                "Plateau detected but cooldown active (%.0fs remaining)",
                self.config.plateau_cooldown_seconds - elapsed,
            )
            return False

        logger.info(
            "Plateau detected: acceptance_rate=%.4f, variance=%.6f, window=%d",
            signal.acceptance_rate,
            signal.variance,
            signal.window_size,
        )

        self._state = RetrainState.PLATEAU_DETECTED
        self._last_trigger_time = time.time()

        try:
            self._run_retrain_cycle(signal)
            return True
        except Exception:
            self._state = RetrainState.FAILED
            logger.exception("Retraining cycle failed")
            return False

    def _run_retrain_cycle(self, signal: PlateauSignal) -> None:
        """Execute the full retraining cycle."""
        rejection_data = self._collect_rejection_data()
        if rejection_data is None:
            logger.warning("Insufficient rejection data, aborting retrain cycle")
            self._state = RetrainState.IDLE
            return

        # Step 1: Retrain base model via distillation
        self._state = RetrainState.RETRAINING_BASE
        current_checkpoint = self._find_current_checkpoint()
        new_checkpoint = self._distill_fn(current_checkpoint, rejection_data, self.config.models_dir)
        logger.info("Base model distillation complete: %s", new_checkpoint)

        # Step 2: Update draft registry
        self._state = RetrainState.UPDATING_REGISTRY
        self._register_checkpoint(new_checkpoint, signal)
        logger.info("Draft registry updated with new checkpoint")

        # Step 3: Retrain LoRA adapters on new base
        self._state = RetrainState.RETRAINING_LORA
        self._lora_retrain_fn(new_checkpoint)
        logger.info("LoRA adapters retrained on new base")

        self._state = RetrainState.COMPLETED
        logger.info("Retraining cycle completed successfully")

    def _collect_rejection_data(self) -> Path | None:
        """Gather rejection data from the corpus directory."""
        rejection_dir = self.config.rejection_data_dir
        if not rejection_dir.exists():
            logger.warning("Rejection data directory does not exist: %s", rejection_dir)
            return None

        rejection_files = list(rejection_dir.glob("**/*.json"))
        if len(rejection_files) < self.config.min_rejection_samples:
            logger.info(
                "Only %d rejection samples (need %d)",
                len(rejection_files),
                self.config.min_rejection_samples,
            )
            return None

        return rejection_dir

    def _find_current_checkpoint(self) -> Path:
        """Find the current base draft checkpoint."""
        checkpoints = sorted(
            self.config.models_dir.glob(f"{self.config.checkpoint_prefix}-*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not checkpoints:
            raise FileNotFoundError(
                f"No existing checkpoint with prefix '{self.config.checkpoint_prefix}' "
                f"in {self.config.models_dir}"
            )
        return checkpoints[0]

    def _register_checkpoint(self, checkpoint_path: Path, signal: PlateauSignal) -> None:
        """Register the new checkpoint in the draft registry via Dolt."""
        metadata = {
            "checkpoint_path": str(checkpoint_path),
            "trigger_signal": signal.to_dict(),
            "registered_at": time.time(),
            "previous_acceptance_rate": signal.acceptance_rate,
        }
        registry_file = checkpoint_path / "registry_metadata.json"
        registry_file.write_text(json.dumps(metadata, indent=2))
        logger.info("Registry metadata written to %s", registry_file)


def _default_distill(
    current_checkpoint: Path,
    rejection_data: Path,
    output_dir: Path,
) -> Path:
    """
    Default distillation function using subprocess.

    Invokes the node0 training script to produce a new base checkpoint
    from the current checkpoint + rejection data.
    """
    timestamp = int(time.time())
    new_checkpoint_dir = output_dir / f"base-draft-{timestamp}"
    new_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "node0.distillation.train",
        "--base-checkpoint",
        str(current_checkpoint),
        "--rejection-data",
        str(rejection_data),
        "--output-dir",
        str(new_checkpoint_dir),
    ]

    logger.info("Starting distillation: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

    if result.returncode != 0:
        logger.error("Distillation failed:\nstdout: %s\nstderr: %s", result.stdout, result.stderr)
        raise RuntimeError(f"Distillation process exited with code {result.returncode}")

    return new_checkpoint_dir


def _default_lora_retrain(new_checkpoint: Path) -> None:
    """
    Default LoRA retraining function.

    Retrains all LoRA adapters against the new base checkpoint.
    """
    cmd = [
        "python",
        "-m",
        "node0.distillation.lora_retrain",
        "--base-checkpoint",
        str(new_checkpoint),
    ]

    logger.info("Starting LoRA retraining: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

    if result.returncode != 0:
        logger.error("LoRA retrain failed:\nstdout: %s\nstderr: %s", result.stdout, result.stderr)
        raise RuntimeError(f"LoRA retrain process exited with code {result.returncode}")
