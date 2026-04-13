import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import soundfile as sf


DEFAULT_WAVS_DIR = Path(__file__).parent / "wavs"
DEFAULT_REGISTRY_PATH = Path(__file__).parent / "voice_registry.json"

class VoiceWavManager:
    """
    Manages reference WAV files for voice cloning.

    Features:
    - Looks up transcripts from WAV metadata first, then from a persisted registry.
    - Downloads missing files from remote URLs.
    - Saves new WAVs with metadata embedded and updates the registry.
    """

    def __init__(
        self,
        wavs_dir: Optional[Path] = None,
        registry_path: Optional[Path] = None,
    ):
        self.wavs_dir = Path(wavs_dir) if wavs_dir else DEFAULT_WAVS_DIR
        self.registry_path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
        self.wavs_dir.mkdir(parents=True, exist_ok=True)
        self._registry = self._load_registry()

    # ------------------------------------------------------------------
    # Registry persistence
    # ------------------------------------------------------------------
    def _load_registry(self) -> Dict[str, Dict[str, str]]:
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            raise "voice registry missing"

    def _save_registry(self) -> None:
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._registry, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _read_metadata(filepath: str) -> Optional[Dict[str, str]]:
        """Read metadata (text, tags, language, url) from a WAV file's comment tag."""
        try:
            info = sf.info(filepath)
            comment = getattr(info, "comment", None)
            if comment:
                try:
                    return json.loads(comment)
                except json.JSONDecodeError:
                    return {"text": comment.strip()}
        except Exception:
            pass
        return None

    @staticmethod
    def _write_wav_with_metadata(
        filepath: str,
        audio_array: np.ndarray,
        sample_rate: int,
        text: str,
        tags: str = "",
        language: str = "English",
        url: str = "",
    ) -> None:
        """Write a WAV file with metadata stored as JSON in the comment tag."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        metadata = {
            "text": text,
            "tags": tags,
            "language": language,
            "url": url,
        }
        # Write WAV file - metadata is stored in registry, not embedded in file
        sf.write(filepath, audio_array, sample_rate)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_path(self, audio_id: str) -> str:
        """
        Return the local filesystem path for *audio_id*.
        Downloads from the registered URL if the file is missing.
        """
        entry = self._get_registry_entry(audio_id)
        if entry is None:
            raise KeyError(f"No registry entry for audio_id '{audio_id}'")

        # Determine the actual key used in registry
        normalized = self._normalize_id(audio_id)
        base_id = audio_id.rsplit('.wav', 1)[0] if audio_id.endswith('.wav') else audio_id
        actual_key = normalized if normalized in self._registry else base_id

        filepath = Path(entry.get("filepath", self.wavs_dir / f"{base_id}.wav"))

        if not filepath.exists():
            url = entry.get("url", "")
            if not url:
                raise FileNotFoundError(
                    f"Local file missing for '{audio_id}' and no URL is registered."
                )
            self._download(url, str(filepath))

        return str(filepath)

    def get_text(self, audio_id: str) -> str:
        """
        Return the transcript for *audio_id*.
        Priority:
        1. Read from the WAV file's metadata.
        2. Fall back to the registry dict.
        """
        filepath = self.get_path(audio_id)
        metadata = self._read_metadata(filepath)
        if metadata and metadata.get("text"):
            return metadata["text"]

        entry = self._get_registry_entry(audio_id)
        if entry is None:
            raise ValueError(f"No registry entry for '{audio_id}'")
        text = entry.get("text", "")
        if not text:
            raise ValueError(f"No transcript found in metadata or registry for '{audio_id}'")
        return text

    def get_tags(self, audio_id: str) -> str:
        """Return the tags/description for *audio_id*."""
        entry = self._get_registry_entry(audio_id)
        if entry is None:
            return ""
        return entry.get("tags", "")

    def get_language(self, audio_id: str) -> str:
        """Return the language for *audio_id*."""
        entry = self._get_registry_entry(audio_id)
        if entry is None:
            return "English"
        return entry.get("language", "English")

    def save_wav(
        self,
        audio_id: str,
        audio_array: np.ndarray,
        sample_rate: int,
        text: str,
        url: str = "",
        tags: str = "",
        language: str = "English",
        copy_from: Optional[str] = None,
    ) -> str:
        """
        Save a new reference WAV and register it.

        Args:
            audio_id: Unique identifier for this voice reference.
            audio_array: Audio samples as a numpy array.
            sample_rate: Sample rate of the audio.
            text: Transcript of the audio (embedded in WAV metadata).
            url: Optional remote URL where this file can be fetched later.
            tags: Voice description/tags (e.g., "Male, 30s, professional").
            language: Language code (default: "English").
            copy_from: Optional existing filepath to copy instead of writing from array.

        Returns:
            The local filepath where the WAV was saved.
        """
        filepath = self.wavs_dir / f"{audio_id}.wav"

        if copy_from:
            shutil.copy2(copy_from, filepath)
            # if the source already has metadata, keep it; otherwise overwrite.
            existing_metadata = self._read_metadata(str(filepath))
            if not existing_metadata or not existing_metadata.get("text"):
                data, sr = sf.read(str(filepath), dtype="float32")
                self._write_wav_with_metadata(str(filepath), data, sr, text, tags, language, url)
        else:
            self._write_wav_with_metadata(str(filepath), audio_array, sample_rate, text, tags, language, url)

        self._registry[audio_id] = {
            "url": url,
            "filepath": str(filepath),
            "text": text,
            "tags": tags,
            "language": language,
        }
        self._save_registry()
        return str(filepath)

    def list_voices(self) -> Dict[str, Dict[str, str]]:
        return {k: dict(v) for k, v in self._registry.items()}

    def _normalize_id(self, audio_id: str) -> str:
        # strip .wav if present to get base ID, then ensure it has .wav
        base_id = audio_id.rsplit('.wav', 1)[0] if audio_id.endswith('.wav') else audio_id
        return base_id + '.wav'

    def _get_registry_entry(self, audio_id: str) -> Optional[Dict[str, str]]:
        """Get registry entry for audio_id, trying both normalized and base forms."""
        # Try normalized (with .wav) first
        normalized = self._normalize_id(audio_id)
        if normalized in self._registry:
            return self._registry[normalized]
        # Try base (without .wav)
        base_id = audio_id.rsplit('.wav', 1)[0] if audio_id.endswith('.wav') else audio_id
        if base_id in self._registry:
            return self._registry[base_id]
        return None

    def has_voice(self, audio_id: str) -> bool:
        return self._get_registry_entry(audio_id) is not None

    def add_voice(self, audio_id: str, url: str, text: str, tags: str = "", language: str = "English") -> None:
        audio_id = self._normalize_id(audio_id)
        filepath = self.wavs_dir / audio_id
        self._registry[audio_id] = {
            "url": url,
            "filepath": str(filepath),
            "text": text,
            "tags": tags,
            "language": language,
        }
        self._save_registry()

    def ensure_available(self, audio_id: str) -> str:
        """Ensure the voice file is available locally, downloading if necessary. Returns the filepath."""
        return self.get_path(self._normalize_id(audio_id))

    def delete_voice(self, audio_id: str) -> None:
        audio_id = self._normalize_id(audio_id)
        entry = self._registry.pop(audio_id, None)
        if entry:
            filepath = Path(entry.get("filepath", ""))
            if filepath.exists():
                filepath.unlink()
            self._save_registry()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _download(self, url: str, dest_path: str) -> None:
        """Download *url* to *dest_path*."""
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        print(f"[VoiceWavManager] Downloading {url} -> {dest_path}")
        urllib.request.urlretrieve(url, dest_path)


# ----------------------------------------------------------------------
# Convenience singleton
# ----------------------------------------------------------------------
_default_manager: Optional[VoiceWavManager] = None

def get_manager() -> VoiceWavManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = VoiceWavManager()
    return _default_manager
