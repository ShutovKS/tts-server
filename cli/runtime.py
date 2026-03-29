from __future__ import annotations

import gc
import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Optional

from cli.bootstrap import build_cli_runtime
from cli.runtime_config import CliSettings
from core.contracts.commands import CustomVoiceCommand, VoiceCloneCommand, VoiceDesignCommand
from core.infrastructure.audio_io import convert_audio_to_wav_if_needed
from core.models.catalog import EMOTION_EXAMPLES, MODEL_SPECS, SPEAKER_MAP

# Suppress harmless library warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CLI_MODELS = {spec.key: spec for spec in MODEL_SPECS.values()}


class CliRuntime:
    def __init__(self, settings: Optional[CliSettings] = None):
        runtime = build_cli_runtime(settings)
        self.settings = runtime.settings
        self.registry = runtime.core.registry
        self.service = runtime.core.application
        self.backend_registry = runtime.core.backend_registry

    def flush_input(self) -> None:
        try:
            import termios

            termios.tcflush(sys.stdin, termios.TCIOFLUSH)
        except (ImportError, OSError):
            pass

    def clean_memory(self) -> None:
        gc.collect()

    @staticmethod
    def clean_path(user_input: str) -> str:
        path = user_input.strip()
        if len(path) > 1 and path[0] in ["'", '"'] and path[-1] == path[0]:
            path = path[1:-1]
        return path.replace("\\ ", " ")

    def maybe_play_audio(self, path: Path) -> None:
        if not self.settings.auto_play_cli:
            return

        commands = []
        if sys.platform == "darwin":
            commands.append(["afplay", str(path)])
        elif os.name == "nt":
            commands.append(["cmd", "/c", "start", "", str(path)])
        else:
            if shutil.which("ffplay"):
                commands.append(["ffplay", "-nodisp", "-autoexit", str(path)])
            commands.append(["xdg-open", str(path)])

        for command in commands:
            try:
                subprocess.run(
                    command,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            except FileNotFoundError:
                continue

    def display_saved_output(self, saved_path: Optional[Path], backend: Optional[str] = None) -> None:
        if not saved_path:
            return
        try:
            relative_path = saved_path.relative_to(Path.cwd())
        except ValueError:
            relative_path = saved_path
        if backend:
            print(f"Saved ({backend}): {relative_path}")
        else:
            print(f"Saved: {relative_path}")
        self.maybe_play_audio(saved_path)

    def print_runtime_banner(self) -> None:
        selection = self.backend_registry.selection
        backend = self.registry.backend
        print(f"Backend: {backend.label} [{backend.key}]")
        print(f"Selection: {selection.selection_reason}")

    def get_safe_input(self, prompt: str = "\nEnter text (or drag .txt file): ") -> Optional[str]:
        try:
            raw_input = input(prompt).strip()
            if raw_input.lower() in ["exit", "quit", "q"]:
                return None

            clean_p = self.clean_path(raw_input)
            if os.path.exists(clean_p) and clean_p.endswith(".txt"):
                print(f"Reading from: {os.path.basename(clean_p)}")
                try:
                    with open(clean_p, "r", encoding="utf-8") as file_handle:
                        return file_handle.read().strip()
                except IOError as exc:
                    print(f"Error reading file: {exc}")
                    return None

            return raw_input
        except KeyboardInterrupt:
            self.flush_input()
            return None

    def get_saved_voices(self) -> list[str]:
        if not self.settings.voices_dir.exists():
            return []
        voices = [path.stem for path in self.settings.voices_dir.glob("*.wav")]
        return sorted(voices)

    def enroll_new_voice(self) -> None:
        print("\n--- Enroll New Voice ---")
        self.flush_input()

        name = input("1. Voice name (e.g. Boss, Mom): ").strip()
        if not name:
            return

        safe_name = re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_")

        ref_input = input("2. Drag & Drop Reference File: ").strip()
        raw_path = Path(self.clean_path(ref_input))

        if len(str(raw_path)) > 300 or "\n" in str(raw_path):
            print("Error: Input too long.")
            self.flush_input()
            return

        try:
            clean_wav_path, converted = convert_audio_to_wav_if_needed(raw_path, self.settings)
        except Exception as exc:
            print(f"Error: {exc}")
            return

        print("3. Transcript (important for quality):")
        ref_text = input("   Type EXACTLY what the audio says: ").strip()

        self.settings.voices_dir.mkdir(parents=True, exist_ok=True)
        target_wav = self.settings.voices_dir / f"{safe_name}.wav"
        target_txt = self.settings.voices_dir / f"{safe_name}.txt"

        shutil.copy2(clean_wav_path, target_wav)
        target_txt.write_text(ref_text, encoding="utf-8")

        if converted and clean_wav_path.exists():
            clean_wav_path.unlink(missing_ok=True)

        print(f"Voice saved as '{safe_name}'")

    def run_custom_session(self, model_key: str) -> None:
        spec = CLI_MODELS[model_key]
        if self.registry.resolve_model_path(spec.folder) is None:
            print("Error: Model not found.")
            return

        print(f"\n--- {spec.public_name} ---")
        speaker = "Vivian"
        all_speakers = [name for names in SPEAKER_MAP.values() for name in names]
        print("Available Speakers: " + ", ".join(all_speakers))

        user_choice = input("\nSelect Speaker (Name): ").strip()
        if user_choice in all_speakers:
            speaker = user_choice
        print(f"Using: {speaker}")

        print("\nEmotion Examples:")
        for example in EMOTION_EXAMPLES:
            print(f"  - {example}")
        base_instruct = input("Emotion Instruction: ").strip() or "Normal tone"

        print("\nSpeed:")
        print("  1. Normal (1.0x)")
        print("  2. Fast (1.3x)")
        print("  3. Slow (0.8x)")
        speed_choice = input("Choice (1-3): ").strip()
        speed = 1.0
        if speed_choice == "2":
            speed = 1.3
        elif speed_choice == "3":
            speed = 0.8

        while True:
            text = self.get_safe_input()
            if text is None:
                break
            print("Generating...")
            try:
                result = self.service.synthesize_custom(
                    CustomVoiceCommand(
                        text=text,
                        model=spec.folder,
                        save_output=True,
                        speaker=speaker,
                        instruct=base_instruct,
                        speed=speed,
                    )
                )
                self.display_saved_output(result.saved_path, result.backend)
            except Exception as exc:
                print(f"Error: {exc}")
        self.clean_memory()

    def run_design_session(self, model_key: str) -> None:
        spec = CLI_MODELS[model_key]
        if self.registry.resolve_model_path(spec.folder) is None:
            print("Error: Model not found.")
            return

        print(f"\n--- {spec.public_name} ---")
        instruct = input("Describe the voice: ").strip()
        if not instruct:
            return

        while True:
            text = self.get_safe_input()
            if text is None:
                break
            print("Generating...")
            try:
                result = self.service.synthesize_design(
                    VoiceDesignCommand(
                        text=text,
                        model=spec.folder,
                        save_output=True,
                        voice_description=instruct,
                    )
                )
                self.display_saved_output(result.saved_path, result.backend)
            except Exception as exc:
                print(f"Error: {exc}")
        self.clean_memory()

    def run_clone_manager(self, model_key: str) -> None:
        print("\n--- Voice Cloning Manager ---")
        print("  1. Pick from Saved Voices")
        print("  2. Enroll New Voice")
        print("  3. Quick Clone")
        print("  4. Back")

        sub_choice = input("\nChoice: ").strip()
        if sub_choice == "2":
            self.enroll_new_voice()
            return
        if sub_choice == "4":
            return

        spec = CLI_MODELS[model_key]
        if self.registry.resolve_model_path(spec.folder) is None:
            print("Error: Model not found.")
            return

        ref_audio: Optional[Path] = None
        ref_text: Optional[str] = None
        converted = False

        if sub_choice == "1":
            saved = self.get_saved_voices()
            if not saved:
                print("No saved voices found.")
                return
            print("\nSaved Voices:")
            for index, voice in enumerate(saved, start=1):
                print(f"  {index}. {voice}")
            try:
                selected_index = int(input("\nPick Number: ")) - 1
                if selected_index < 0 or selected_index >= len(saved):
                    print("Invalid selection.")
                    return
                name = saved[selected_index]
                ref_audio = self.settings.voices_dir / f"{name}.wav"
                txt_path = self.settings.voices_dir / f"{name}.txt"
                if txt_path.exists():
                    ref_text = txt_path.read_text(encoding="utf-8").strip()
                print(f"Loaded: {name}")
            except (ValueError, IndexError):
                print("Invalid selection.")
                return
        elif sub_choice == "3":
            ref_input = input("\nDrag Reference Audio: ").strip()
            raw_path = Path(self.clean_path(ref_input))
            try:
                ref_audio, converted = convert_audio_to_wav_if_needed(raw_path, self.settings)
            except Exception as exc:
                print(f"Error: {exc}")
                return
            ref_text = input("   Transcript (Optional): ").strip() or "."
        else:
            return

        while True:
            text = self.get_safe_input(f"\nText for '{os.path.basename(str(ref_audio))}' (or 'exit'): ")
            if text is None:
                break
            print("Cloning...")
            try:
                result = self.service.synthesize_clone(
                    VoiceCloneCommand(
                        text=text,
                        model=spec.folder,
                        save_output=True,
                        ref_audio_path=ref_audio,
                        ref_text=ref_text,
                    )
                )
                self.display_saved_output(result.saved_path, result.backend)
            except Exception as exc:
                print(f"Error: {exc}")
        if sub_choice == "3" and converted and ref_audio and ref_audio.exists():
            ref_audio.unlink(missing_ok=True)
        self.clean_memory()

    def main_menu(self) -> None:
        print("\n" + "=" * 40)
        print(" Qwen3-TTS Manager")
        print("=" * 40)

        print("\n  Pro Models (1.7B - Best Quality)")
        print("  ---------------------------------")
        print("  1. Custom Voice")
        print("  2. Voice Design")
        print("  3. Voice Cloning")

        print("\n  Lite Models (0.6B - Faster)")
        print("  ---------------------------")
        print("  4. Custom Voice")
        print("  5. Voice Design")
        print("  6. Voice Cloning")

        print("\n  q. Exit")

        choice = input("\nSelect: ").strip().lower()

        if choice == "q":
            raise SystemExit()

        if choice not in CLI_MODELS:
            print("Invalid selection.")
            self.flush_input()
            return

        mode = CLI_MODELS[choice].mode
        if mode == "custom":
            self.run_custom_session(choice)
        elif mode == "design":
            self.run_design_session(choice)
        elif mode == "clone":
            self.run_clone_manager(choice)

    def run(self) -> None:
        self.settings.ensure_directories()
        self.print_runtime_banner()
        while True:
            self.main_menu()


def run_cli() -> None:
    try:
        CliRuntime().run()
    except KeyboardInterrupt:
        print("\nExiting...")
