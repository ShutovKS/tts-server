# FILE: cli/runtime.py
# VERSION: 1.1.0
# START_MODULE_CONTRACT
#   PURPOSE: Define CLI runtime container with core runtime and CLI-specific state.
#   SCOPE: CLIRuntime dataclass
#   DEPENDS: M-BOOTSTRAP
#   LINKS: M-CLI
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   CLI_MODELS - Mapping of CLI model keys to model specifications
#   CliRuntime - Interactive CLI runtime container
#   run_cli - Launch the interactive CLI workflow loop
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Hardened Windows CLI audio playback by preferring os.startfile over fragile cmd/start shell invocation]
# END_CHANGE_SUMMARY

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
from core.contracts.commands import (
    CustomVoiceCommand,
    VoiceCloneCommand,
    VoiceDesignCommand,
)
from core.infrastructure.audio_io import convert_audio_to_wav_if_needed
from core.models.catalog import EMOTION_EXAMPLES, MODEL_SPECS, SPEAKER_MAP

# Suppress harmless library warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CLI_MODELS = {spec.key: spec for spec in MODEL_SPECS.values()}


# START_CONTRACT: CliRuntime
#   PURPOSE: Drive the interactive CLI experience for TTS, voice design, and voice cloning workflows.
#   INPUTS: { settings: Optional[CliSettings] - optional CLI configuration override }
#   OUTPUTS: { CliRuntime - interactive runtime with loaded services and registries }
#   SIDE_EFFECTS: Builds the shared CLI runtime and retains mutable session state.
#   LINKS: M-CLI
# END_CONTRACT: CliRuntime
class CliRuntime:
    def __init__(self, settings: Optional[CliSettings] = None):
        runtime = build_cli_runtime(settings)
        self.settings = runtime.settings
        self.registry = runtime.core.registry
        self.service = runtime.core.application
        self.backend_registry = runtime.core.backend_registry

    # START_CONTRACT: flush_input
    #   PURPOSE: Clear pending terminal input to avoid accidental buffered responses.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Flushes stdin when terminal controls are available.
    #   LINKS: M-CLI
    # END_CONTRACT: flush_input
    def flush_input(self) -> None:
        try:
            import termios

            termios.tcflush(sys.stdin, termios.TCIOFLUSH)
        except (ImportError, OSError):
            pass

    # START_CONTRACT: clean_memory
    #   PURPOSE: Trigger garbage collection after a CLI workflow completes.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Invokes Python garbage collection.
    #   LINKS: M-CLI
    # END_CONTRACT: clean_memory
    def clean_memory(self) -> None:
        gc.collect()

    # START_CONTRACT: clean_path
    #   PURPOSE: Normalize user-supplied file paths captured from terminal input.
    #   INPUTS: { user_input: str - raw CLI path input }
    #   OUTPUTS: { str - cleaned filesystem path string }
    #   SIDE_EFFECTS: none
    #   LINKS: M-CLI
    # END_CONTRACT: clean_path
    @staticmethod
    def clean_path(user_input: str) -> str:
        path = user_input.strip()
        if len(path) > 1 and path[0] in ["'", '"'] and path[-1] == path[0]:
            path = path[1:-1]
        return path.replace("\\ ", " ")

    # START_CONTRACT: maybe_play_audio
    #   PURPOSE: Play a generated audio file automatically when CLI autoplay is enabled.
    #   INPUTS: { path: Path - saved audio output path }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Launches an OS audio playback command.
    #   LINKS: M-CLI
    # END_CONTRACT: maybe_play_audio
    def maybe_play_audio(self, path: Path) -> None:
        if not self.settings.auto_play_cli:
            return

        if os.name == "nt":
            try:
                os.startfile(str(path))
                return
            except OSError:
                pass

        commands = []
        if sys.platform == "darwin":
            commands.append(["afplay", str(path)])
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

    # START_CONTRACT: display_saved_output
    #   PURPOSE: Print saved output information and optionally play the generated audio.
    #   INPUTS: { saved_path: Optional[Path] - persisted audio output path, backend: Optional[str] - backend label for display }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Writes status text to stdout and may trigger audio playback.
    #   LINKS: M-CLI
    # END_CONTRACT: display_saved_output
    def display_saved_output(
        self, saved_path: Optional[Path], backend: Optional[str] = None
    ) -> None:
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

    # START_CONTRACT: print_runtime_banner
    #   PURPOSE: Show the active backend and selection rationale for the CLI session.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Writes runtime banner text to stdout.
    #   LINKS: M-CLI
    # END_CONTRACT: print_runtime_banner
    def print_runtime_banner(self) -> None:
        selection = self.backend_registry.selection
        backend = self.registry.backend
        print(f"Backend: {backend.label} [{backend.key}]")
        print(f"Selection: {selection.selection_reason}")

    # START_CONTRACT: get_safe_input
    #   PURPOSE: Read interactive text input or load text content from a dragged file.
    #   INPUTS: { prompt: str - prompt shown to the CLI user }
    #   OUTPUTS: { Optional[str] - input text or None when the user exits }
    #   SIDE_EFFECTS: Reads stdin and may read text content from disk.
    #   LINKS: M-CLI
    # END_CONTRACT: get_safe_input
    def get_safe_input(
        self, prompt: str = "\nEnter text (or drag .txt file): "
    ) -> Optional[str]:
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

    # START_CONTRACT: get_saved_voices
    #   PURPOSE: List saved cloned voice profiles available to the CLI.
    #   INPUTS: {}
    #   OUTPUTS: { list[str] - sorted saved voice names }
    #   SIDE_EFFECTS: Reads the configured voices directory.
    #   LINKS: M-CLI
    # END_CONTRACT: get_saved_voices
    def get_saved_voices(self) -> list[str]:
        if not self.settings.voices_dir.exists():
            return []
        voices = [path.stem for path in self.settings.voices_dir.glob("*.wav")]
        return sorted(voices)

    # START_CONTRACT: enroll_new_voice
    #   PURPOSE: Register a new saved voice profile from a user-provided reference recording.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Prompts the user, normalizes audio, and writes voice assets to disk.
    #   LINKS: M-CLI
    # END_CONTRACT: enroll_new_voice
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
            clean_wav_path, converted = convert_audio_to_wav_if_needed(
                raw_path, self.settings
            )
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

    # START_CONTRACT: run_custom_session
    #   PURPOSE: Run the interactive custom-voice synthesis flow for the selected model.
    #   INPUTS: { model_key: str - CLI model selection key }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Prompts the user, submits synthesis requests, and may save or play audio.
    #   LINKS: M-CLI
    # END_CONTRACT: run_custom_session
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

        language = input("Language (default auto): ").strip().lower() or "auto"

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
                        language=language,
                        speaker=speaker,
                        instruct=base_instruct,
                        speed=speed,
                    )
                )
                self.display_saved_output(result.saved_path, result.backend)
            except Exception as exc:
                print(f"Error: {exc}")
        self.clean_memory()

    # START_CONTRACT: run_design_session
    #   PURPOSE: Run the interactive voice-design synthesis flow for the selected model.
    #   INPUTS: { model_key: str - CLI model selection key }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Prompts the user, submits synthesis requests, and may save or play audio.
    #   LINKS: M-CLI
    # END_CONTRACT: run_design_session
    def run_design_session(self, model_key: str) -> None:
        spec = CLI_MODELS[model_key]
        if self.registry.resolve_model_path(spec.folder) is None:
            print("Error: Model not found.")
            return

        print(f"\n--- {spec.public_name} ---")
        instruct = input("Describe the voice: ").strip()
        if not instruct:
            return

        language = input("Language (default auto): ").strip().lower() or "auto"

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
                        language=language,
                        voice_description=instruct,
                    )
                )
                self.display_saved_output(result.saved_path, result.backend)
            except Exception as exc:
                print(f"Error: {exc}")
        self.clean_memory()

    # START_CONTRACT: run_clone_manager
    #   PURPOSE: Run the interactive voice-cloning workflow, including saved voices and quick clone mode.
    #   INPUTS: { model_key: str - CLI model selection key }
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Prompts the user, reads or writes voice assets, and may save or play audio.
    #   LINKS: M-CLI
    # END_CONTRACT: run_clone_manager
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
        language = input("Language (default auto): ").strip().lower() or "auto"

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
                ref_audio, converted = convert_audio_to_wav_if_needed(
                    raw_path, self.settings
                )
            except Exception as exc:
                print(f"Error: {exc}")
                return
            ref_text = input("   Transcript (Optional): ").strip() or "."
        else:
            return

        while True:
            text = self.get_safe_input(
                f"\nText for '{os.path.basename(str(ref_audio))}' (or 'exit'): "
            )
            if text is None:
                break
            print("Cloning...")
            try:
                result = self.service.synthesize_clone(
                    VoiceCloneCommand(
                        text=text,
                        model=spec.folder,
                        save_output=True,
                        language=language,
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

    # START_CONTRACT: main_menu
    #   PURPOSE: Present the top-level CLI menu and dispatch the chosen workflow.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Writes menu text to stdout and reads user input.
    #   LINKS: M-CLI
    # END_CONTRACT: main_menu
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

    # START_CONTRACT: run
    #   PURPOSE: Start the CLI session and keep serving the main menu until exit.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Ensures directories exist, writes to stdout, and reads interactive input.
    #   LINKS: M-CLI
    # END_CONTRACT: run
    def run(self) -> None:
        self.settings.ensure_directories()
        self.print_runtime_banner()
        while True:
            self.main_menu()


# START_CONTRACT: run_cli
#   PURPOSE: Launch the interactive CLI runtime with keyboard interrupt handling.
#   INPUTS: {}
#   OUTPUTS: { None - no return value }
#   SIDE_EFFECTS: Starts an interactive terminal session and writes to stdout.
#   LINKS: M-CLI
# END_CONTRACT: run_cli
def run_cli() -> None:
    try:
        CliRuntime().run()
    except KeyboardInterrupt:
        print("\nExiting...")

__all__ = [
    "CLI_MODELS",
    "CliRuntime",
    "run_cli",
]
