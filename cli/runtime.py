# FILE: cli/runtime.py
# VERSION: 1.1.1
# START_MODULE_CONTRACT
#   PURPOSE: Define CLI runtime container with core runtime and CLI-specific state.
#   SCOPE: Interactive CLI runtime orchestration, family/model selection helpers, exported model lookup, and CLI entrypoint
#   DEPENDS: M-BOOTSTRAP, M-CONFIG, M-CONTRACTS, M-INFRASTRUCTURE, M-MODELS
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
#   LAST_CHANGE: [v1.1.1 - Refreshed GRACE module scope, dependencies, and export map to match the current CLI runtime surface]
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
from typing import Iterable, Optional

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
CLI_FAMILY_LABELS = {
    "qwen": "Qwen3-TTS",
    "piper": "Piper",
    "omnivoice": "OmniVoice",
}


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
        self._family_menu_order = ("qwen", "piper", "omnivoice")

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

    @staticmethod
    def _normalize_family(family: str) -> str:
        normalized = family.strip().lower()
        if normalized in {"qwen3-tts", "qwen3_tts", "qwen3tts"}:
            return "qwen"
        return normalized

    def _available_model_specs(self) -> list:
        specs = []
        for spec in CLI_MODELS.values():
            model_path = self.settings.models_dir / spec.folder
            if model_path.exists():
                specs.append(spec)
        return specs

    def _family_specs(self, family_key: str) -> list:
        normalized_family = self._normalize_family(family_key)
        return [
            spec
            for spec in self._available_model_specs()
            if self._normalize_family(spec.family) == normalized_family
        ]

    def _family_specs_for_capability(self, family_key: str, capability: str) -> list:
        return [
            spec
            for spec in self._family_specs(family_key)
            if capability in spec.supported_capabilities
        ]

    def _family_label(self, family_key: str) -> str:
        return CLI_FAMILY_LABELS.get(family_key, family_key.title())

    def _family_actions(self, family_key: str) -> list[tuple[str, str, str]]:
        action_catalog = [
            ("preset_speaker_tts", "Preset Speaker TTS", "custom"),
            ("voice_description_tts", "Voice Design", "design"),
            ("reference_voice_clone", "Voice Clone", "clone"),
        ]
        actions: list[tuple[str, str, str]] = []
        for capability, label, mode in action_catalog:
            if self._family_specs_for_capability(family_key, capability):
                actions.append((capability, label, mode))
        return actions

    def _pick_family(self) -> Optional[str]:
        available_families = [
            family_key
            for family_key in self._family_menu_order
            if self._family_specs(family_key)
        ]
        if not available_families:
            print("No runtime-ready CLI families found in the local models directory.")
            return None

        print("\n" + "=" * 40)
        print(" Multi-Family TTS CLI")
        print("=" * 40)
        for index, family_key in enumerate(available_families, start=1):
            label = self._family_label(family_key)
            capabilities = ", ".join(label for _, label, _ in self._family_actions(family_key))
            print(f"  {index}. {label} — {capabilities}")
        print("\n  q. Exit")

        choice = input("\nSelect family: ").strip().lower()
        if choice == "q":
            raise SystemExit()
        try:
            selected_index = int(choice) - 1
        except ValueError:
            print("Invalid selection.")
            self.flush_input()
            return None
        if selected_index < 0 or selected_index >= len(available_families):
            print("Invalid selection.")
            self.flush_input()
            return None
        return available_families[selected_index]

    def _pick_spec_from_list(self, specs: Iterable, *, prompt: str) -> Optional[str]:
        spec_list = list(specs)
        if not spec_list:
            print("No runtime-ready models available for this flow.")
            return None
        if len(spec_list) == 1:
            spec = spec_list[0]
            print(f"Using model: {spec.public_name} [{spec.folder}]")
            return spec.key

        print("\nAvailable Models:")
        for index, spec in enumerate(spec_list, start=1):
            print(f"  {index}. {spec.public_name} [{spec.folder}]")
        raw_choice = input(prompt).strip().lower()
        if raw_choice in {"q", "back", "b"}:
            return None
        try:
            selected_index = int(raw_choice) - 1
        except ValueError:
            print("Invalid selection.")
            return None
        if selected_index < 0 or selected_index >= len(spec_list):
            print("Invalid selection.")
            return None
        return spec_list[selected_index].key

    def _pick_family_action(self, family_key: str) -> Optional[tuple[str, str, str]]:
        actions = self._family_actions(family_key)
        if not actions:
            print("No supported CLI actions are available for this family.")
            return None

        print("\n" + "-" * 40)
        print(f" {self._family_label(family_key)}")
        print("-" * 40)
        for index, (_, label, _) in enumerate(actions, start=1):
            print(f"  {index}. {label}")
        print("\n  b. Back")
        print("  q. Exit")

        choice = input("\nSelect action: ").strip().lower()
        if choice == "q":
            raise SystemExit()
        if choice in {"b", "back"}:
            return None
        try:
            selected_index = int(choice) - 1
        except ValueError:
            print("Invalid selection.")
            return None
        if selected_index < 0 or selected_index >= len(actions):
            print("Invalid selection.")
            return None
        return actions[selected_index]

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

    @staticmethod
    def _render_exception_text(exc: Exception) -> str:
        rendered = str(exc)
        try:
            rendered.encode(sys.stdout.encoding or "utf-8", errors="strict")
            return rendered
        except Exception:
            return rendered.encode("ascii", errors="replace").decode("ascii")

    def _prompt_language(self) -> str:
        return input("Language (default auto): ").strip().lower() or "auto"

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

    def _prompt_speed(self) -> float:
        print("\nSpeed:")
        print("  1. Normal (1.0x)")
        print("  2. Fast (1.3x)")
        print("  3. Slow (0.8x)")
        speed_choice = input("Choice (1-3): ").strip()
        if speed_choice == "2":
            return 1.3
        if speed_choice == "3":
            return 0.8
        return 1.0

    def _prompt_speaker(self, *, family_key: str) -> str:
        if family_key == "piper":
            return "default"
        if family_key == "omnivoice":
            return "female"
        speaker = "Vivian"
        all_speakers = [name for names in SPEAKER_MAP.values() for name in names]
        print("Available Speakers: " + ", ".join(all_speakers))
        user_choice = input("\nSelect Speaker (Name): ").strip()
        if user_choice in all_speakers:
            speaker = user_choice
        print(f"Using: {speaker}")
        return speaker

    def _prompt_instruct(self, *, family_key: str, capability: str) -> str:
        if family_key == "omnivoice" and capability == "voice_description_tts":
            print("\nOmniVoice design style hints (examples):")
            print("  - female")
            print("  - female, whisper")
            print("  - male, british accent")
            print("  - young adult, moderate pitch")
            return input("Design style tokens: ").strip()
        if capability == "voice_description_tts":
            return input("Describe the voice: ").strip()
        if family_key == "piper":
            return ""
        if family_key == "omnivoice":
            print("\nOmniVoice style hints (examples):")
            print("  - female")
            print("  - male, british accent")
            print("  - whisper")
            print("  - young adult, moderate pitch")
            return input("Voice style (default female): ").strip() or "female"
        print("\nEmotion Examples:")
        for example in EMOTION_EXAMPLES:
            print(f"  - {example}")
        return input("Emotion Instruction: ").strip() or "Normal tone"

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
        if not (self.settings.models_dir / spec.folder).exists():
            print("Error: Model not found.")
            return

        print(f"\n--- {spec.public_name} ---")
        family_key = self._normalize_family(spec.family)
        speaker = self._prompt_speaker(family_key=family_key)
        base_instruct = self._prompt_instruct(
            family_key=family_key,
            capability="preset_speaker_tts",
        )
        speed = self._prompt_speed()
        language = self._prompt_language()

        while True:
            text = self.get_safe_input()
            if text is None:
                break
            print("Generating...")
            try:
                result = self.service.synthesize_custom(
                    CustomVoiceCommand(
                        text=text,
                        model=spec.metadata_id,
                        save_output=True,
                        language=language,
                        speaker=speaker,
                        instruct=base_instruct,
                        speed=speed,
                    )
                )
                self.display_saved_output(result.saved_path, result.backend)
            except Exception as exc:
                print(f"Error: {self._render_exception_text(exc)}")
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
        if not (self.settings.models_dir / spec.folder).exists():
            print("Error: Model not found.")
            return

        print(f"\n--- {spec.public_name} ---")
        instruct = self._prompt_instruct(
            family_key=self._normalize_family(spec.family),
            capability="voice_description_tts",
        )
        if not instruct:
            return

        language = self._prompt_language()

        while True:
            text = self.get_safe_input()
            if text is None:
                break
            print("Generating...")
            try:
                result = self.service.synthesize_design(
                    VoiceDesignCommand(
                        text=text,
                        model=spec.metadata_id,
                        save_output=True,
                        language=language,
                        voice_description=instruct,
                    )
                )
                self.display_saved_output(result.saved_path, result.backend)
            except Exception as exc:
                print(f"Error: {self._render_exception_text(exc)}")
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
        if not (self.settings.models_dir / spec.folder).exists():
            print("Error: Model not found.")
            return

        ref_audio: Optional[Path] = None
        ref_text: Optional[str] = None
        converted = False
        language = self._prompt_language()

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
            ref_text = input("   Transcript (Optional): ").strip() or None
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
                        model=spec.metadata_id,
                        save_output=True,
                        language=language,
                        ref_audio_path=ref_audio,
                        ref_text=ref_text,
                    )
                )
                self.display_saved_output(result.saved_path, result.backend)
            except Exception as exc:
                print(f"Error: {self._render_exception_text(exc)}")
        if sub_choice == "3" and converted and ref_audio and ref_audio.exists():
            ref_audio.unlink(missing_ok=True)
        self.clean_memory()

    def run_piper_session(self) -> None:
        model_key = self._pick_spec_from_list(
            self._family_specs_for_capability("piper", "preset_speaker_tts"),
            prompt="Select Piper model (or 'q' to cancel): ",
        )
        if model_key is None:
            return
        self.run_custom_session(model_key)

    def run_family_action(self, family_key: str, capability: str, mode: str) -> None:
        if family_key == "piper":
            self.run_piper_session()
            return

        model_key = self._pick_spec_from_list(
            self._family_specs_for_capability(family_key, capability),
            prompt="Select model (or 'q' to cancel): ",
        )
        if model_key is None:
            return

        if mode == "custom":
            self.run_custom_session(model_key)
        elif mode == "design":
            self.run_design_session(model_key)
        elif mode == "clone":
            self.run_clone_manager(model_key)

    # START_CONTRACT: main_menu
    #   PURPOSE: Present the top-level CLI menu and dispatch the chosen workflow.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Writes menu text to stdout and reads user input.
    #   LINKS: M-CLI
    # END_CONTRACT: main_menu
    def main_menu(self) -> None:
        family_key = self._pick_family()
        if family_key is None:
            return
        selection = self._pick_family_action(family_key)
        if selection is None:
            return
        capability, _, mode = selection
        self.run_family_action(family_key, capability, mode)

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
