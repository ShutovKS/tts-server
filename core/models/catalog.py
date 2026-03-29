from __future__ import annotations

from dataclasses import dataclass


SPEAKER_MAP = {
    "English": ["Ryan", "Aiden", "Ethan", "Chelsie", "Serena", "Vivian"],
    "Chinese": ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"],
    "Japanese": ["Ono_Anna"],
    "Korean": ["Sohee"],
}

EMOTION_EXAMPLES = [
    "Sad and crying, speaking slowly",
    "Excited and happy, speaking very fast",
    "Angry and shouting",
    "Whispering quietly",
]

@dataclass(frozen=True)
class ModelSpec:
    key: str
    public_name: str
    folder: str
    mode: str
    output_subfolder: str

    @property
    def api_name(self) -> str:
        return self.folder


MODEL_SPECS = {
    "1": ModelSpec(
        key="1",
        public_name="Custom Voice",
        folder="Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        mode="custom",
        output_subfolder="CustomVoice",
    ),
    "2": ModelSpec(
        key="2",
        public_name="Voice Design",
        folder="Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
        mode="design",
        output_subfolder="VoiceDesign",
    ),
    "3": ModelSpec(
        key="3",
        public_name="Voice Cloning",
        folder="Qwen3-TTS-12Hz-1.7B-Base-8bit",
        mode="clone",
        output_subfolder="Clones",
    ),
    "4": ModelSpec(
        key="4",
        public_name="Custom Voice",
        folder="Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
        mode="custom",
        output_subfolder="CustomVoice",
    ),
    "5": ModelSpec(
        key="5",
        public_name="Voice Design",
        folder="Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit",
        mode="design",
        output_subfolder="VoiceDesign",
    ),
    "6": ModelSpec(
        key="6",
        public_name="Voice Cloning",
        folder="Qwen3-TTS-12Hz-0.6B-Base-8bit",
        mode="clone",
        output_subfolder="Clones",
    ),
}
