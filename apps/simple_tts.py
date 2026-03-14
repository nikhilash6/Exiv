import torch
import os
import soundfile as sf
from exiv.utils.logging import app_logger
from exiv.components.models.qwen3_tts.constructor import get_qwen3_tts_instance
from exiv.components.models.qwen3_tts.inference.qwen3_tts_model import Qwen3TTSModel
from exiv.server.app_core import App, AppOutputType, Input, Output
from exiv.utils.inputs import ModelInput
from exiv.utils.device import MemoryManager
from exiv.utils.file import get_numbered_filename
from exiv.utils.file_path import FilePaths

def main(**params):
    text = params.get("text")
    instruct = params.get("instruct", "Female, clear voice, natural pace.")
    language = params.get("language", "Auto")
    model_name = params.get("model_name", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    
    app_logger.info(f"Starting TTS with text: {text}")

    # 1. Load Model
    # Resolving path using FilePaths
    model_path_data = FilePaths.get_path(filename=model_name, file_type="checkpoint")
    model_path = model_path_data.path if model_path_data.is_present else model_name
    
    app_logger.info(f"Loading model: {model_name}")
    
    # get_qwen3_tts_instance returns (model, text_tokenizer, speech_tokenizer)
    # It internally handles model loading and tokenizer initialization
    raw_model, text_tokenizer, _ = get_qwen3_tts_instance(
        model_path=model_path,
        force_dtype=torch.float16
    )
    
    # Qwen3TTSModel is the high-level wrapper that provides generation APIs
    tts_model = Qwen3TTSModel(model=raw_model, processor=text_tokenizer)
    
    # 2. Generate Audio
    app_logger.info(f"Generating audio for: '{text}' with instruction: '{instruct}'")
    
    # Use generate_voice_design as it's the version currently supported by the constructor
    wavs, sr = tts_model.generate_voice_design(
        text=text,
        instruct=instruct,
        language=language
    )
    
    # 3. Save Output
    output_dir = FilePaths.get_output_directory()
    output_filename = "qwen3_tts_output.wav"
    output_path = get_numbered_filename(output_dir, output_filename)
    
    # Save the first generated waveform
    sf.write(output_path, wavs[0], sr)
    app_logger.info(f"Audio successfully saved to: {output_path}")
    
    # Return relative path for the frontend/server to resolve
    rel_path = os.path.relpath(output_path, output_dir)
    
    # Clean up memory
    MemoryManager.clear_memory()
    
    return {"1": rel_path}

app = App(
    name="Simple Qwen3 TTS",
    description="Generate speech from text using Qwen3-TTS VoiceDesign model.",
    inputs={
        'text': Input(
            label="Input Text", 
            type="text", 
            default="Hello, I am Qwen3, a next-generation text-to-speech model. How can I help you today?"
        ),
        'instruct': Input(
            label="Voice Instruction", 
            type="text", 
            default="Female, clear voice, natural pace, friendly tone."
        ),
        'language': Input(
            label="Language", 
            type="text", 
            default="Auto"
        ),
        'model_name': ModelInput(
            label="Model Name/Path", 
            categories=["checkpoint"], 
            default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        ),
    },
    outputs=[Output(id=1, type=AppOutputType.AUDIO.value)],
    handler=main
)

if __name__ == "__main__":
    app.run_standalone()
