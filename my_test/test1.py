import torch
from TTS.api import TTS
import os

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set TTS_HOME to the current working directory
os.environ["TTS_HOME"] = os.getcwd()

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
model_dir = os.path.join(os.getenv("TTS_HOME"),"tts",model_name.replace("/", "--"))
if not os.path.exists(model_dir):
    tts = TTS(model_name).to(device)
else:
    tts = TTS(model_path=model_dir,
              config_path=os.path.join(model_dir, "config.json")).to(device)
    # TSS actually need xtts in model name to determine if model is multilingual, lol
    tts.model_name = model_name
    assert tts.is_multi_lingual

# Run TTS
speakers = [
    {
        "speaker_wav" : os.path.join(os.getenv("TTS_HOME"), "click-the-button-female-speaking-vocal_87bpm_E_major.wav"),
        "language" : 'en',
        "text" : "Hello world"
    },
        {
        "speaker_wav" : os.path.join(os.getenv("TTS_HOME"), "japanese-hello_D_major.wav"),
        "language" : 'ja',
        "text" : "こんにちは world"
    },
        {
        "speaker_wav" : os.path.join(os.getenv("TTS_HOME"), "click-the-button-female-speaking-vocal_87bpm_E_major.wav"),
        "language" : 'zh-cn',
        "text" : "世界，你好啊"
    },
        {
        "speaker_wav" : os.path.join(os.getenv("TTS_HOME"), "japanese-hello_D_major.wav"),
        "language" : 'zh-cn',
        "text" : "世界，你好啊"
    },
]

for speaker in speakers:
    speaker_wav = speaker['speaker_wav']
    language = speaker['language']
    text = speaker['text']
    out_wav = os.path.join(os.getenv("TTS_HOME"), f"{language}_{os.path.basename(speaker_wav)}_{text}.wav")

    # ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
    # Text to speech list of amplitude values as output
    # wav = tts.tts(text=text, speaker_wav=speaker_wav, language=language)
    # print(wav)
    # Text to speech to a file
    tts.tts_to_file(**speaker, file_path=out_wav)