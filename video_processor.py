import subprocess
import os
import sys
import asyncio
import whisper
import datetime
import time
from deep_translator import GoogleTranslator
import edge_tts
import concurrent.futures
from functools import partial

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

XTTS_SUPPORTED_LANGUAGES = {"hi", "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"}

_xtts_model = None
_xtts_device = "cpu"

def _patch_xtts_audio_loading():
    """The TTS library's xtts.load_audio() calls torchaudio.load(), which on
    some Windows installs tries to use the torchcodec backend and fails with
    'Could not load libtorchcodec' because the matching FFmpeg DLLs aren't
    present. We replace that one helper with a librosa-based loader, which
    sidesteps torchaudio entirely and gives back the same tensor shape."""
    import torch
    import librosa
    import TTS.tts.models.xtts as xtts_module

    def _safe_load_audio(audiopath, sampling_rate):
        audio_np, _ = librosa.load(audiopath, sr=sampling_rate, mono=True)
        audio = torch.from_numpy(audio_np).unsqueeze(0).float()
        audio.clip_(-1, 1)
        return audio

    xtts_module.load_audio = _safe_load_audio

def _patch_xtts_number_expansion():
    """XTTS's text tokenizer spells out numbers (e.g. '3.5' -> 'three point
    five') using the num2words library, in the target language. num2words
    doesn't support decimal/number expansion for several of the languages
    this app offers (Marathi, Tamil, Telugu, Bengali, Gujarati aren't in its
    supported set) and raises NotImplementedError instead of degrading
    gracefully - which currently crashes the whole generation job over a
    single number in the subtitles. We wrap it so an unsupported language
    falls back to English number words rather than aborting."""
    import TTS.tts.layers.xtts.tokenizer as tokenizer_module
    original_num2words = tokenizer_module.num2words

    def _safe_num2words(number, lang=None, **kwargs):
        try:
            return original_num2words(number, lang=lang, **kwargs)
        except NotImplementedError:
            try:
                return original_num2words(number, lang='en', **kwargs)
            except Exception:
                return str(number)

    tokenizer_module.num2words = _safe_num2words

def get_xtts_model():
    global _xtts_model, _xtts_device
    if _xtts_model is None:
        print("Loading XTTS model (lazy-load)...")
        os.environ["COQUI_TOS_AGREED"] = "1"
        _patch_xtts_audio_loading()
        _patch_xtts_number_expansion()
        from TTS.api import TTS
        import torch
        _xtts_device = "cuda" if torch.cuda.is_available() else "cpu"
        if _xtts_device == "cpu":
            print("WARNING: XTTS is loading on CPU. Each line of speech can take "
                  "several seconds to generate. A CUDA GPU is strongly recommended.")
        _xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(_xtts_device)
        print("XTTS model loaded successfully.")
    return _xtts_model

def unload_xtts_model():
    global _xtts_model
    if _xtts_model is not None:
        print("Unloading XTTS model to free VRAM...")
        try:
            _xtts_model.to("cpu")
        except Exception:
            pass
        del _xtts_model
        _xtts_model = None
    _xtts_latents_cache.clear()
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

# Cache of precomputed speaker conditioning latents, keyed by speaker_wav path.
# Computing these requires running the speaker-encoder over the reference audio,
# which is expensive - we do it once per job instead of once per subtitle segment.
_xtts_latents_cache = {}

def _get_xtts_latents(model, speaker_wav):
    if speaker_wav not in _xtts_latents_cache:
        xtts = model.synthesizer.tts_model
        gpt_cond_latent, speaker_embedding = xtts.get_conditioning_latents(audio_path=speaker_wav)
        _xtts_latents_cache[speaker_wav] = (gpt_cond_latent, speaker_embedding)
    return _xtts_latents_cache[speaker_wav]

def _tts_to_file_xtts(text, speaker_wav, language, output_path):
    os.environ["COQUI_TOS_AGREED"] = "1"
    model = get_xtts_model()
    xtts = model.synthesizer.tts_model
    gpt_cond_latent, speaker_embedding = _get_xtts_latents(model, speaker_wav)

    out = xtts.inference(
        text=text,
        language=language,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
    )

    import numpy as np
    import soundfile as sf
    wav = np.asarray(out["wav"], dtype=np.float32)
    sf.write(output_path, wav, 24000)


LANGUAGE_MAPPINGS = {
    "hi": "Hindi",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "gu": "Gujarati",
}

EDGE_TTS_VOICES = {
    "hi": "hi-IN-SwaraNeural",
    "mr": "mr-IN-AarohiNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "bn": "bn-IN-BashkarNeural",
    "gu": "gu-IN-NiranjanNeural",
}

def time_to_sec(t):
    t = t.replace(",", ".")
    h, m, s = t.split(":")
    return float(h) * 3600 + float(m) * 60 + float(s)

def sec_to_time(sec):
    return str(datetime.timedelta(seconds=sec))[:12].replace(".", ",")

def extract_audio(video_path, output_audio_path):
    # check=True raises CalledProcessError on failure (caught in app.py)
    subprocess.run([
        "ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", "-y", output_audio_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_audio_path

_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model

def unload_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        print("Unloading Whisper model to free VRAM...")
        del _whisper_model
        _whisper_model = None
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def transcribe_audio(audio_path):
    model = get_whisper_model()
    result = model.transcribe(audio_path)
    segments = []
    for i, seg in enumerate(result["segments"]):
        start = str(datetime.timedelta(seconds=seg["start"]))[:12].replace(".", ",")
        end = str(datetime.timedelta(seconds=seg["end"]))[:12].replace(".", ",")
        segments.append({
            "id": i + 1,
            "start": start,
            "end": end,
            "text": seg['text'].strip()
        })
    return segments

def translate_single_segment(seg, target_language):
    translator = GoogleTranslator(source="auto", target=target_language)
    max_retries = 3
    translated_text = seg['text']
    for attempt in range(max_retries):
        try:
            translated_text = translator.translate(seg['text'])
            break
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Translation error: {e}")
            time.sleep(1)

    return {
        "id": seg["id"],
        "start": seg["start"],
        "end": seg["end"],
        "text": translated_text
    }

def translate_segments(segments, target_language):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        func = partial(translate_single_segment, target_language=target_language)
        results = list(executor.map(func, segments))
    return results

async def _tts_to_file(text, voice, output_path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def adjust_audio_speed(input_path, output_path, target_duration):
    cmd = ["ffprobe", "-i", input_path, "-show_entries", "format=duration",
           "-v", "quiet", "-of", "csv=p=0"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        actual_duration = float(result.stdout.strip())
    except ValueError:
        actual_duration = target_duration

    if actual_duration <= 0 or target_duration <= 0:
        import shutil
        shutil.copy(input_path, output_path)
        return

    ratio = actual_duration / target_duration

    # FIX #8: atempo only supports 0.5–2.0 per filter; chain filters for ratio > 2.0
    # e.g. ratio=3.0 → atempo=2.0,atempo=1.5
    if ratio < 0.5:
        ratio = 0.5

    if ratio <= 2.0:
        filter_str = f"atempo={ratio:.4f}"
    else:
        # Chain multiple atempo=2.0 stages, then a remainder stage
        filter_parts = []
        remaining = ratio
        while remaining > 2.0:
            filter_parts.append("atempo=2.0")
            remaining /= 2.0
        filter_parts.append(f"atempo={remaining:.4f}")
        filter_str = ",".join(filter_parts)

    subprocess.run([
        "ffmpeg", "-i", input_path, "-filter:a", filter_str, "-y", output_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

_ffmpeg_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# XTTS is a single heavy neural-net instance - running multiple inferences on it
# concurrently just makes threads fight over the same CPU/GPU resources, which is
# slower than running them one at a time. Edge-TTS is a lightweight network call
# and stays fully parallel.
_xtts_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

async def process_segment_audio(i, seg, target_language, work_dir, tts_engine='edge', speaker_wav=None):
    start_s = time_to_sec(seg["start"])
    end_s = time_to_sec(seg["end"])
    duration = end_s - start_s

    if duration <= 0:
        return None

    raw_tts_file = os.path.join(work_dir, f"raw_tts_{i}.mp3")
    adjusted_tts_file = os.path.join(work_dir, f"adj_tts_{i}.mp3")

    actual_engine = tts_engine
    if tts_engine == 'xtts' and target_language not in XTTS_SUPPORTED_LANGUAGES:
        print(f"XTTS voice cloning not supported for language '{target_language}'. Falling back to Edge-TTS.")
        actual_engine = 'edge'

    if actual_engine == 'xtts' and speaker_wav and os.path.exists(speaker_wav):
        raw_tts_wav = os.path.join(work_dir, f"raw_tts_{i}.wav")
        loop = asyncio.get_running_loop()
        # Run on the dedicated single-worker XTTS executor so segments are
        # generated one at a time instead of contending for the same model.
        await loop.run_in_executor(
            _xtts_executor,
            _tts_to_file_xtts,
            seg["text"],
            speaker_wav,
            target_language,
            raw_tts_wav
        )
        raw_tts_file = raw_tts_wav
    else:
        voice = EDGE_TTS_VOICES.get(target_language, "hi-IN-SwaraNeural")
        await _tts_to_file(seg["text"], voice, raw_tts_file)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_ffmpeg_executor, adjust_audio_speed, raw_tts_file, adjusted_tts_file, duration)

    return i, start_s, end_s, adjusted_tts_file

async def process_all_segments_audio(segments, target_language, work_dir, tts_engine='edge', speaker_wav=None):
    if tts_engine == 'xtts' and speaker_wav and os.path.exists(speaker_wav):
        # Load the model and compute the speaker's conditioning latents once,
        # up front, instead of letting the first segment(s) pay for it and
        # instead of recomputing it per segment.
        loop = asyncio.get_running_loop()
        print("Preparing XTTS model and speaker voice profile (one-time cost)...")
        await loop.run_in_executor(_xtts_executor, get_xtts_model)
        model = get_xtts_model()
        await loop.run_in_executor(_xtts_executor, _get_xtts_latents, model, speaker_wav)

    tasks = [process_segment_audio(i, seg, target_language, work_dir, tts_engine, speaker_wav) for i, seg in enumerate(segments)]
    return await asyncio.gather(*tasks)

def run_wav2lip(video_path, audio_path, output_path):
    print(f"Running Wav2Lip on {video_path} with audio {audio_path}...")

    wav2lip_dir = os.path.join(BASE_DIR, "wav2lip")
    checkpoint_path = os.path.join(wav2lip_dir, "checkpoints", "wav2lip_gan.pth")
    inference_script = os.path.join(wav2lip_dir, "inference.py")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Wav2Lip checkpoint not found at {checkpoint_path}"
        )

    temp_dir = os.path.join(wav2lip_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # -------------------------------------------------------
    # Resize video to 720p maximum width (better for low VRAM GPUs, preserves quality for smaller inputs)
    # -------------------------------------------------------
    resized_video = os.path.join(temp_dir, "resized_input.mp4")

    print("Resizing input video...")

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vf", "scale=min(720\\,iw):-2",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",
        resized_video
    ], check=True)

    video_path = resized_video

    # Get width to compute dynamic resize factor
    import cv2
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    # Clear CUDA cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    resize_factors = [1, 2]
    last_error = None

    for resize in resize_factors:

        print(f"Trying resize_factor={resize}")

        cmd = [
            sys.executable,
            inference_script,

            "--checkpoint_path", checkpoint_path,
            "--face", video_path,
            "--audio", audio_path,
            "--outfile", output_path,

            "--resize_factor", str(resize),
            "--face_det_batch_size", "1",
            "--wav2lip_batch_size", "8"
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                cwd=wav2lip_dir
            )

            print("Wav2Lip completed successfully.")
            return

        except subprocess.CalledProcessError as e:

            print(f"Failed with resize_factor={resize}")
            last_error = e

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # -------------------------------------------------------
    # Final fallback: Run on CPU
    # -------------------------------------------------------
    print("GPU failed. Trying CPU...")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    cpu_resize = max(1, width // 320)
    print(f"Running CPU fallback with dynamic resize_factor={cpu_resize}")

    cmd = [
        sys.executable,
        inference_script,
        "--checkpoint_path", checkpoint_path,
        "--face", video_path,
        "--audio", audio_path,
        "--outfile", output_path,
        "--resize_factor", str(cpu_resize),
        "--face_det_batch_size", "1",
        "--wav2lip_batch_size", "4"
    ]

    subprocess.run(
        cmd,
        check=True,
        cwd=wav2lip_dir,
        env=env
    )

    print("Wav2Lip completed on CPU.")
def merge_audio_video(video_path, segments, target_language, output_video_path, work_dir, tts_engine='edge', lip_sync=False):
    speaker_wav = os.path.join(work_dir, "audio.wav")
    results = asyncio.run(process_all_segments_audio(segments, target_language, work_dir, tts_engine, speaker_wav))

    # Free GPU VRAM from XTTS or Whisper models after audio generation completes
    unload_xtts_model()
    unload_whisper_model()

    valid_results = [r for r in results if r is not None]
    valid_results.sort(key=lambda x: x[0])

    audio_files = []
    prev_end = 0

    for i, start_s, end_s, adjusted_tts_file in valid_results:
        if start_s > prev_end:
            silence_duration = start_s - prev_end
            silence_file = os.path.join(work_dir, f"silence_{i}_{int(start_s * 1000)}.mp3")
            subprocess.run([
                "ffmpeg", "-f", "lavfi",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=24000",
                "-t", str(silence_duration),
                "-y", silence_file
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            audio_files.append(silence_file)

        audio_files.append(adjusted_tts_file)
        prev_end = end_s

    concat_file = os.path.join(work_dir, "concat.txt")
    with open(concat_file, "w") as f:
        for file in audio_files:
            file_path = file.replace("\\", "/")
            f.write(f"file '{file_path}'\n")

    final_audio = os.path.join(work_dir, "final_audio.mp3")

    subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0",
        "-i", concat_file,
        "-c:a", "libmp3lame", "-q:a", "2",
        "-y", final_audio
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if lip_sync:
        try:
            run_wav2lip(video_path, final_audio, output_video_path)
        except Exception as e:
            print(f"Wav2Lip failed: {e}. Falling back to standard merge.")
            subprocess.run([
                "ffmpeg", "-i", video_path, "-i", final_audio,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-shortest", "-y", output_video_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run([
            "ffmpeg", "-i", video_path, "-i", final_audio,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-shortest", "-y", output_video_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output_video_path

def generate_srt(segments, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments):
            f.write(f"{i+1}\n{seg['start']} --> {seg['end']}\n{seg['text']}\n\n")
    return output_path