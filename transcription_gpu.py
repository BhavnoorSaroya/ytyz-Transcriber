import os; import sys; import time
import logging; import warnings; import shutil
import contextlib; import argparse
import subprocess; import json
import whisperx; import torch
# from pydub import AudioSegment
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator
from lightning.pytorch.utilities import disable_possible_user_warnings

WordDict = Dict[str, Any]
DEVICE: str = "cuda"
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN")
VERBOSE: bool = False

if not HF_TOKEN:
    print("missing huggingface token. `set/export HF_TOKEN=token`")
    sys.exit(1)

if not torch.cuda.is_available():
    print("cuda is unavailable. https://developer.nvidia.com/cuda-downloads")
    sys.exit(1)

def check_ffmpeg():
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found. https://ffmpeg.org/download.html")
        sys.exit(1)

def suppress_logs() -> None:
    if VERBOSE: return
    disable_possible_user_warnings()
    warnings.filterwarnings("ignore")
    logging.getLogger("pyannote").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("whisperx").setLevel(logging.ERROR)

@contextlib.contextmanager
def suppress_everything() -> Iterator[None]:
    if VERBOSE: yield; return
    with open(os.devnull, "w") as devnull:
        original_stdout = sys.stdout; original_stderr = sys.stderr
        try: sys.stdout = devnull; sys.stderr = devnull; yield
        finally: sys.stdout = original_stdout; sys.stderr = original_stderr

def format_timestamp(total_seconds: float) -> str:
    hours: int = int(total_seconds // 3600)
    minutes: int = int((total_seconds % 3600) // 60)
    seconds: int = int(total_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def format_duration(seconds: float) -> str:
    if seconds < 60: return f"{seconds:.1f} second{'s' if int(seconds) != 1 else ''}"
    seconds = int(seconds)
    minutes: int = int(seconds // 60)
    remaining_seconds: float = seconds % 60
    if minutes < 60: return f"{minutes} minute{'s' if minutes != 1 else ''} and {int(remaining_seconds)} seconds"
    hours: int = int(minutes // 60)
    minutes = minutes % 60
    return f"{hours} hour{'s' if hours != 1 else ''}, {minutes} minute{'s' if minutes != 1 else ''} and {int(remaining_seconds)} second{'s' if int(remaining_seconds) != 1 else ''}"

def convert_to_wav(input_path: Path) -> Path:
    check_ffmpeg()
    wav_output_path = input_path.with_suffix(".wav")
    if not input_path.is_file():
        print(f"'{input_path}' not found")
        sys.exit(1)
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        # "-ac", "1",
        "-ar", "16000",          # 16 kHz
        "-sample_fmt", "s16",    # 16-bit
        "-c:a", "pcm_s16le",
        str(wav_output_path),
    ]
    try:
        with suppress_everything(): ffmpeg_proc = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ffmpeg_proc.returncode != 0: raise RuntimeError(ffmpeg_proc.stderr.decode(errors="ignore"))
        if VERBOSE: print(f"{input_path} converted to {wav_output_path}")
        return wav_output_path
    except Exception as e:
        print(f"file conversion failed: {e}")
        sys.exit(1)

def transcribe_audio(audio_path: Path, model) -> WordDict: # after model ", device: str"
    with suppress_everything():
        # whisper_model = whisperx.load_model(model_name, device, compute_type="float16")
        with torch.inference_mode(): result: WordDict = model.transcribe(str(audio_path), language="en", task="transcribe")
    return result

def align_transcription(segments, audio_path: Path, device: str) -> WordDict:
    with suppress_everything():
        align_model, align_metadata = whisperx.load_align_model(language_code="en", device=device)
        alignment_result: WordDict = whisperx.align(segments, align_model, align_metadata, str(audio_path), device)
    return alignment_result

def run_diarization(audio_path: Path, hf_token: str, device: str):
    with suppress_everything():
        diarization_pipeline = whisperx.diarize.DiarizationPipeline(
           # model_name='pyannote/speaker-diarization@2.1',
            model_name='pyannote/speaker-diarization-3.1',
            use_auth_token=hf_token,
            device=device
        )
        try: diarization_pipeline.set_params({"clustering": {"threshold": 0.70}})
        except Exception: pass
        diarization_result = diarization_pipeline(str(audio_path))
    return diarization_result

def postprocess_segments(alignment_result: WordDict, diarization_result, speaker_gap_threshold: float = 0.8) -> List[WordDict]:
    aligned_segments: List[WordDict] = alignment_result.get("segments") or []
    cleaned_segments: List[WordDict] = []
    for segment in aligned_segments:
        text: str = (segment.get("text") or "").strip()
        if not text: continue
        start_time: float = float(segment.get("start", 0.0))
        end_time: float = float(segment.get("end", start_time))
        if end_time > start_time: cleaned_segments.append({"start": start_time, "end": end_time, "text": text})
    if not cleaned_segments: return []

    speaker_segments: List[WordDict] = []
    try:
        for time_span, _, speaker_label in diarization_result.itertracks(yield_label=True):
            speaker_segments.append({"start": float(time_span.start), "end": float(time_span.end), "label": str(speaker_label)})
    except Exception:
        if hasattr(diarization_result, "iterrows"):
            for _, row in diarization_result.iterrows():
                speaker_segments.append({
                    "start": float(row["start"]),
                    "end": float(row["end"]),
                    "label": str(row.get("speaker", row.get("label", "UNKNOWN")))
                })
        elif isinstance(diarization_result, dict) and "segments" in diarization_result:
            for row in diarization_result["segments"]:
                speaker_segments.append({
                    "start": float(row.get("start", 0.0)),
                    "end": float(row.get("end", 0.0)),
                    "label": str(row.get("speaker") or row.get("label") or "UNKNOWN")
                })
    speaker_segments.sort(key=lambda span: (span["start"], span["end"]))

    def speaker_for_interval(start_time: float, end_time: float) -> str:
        best_label: Optional[str]
        max_overlap_seconds: float
        best_label, max_overlap_seconds = None, 0.0
        for span in speaker_segments:
            if span["end"] <= start_time: continue
            if span["start"] >= end_time: break
            overlap_seconds: float = max(0.0, min(end_time, span["end"]) - max(start_time, span["start"]))
            if overlap_seconds > max_overlap_seconds: max_overlap_seconds, best_label = overlap_seconds, span["label"]
        if best_label is not None: return best_label
        midpoint_second: float = 0.5 * (start_time + end_time)
        for span in speaker_segments:
            if span["start"] <= midpoint_second < span["end"]: return span["label"]
            if span["start"] > midpoint_second: break
        return "UNKNOWN"

    labeled_segments: List[WordDict] = []
    for segment in cleaned_segments:
        speaker_label: str = speaker_for_interval(segment["start"], segment["end"])
        labeled_segments.append({"start": segment["start"], "end": segment["end"], "speaker": speaker_label, "text": segment["text"]})

    utterances: List[WordDict] = []
    for segment in labeled_segments:
        if utterances and utterances[-1]["speaker"] == segment["speaker"] and (segment["start"] - utterances[-1]["end"]) <= speaker_gap_threshold:
            last_utterance: WordDict = utterances[-1]

            # if segment["text"] and segment["text"][0] in ".,!?;:%)]}": last_utterance["text"] += segment["text"]
            # else:
            #     if last_utterance["text"] and not last_utterance["text"].endswith(" "): last_utterance["text"] += " "
            #     last_utterance["text"] += segment["text"]
            # last_utterance["end"] = segment["end"]

            next_segment = segment["text"] or ""
            if last_utterance["text"] and not last_utterance["text"].endswith((" ", "\n", "—", "-", "(", "[", "{", "“", "'")) and not (next_segment and next_segment[0] in ".,!?;:%)]}"):
                last_utterance["text"] += " "
            last_utterance["text"] += next_segment.lstrip()
            last_utterance["end"] = segment["end"]
            
        else: utterances.append(segment)

    return utterances

def transcribe_with_diarization(audio_path: Path, model, output_path: Path, output_format: str) -> None:
    print(f"\nprocessing: {audio_path}")
    start_time: float = time.perf_counter()

    try:
        print("transcribing")
        transcription_result: WordDict = transcribe_audio(audio_path, model) # after model ", DEVICE"
        # torch.cuda.empty_cache()

        print("aligning")
        alignment_result: WordDict = align_transcription(transcription_result["segments"], audio_path, DEVICE)
        # torch.cuda.empty_cache()

        print("diarizing")
        diarization_result = run_diarization(audio_path, HF_TOKEN, DEVICE)
        # torch.cuda.empty_cache()

        print("post processing")
        utterance_lines = postprocess_segments(alignment_result, diarization_result)

    except Exception as e: print(f"{e}"); sys.exit(1)

    end_time: float = time.perf_counter()
    if output_format == "json": file_body = json.dumps({"utterances": utterance_lines}, ensure_ascii=False, indent=2)
    else:
        formatted_lines = [f"[{format_timestamp(utterance_line['start'])}] {utterance_line['speaker']}: {utterance_line['text']}" for utterance_line in utterance_lines]
        file_body = "\n".join(formatted_lines)

    output_path.write_text(file_body, encoding="utf-8")
    print(f"saved to: {output_path}")
    print(f"\ntook {format_duration(end_time-start_time)}")

def overwrite(output_path: Path, overwrite_flag: bool) -> Path:
    if not output_path.exists() or overwrite_flag: return output_path

    # response = input(f"'{output_path}' already exists. overwrite? [y/n]: ").strip().lower()
    # if response == "y": return output_path

    def bump(path: Path) -> Path:
        base = path.stem
        suffix = path.suffix
        new_base = base + "_new"
        candidate = path.with_name(new_base + suffix)

        if candidate.exists(): return bump(candidate)
        return candidate
    
    return bump(output_path)

def delete_wav(wav_path: Path):
    if wav_path.exists():
        try: wav_path.unlink()
        except Exception as e: print(f"could not delete wav. {e}")

if __name__ == "__main__":
    # total_start_time: float = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--model", default="medium")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-k", "--keep-wav", action="store_true")
    parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument('-f', '--out_format', choices=['txt','json'], default='txt')
    args = parser.parse_args()

    VERBOSE = args.verbose
    suppress_logs()

    input_path = Path(args.input_file)
    if not input_path.is_file():
        print(f"{input_path} does not exist")
        sys.exit(1)

    extension = input_path.suffix.lower()
    if extension == ".wav":
        wav_path = input_path
        wav_converted = False
    else:
        wav_path = convert_to_wav(input_path)
        wav_converted = True

    base_stem = wav_path.stem
    ext = ".json" if args.out_format == "json" else ".txt"
    output_path = Path(f"{base_stem}{ext}")
    output_path = overwrite(output_path, args.overwrite)

    whisper_model = None
    if args.model != parser.get_default("model"):
        try:
            with suppress_everything(): whisper_model = whisperx.load_model(args.model, DEVICE, compute_type="float16")
            # torch.cuda.empty_cache()
        except Exception as e:
            print(f"model '{args.model}' failed to load. {e}")
            sys.exit(1)
    else:
        try:
            with suppress_everything(): whisper_model = whisperx.load_model("medium", DEVICE, compute_type="float16")
            # torch.cuda.empty_cache()
        except Exception as e:
            print(f"model '{args.model}' failed to load. {e}")
            sys.exit(1)

    try: transcribe_with_diarization(wav_path, whisper_model, output_path, args.out_format)
    finally: 
        # total_end_time: float = time.perf_counter()
        # print(f"\ntotal {format_duration(total_end_time-total_start_time)}")
        if wav_converted and not args.keep_wav: delete_wav(wav_path)
