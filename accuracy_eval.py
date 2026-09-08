"""
accuracy_eval.py

Standalone accuracy evaluation for AI-Video-Voice-Translation.

Computes:
  - WER  (Word Error Rate)  -> how accurate the Whisper transcription is
  - BLEU (translation score) -> how accurate the machine translation is

Both metrics need a REFERENCE (ground truth / human-corrected) file to
compare against. The model's own output can't grade itself.

Usage:
    # Word Error Rate: compare Whisper's raw .srt against a corrected version
    python accuracy_eval.py wer --hypothesis whisper_output.srt --reference corrected_transcript.srt

    # BLEU score: compare the machine translation against a human reference translation
    python accuracy_eval.py bleu --hypothesis translated_output.srt --reference human_translation.srt

Install requirements first:
    pip install jiwer sacrebleu
"""

import argparse
import re


def parse_srt(path: str) -> list[str]:
    """Extract just the spoken/translated text lines from an .srt file,
    in order, ignoring index numbers and timestamps."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content.strip())
    lines = []
    for block in blocks:
        block_lines = block.strip().splitlines()
        if len(block_lines) < 3:
            continue
        # block_lines[0] = index, block_lines[1] = timestamp, rest = text
        text = " ".join(block_lines[2:]).strip()
        if text:
            lines.append(text)
    return lines


def join_text(lines: list[str]) -> str:
    return " ".join(lines)


def run_wer(reference_path: str, hypothesis_path: str):
    from jiwer import wer, process_words

    ref_lines = parse_srt(reference_path)
    hyp_lines = parse_srt(hypothesis_path)

    reference_text = join_text(ref_lines)
    hypothesis_text = join_text(hyp_lines)

    error_rate = wer(reference_text, hypothesis_text)
    print(f"Reference segments: {len(ref_lines)}")
    print(f"Hypothesis segments: {len(hyp_lines)}")
    print(f"\nWord Error Rate: {error_rate * 100:.2f}%")
    print(f"(Accuracy ≈ {(1 - error_rate) * 100:.2f}%)")

    # Optional: detailed breakdown (substitutions/deletions/insertions)
    try:
        details = process_words(reference_text, hypothesis_text)
        print(f"\nSubstitutions: {details.substitutions}")
        print(f"Deletions: {details.deletions}")
        print(f"Insertions: {details.insertions}")
        print(f"Hits: {details.hits}")
    except Exception:
        pass  # older jiwer versions may not support process_words


def run_bleu(reference_path: str, hypothesis_path: str):
    import sacrebleu

    ref_lines = parse_srt(reference_path)
    hyp_lines = parse_srt(hypothesis_path)

    if len(ref_lines) != len(hyp_lines):
        print(
            f"WARNING: reference has {len(ref_lines)} segments but hypothesis has "
            f"{len(hyp_lines)}. Segment-by-segment BLEU needs matching counts, "
            f"so falling back to whole-file comparison instead."
        )
        reference_text = join_text(ref_lines)
        hypothesis_text = join_text(hyp_lines)
        score = sacrebleu.corpus_bleu([hypothesis_text], [[reference_text]])
        print(f"\nBLEU score (whole-file): {score.score:.2f}")
        return

    # Segment-by-segment BLEU (more informative — matches subtitle line structure)
    score = sacrebleu.corpus_bleu(hyp_lines, [ref_lines])
    print(f"Reference segments: {len(ref_lines)}")
    print(f"Hypothesis segments: {len(hyp_lines)}")
    print(f"\nBLEU score: {score.score:.2f}")
    print("(0 = no overlap with reference, 100 = perfect match; "
          "30-40+ is generally considered good for machine translation)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate WER or BLEU using .srt files")
    subparsers = parser.add_subparsers(dest="command", required=True)

    wer_parser = subparsers.add_parser("wer", help="Compute Word Error Rate for transcription accuracy")
    wer_parser.add_argument("--hypothesis", required=True, help="Path to Whisper's output .srt")
    wer_parser.add_argument("--reference", required=True, help="Path to human-corrected .srt (ground truth)")

    bleu_parser = subparsers.add_parser("bleu", help="Compute BLEU score for translation accuracy")
    bleu_parser.add_argument("--hypothesis", required=True, help="Path to machine-translated .srt")
    bleu_parser.add_argument("--reference", required=True, help="Path to human reference translation .srt")

    args = parser.parse_args()

    if args.command == "wer":
        run_wer(args.reference, args.hypothesis)
    elif args.command == "bleu":
        run_bleu(args.reference, args.hypothesis)


if __name__ == "__main__":
    main()