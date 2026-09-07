from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str,
					help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str,
					help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str,
					help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.',
								default='results/result_voice.mp4')

parser.add_argument('--static', type=bool,
					help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
					default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int,
					help='Batch size for face detection', default=1)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=4)

parser.add_argument('--resize_factor', default=4, type=int,
			help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
					help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
					'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
					help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
					'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
					help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
					'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')

args = parser.parse_args()
args.img_size = 96

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
	args.static = True

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def _fill_missing_detections(predictions):
	"""Forward-fills, then back-fills, any None entries with the nearest
	valid detection so a single missed frame doesn't need special-casing
	downstream - every entry is guaranteed non-None once this returns
	(assuming at least one valid detection exists, which is checked by
	the caller before this runs)."""
	filled = list(predictions)
	last_valid = None
	for i in range(len(filled)):
		if filled[i] is not None:
			last_valid = filled[i]
		elif last_valid is not None:
			filled[i] = last_valid
	last_valid = None
	for i in range(len(filled) - 1, -1, -1):
		if filled[i] is not None:
			last_valid = filled[i]
		elif last_valid is not None:
			filled[i] = last_valid
	return filled

def face_detect(images):
    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D,
        flip_input=False,
        device=device
    )

    batch_size = args.face_det_batch_size

    while True:
        predictions = []

        try:
            for i in range(0, len(images), batch_size):
                batch = np.array(images[i:i + batch_size])
                predictions.extend(detector.get_detections_for_batch(batch))

        except RuntimeError:

            if batch_size == 1:
                del detector
                return None

            batch_size //= 2
            print(f"Recovering from OOM. New batch size: {batch_size}")
            continue

        break

    # No face detected anywhere
    if all(rect is None for rect in predictions):
        print("No face detected in this batch.")
        del detector
        return None

    # Fill missing detections
    if any(rect is None for rect in predictions):
        predictions = _fill_missing_detections(predictions)

    pady1, pady2, padx1, padx2 = args.pads

    results = []

    for rect, image in zip(predictions, images):

        if rect is None:
            h, w = image.shape[:2]
            results.append([
                image,
                (0, h, 0, w)
            ])
            continue

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)

        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([
            image[y1:y2, x1:x2],
            (y1, y2, x1, x2)
        ])

    del detector

    return results
mel_step_size = 16

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device} for inference.")
def _load(checkpoint_path):
	checkpoint = torch.load(checkpoint_path, map_location=device)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()


class FrameStreamer:
	"""Reads frames from disk one at a time instead of loading the whole
	video into memory up front. If more frames are requested than the video
	contains (translated audio runs longer than the source clip), it loops
	back to the start - matching the original script's `i % len(frames)`
	behaviour, just without needing the full frame list resident in RAM."""

	def __init__(self, video_path, resize_factor, rotate, crop):
		self.cap = cv2.VideoCapture(video_path)
		self.resize_factor = resize_factor
		self.rotate = rotate
		self.crop = crop
		self._count = 0
		self.total_frames = None

	def _process(self, frame):
		if self.resize_factor > 1:
			frame = cv2.resize(frame, (frame.shape[1] // self.resize_factor, frame.shape[0] // self.resize_factor))
		if self.rotate:
			frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
		y1, y2, x1, x2 = self.crop
		if x2 == -1: x2 = frame.shape[1]
		if y2 == -1: y2 = frame.shape[0]
		return frame[y1:y2, x1:x2]

	def next(self):
		still_reading, frame = self.cap.read()
		if not still_reading:
			# reached end of video - record the true frame count on the
			# first pass, then loop back to the beginning
			if self.total_frames is None:
				self.total_frames = self._count
			if self.total_frames == 0:
				raise ValueError('No frames could be read from the input video.')
			self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
			self._count = 0
			still_reading, frame = self.cap.read()
			if not still_reading:
				raise ValueError('No frames could be read from the input video.')
		self._count += 1
		return self._process(frame)

	def release(self):
		self.cap.release()


def chunked_batches(streamer, mel_chunks, batch_size, static, single_static_frame=None):
	"""Yields (frames_chunk, mel_chunk) pairs, reading only as many frames
	from disk as each chunk needs - never materializing the whole video."""
	total = len(mel_chunks)
	for start in range(0, total, batch_size):
		chunk_mels = mel_chunks[start:start + batch_size]
		if static:
			frames_chunk = [single_static_frame] * len(chunk_mels)
		else:
			frames_chunk = [streamer.next() for _ in range(len(chunk_mels))]
		yield frames_chunk, chunk_mels


def process_batch(frames_chunk, mel_chunk, model, out,
                  static, static_face_result=None):

    if args.box[0] == -1:

        if not static:
            face_det_results = face_detect(frames_chunk)
        else:
            face_det_results = [static_face_result]

    else:

        y1, y2, x1, x2 = args.box

        face_det_results = [
            [f[y1:y2, x1:x2], (y1, y2, x1, x2)]
            for f in frames_chunk
        ]

    # If face detection failed, write original frames
    if face_det_results is None:
        print("Skipping lip-sync for this batch.")

        for frame in frames_chunk:
            out.write(frame)

        return

    img_batch = []
    mel_batch = []
    frame_batch = []
    coords_batch = []

    for i, mel in enumerate(mel_chunk):

        idx = 0 if static else i % len(frames_chunk)

        frame = frames_chunk[idx].copy()

        face, coords = face_det_results[idx]

        face = cv2.resize(face, (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(mel)
        frame_batch.append(frame)
        coords_batch.append(coords)

    img_batch = np.asarray(img_batch)
    mel_batch = np.asarray(mel_batch)

    img_masked = img_batch.copy()
    img_masked[:, args.img_size // 2:] = 0

    img_batch = np.concatenate(
        (img_masked, img_batch),
        axis=3
    ) / 255.

    mel_batch = np.reshape(
        mel_batch,
        (
            len(mel_batch),
            mel_batch.shape[1],
            mel_batch.shape[2],
            1
        )
    )

    img_batch = torch.FloatTensor(
        np.transpose(img_batch, (0, 3, 1, 2))
    ).to(device)

    mel_batch = torch.FloatTensor(
        np.transpose(mel_batch, (0, 3, 1, 2))
    ).to(device)

    with torch.no_grad():
        pred = model(mel_batch, img_batch)

    pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

    for p, frame, coords in zip(pred, frame_batch, coords_batch):

        y1, y2, x1, x2 = coords

        p = cv2.resize(
            p.astype(np.uint8),
            (x2 - x1, y2 - y1)
        )

        frame[y1:y2, x1:x2] = p

        out.write(frame)
def main():
	if not os.path.isfile(args.face):
		raise ValueError('--face argument must be a valid path to video/image file')

	single_frame = None
	streamer = None
	static_face_result = None

	if args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		single_frame = cv2.imread(args.face)
		fps = args.fps
	else:
		streamer = FrameStreamer(args.face, args.resize_factor, args.rotate, args.crop)
		probe = cv2.VideoCapture(args.face)
		fps = probe.get(cv2.CAP_PROP_FPS)
		probe.release()

	if not args.audio.endswith('.wav'):
		print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')
		subprocess.call(command, shell=True)
		args.audio = 'temp/temp.wav'

	wav = audio.load_wav(args.audio, 16000)
	mel = audio.melspectrogram(wav)
	print(mel.shape)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

	mel_chunks = []
	mel_idx_multiplier = 80. / fps
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	if args.static:
		# Run face detection only once for static image
		static_results = face_detect([single_frame])

		if static_results is None:
			raise ValueError("No face detected in the input image.")

		static_face_result = static_results[0]

	batch_size = args.wav2lip_batch_size

	gen = chunked_batches(
		streamer,
		mel_chunks,
		batch_size,
		args.static,
		single_frame
	)

	model = None
	out = None

	for i, (frames_chunk, mel_chunk) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)))):
		if i == 0:
			model = load_model(args.checkpoint_path)
			print("Model loaded")
			frame_h, frame_w = frames_chunk[0].shape[:-1]
			out = cv2.VideoWriter('temp/result.avi',
									cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

		process_batch(frames_chunk, mel_chunk, model, out, args.static, static_face_result)

	if streamer is not None:
		streamer.release()
	out.release()

	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)
	subprocess.call(command, shell=platform.system() != 'Windows')

if __name__ == '__main__':
	main()