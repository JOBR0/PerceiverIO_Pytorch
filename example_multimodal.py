import os

import cv2
import numpy as np
import scipy.io.wavfile

import torch
import torch.nn.functional as F

from perceiver_io.multimodal_perceiver import MultiModalPerceiver

# Utilities to fetch videos from UCF101 dataset
from utils.kinetics_700_classes import KINETICS_CLASSES

from utils.utils import show_animation


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


def load_video(path, max_frames=0, resize=(224, 224)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames) / 255.0


def save_video(path, data: np.ndarray):
    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    out = cv2.VideoWriter(path, fourcc, 25, (224, 224))
    for frame in data:
        out.write((frame * 255).astype(np.uint8))
    out.release()

def multimodal_example():
    sample_rate, audio = scipy.io.wavfile.read("sample_data/audio.wav")
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 2 ** 15
    elif audio.dtype != np.float32:
        raise ValueError("Unexpected datatype. Model expects sound samples to lie in [-1, 1]")

    video_path = "./sample_data/video.avi"
    video = load_video(video_path)

    # Visualize inputs
    show_animation(video, title="Input Video")
    FRAMES_PER_SECOND = 25
    SAMPLING_RATE = 48000  # Hz
    NUM_FRAMES = 16
    AUDIO_SAMPLES_PER_FRAME = SAMPLING_RATE // FRAMES_PER_SECOND
    SAMPLES_PER_PATCH = 16
    NUM_CLASSES = 700
    IMG_SZ = 224

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    perceiver = MultiModalPerceiver(
        num_frames=NUM_FRAMES,
        audio_samples_per_frame=AUDIO_SAMPLES_PER_FRAME,
        audio_samples_per_patch=SAMPLES_PER_PATCH,
        num_classes=NUM_CLASSES,
        img_size=(IMG_SZ, IMG_SZ),
    )
    perceiver.eval()
    perceiver.to(device)

    ckpt_file = "./pytorch_checkpoints/video_autoencoding_checkpoint.pth"

    # check if file exists
    if not os.path.isfile(ckpt_file):
        raise ValueError("Please download the model checkpoint and place it in /pytorch_checkpoints")

    checkpoint = torch.load(ckpt_file, map_location=device)

    perceiver.load_state_dict(checkpoint["model_state_dict"])

    video_input = torch.from_numpy(video[None, :16]).movedim(-1, -3).float().to(device)
    audio_input = torch.from_numpy(audio[None, :16 * AUDIO_SAMPLES_PER_FRAME, 0:1]).float().to(device)

    # Auto-encode the first 16 frames of the video and one of the audio channels
    with torch.inference_mode():
        reconstruction = perceiver(video_input, audio_input)

    # Save outputs
    scipy.io.wavfile.write("sample_data/audio_reconstr_p1.wav", SAMPLING_RATE,
                           (reconstruction["audio"][0].cpu().numpy() * 2 ** 15).astype(np.int16))
    save_video("./sample_data/video_reconstr_p1.avi",
               np.clip(reconstruction["image"][0].movedim(-3, -1).cpu().numpy(), 0, 1))

    # Kinetics 700 Labels
    scores, indices = torch.topk(F.softmax(reconstruction["label"], dim=-1), 5)

    for score, index in zip(scores[0], indices[0]):
        print(f"{KINETICS_CLASSES[index]}: {score.item() * 100:.1f}%")

    # Visualize reconstruction of first 16 frames
    show_animation(np.clip(reconstruction["image"][0].movedim(-3, -1).cpu().numpy(), 0, 1), title="Reconstruction First 16 Frames")

    # Auto-encode the entire video, one chunk at a time

    # Partial video and audio into 16-frame chunks
    nframes = video.shape[0]
    # Truncate to be divisible by 16
    nframes = nframes - (nframes % 16)

    video_chunks = np.reshape(video[:nframes], [nframes // 16, 16, 224, 224, 3])
    audio_chunks = np.reshape(audio[:nframes * AUDIO_SAMPLES_PER_FRAME],
                              [nframes // 16, 16 * AUDIO_SAMPLES_PER_FRAME, 2])

    with torch.inference_mode():
        reconstruction = {"image": [], "audio": [], "label": []}
        for i in range(nframes // 16):
            print(f"Processing chunk {i}/{nframes // 16}")
            video_input = torch.from_numpy(video_chunks[None, i]).movedim(-1, -3).float().to(device)
            audio_input = torch.from_numpy(audio_chunks[None, i, :, 0:1]).float().to(device)
            output = perceiver(video_input, audio_input)

            reconstruction["image"].append(output["image"])
            reconstruction["audio"].append(output["audio"])
            reconstruction["label"].append(output["label"][:, None])

        reconstruction["image"] = torch.cat(reconstruction["image"], dim=1)
        reconstruction["audio"] = torch.cat(reconstruction["audio"], dim=1)
        reconstruction["label"] = torch.cat(reconstruction["label"], dim=1).mean(dim=1)

    # Save outputs
    scipy.io.wavfile.write("sample_data/audio_reconstr_full.wav", SAMPLING_RATE,
                           (reconstruction["audio"][0].cpu().numpy() * 2 ** 15).astype(np.int16))
    save_video("./sample_data/video_reconstr_full.avi",
               np.clip(reconstruction["image"][0].movedim(-3, -1).cpu().numpy(), 0, 1))

    # Kinetics 700 Labels
    scores, indices = torch.topk(F.softmax(reconstruction["label"], dim=-1), 5)

    for score, index in zip(scores[0], indices[0]):
        print(f"{KINETICS_CLASSES[index]}: {score.item() * 100:.1f}%")

    # Visualize reconstruction of entire video
    show_animation(np.clip(reconstruction["image"][0].movedim(-3, -1).cpu().numpy(), 0, 1), title="Reconstruction Entire Video")

if __name__ == "__main__":
    multimodal_example()
