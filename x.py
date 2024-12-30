import sys
sys.path.append('./CodeFormer')

import argparse
import os
import cv2
import subprocess
import numpy as np
from gfpgan import GFPGANer
import urllib.request
import torch

def download_model():
    model_path = "GFPGANv1.4.pth"
    if not os.path.exists(model_path):
        print("Downloading GFPGAN model...")
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded successfully!")

def apply_superresolution(image, method="GFPGAN"):
    if method == "GFPGAN":
        download_model()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        try:
            gfpganer = GFPGANer(model_path="GFPGANv1.4.pth", upscale=2, device=device)
            _, _, restored_image = gfpganer.enhance(image_rgb, has_aligned=False, only_center_face=False)
            
            if restored_image is None:
                return image
                
            restored_image_bgr = cv2.cvtColor(restored_image, cv2.COLOR_RGB2BGR)
            return restored_image_bgr
            
        except Exception as e:
            return image
    else:
        raise ValueError("Invalid superresolution method. Currently only 'GFPGAN' is supported.")

def extract_frames(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_frames_to_video(frames, output_path, fps):
    if not frames:
        raise ValueError("No frames to save")
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        if frame is not None:
            out.write(frame)
    out.release()

def process_video(input_video, input_audio, output_video, superres_method):
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if not os.path.exists(input_audio):
        raise FileNotFoundError(f"Input audio not found: {input_audio}")
    
    print("Extracting frames from video...")
    frames = extract_frames(input_video)
    
    if not frames:
        raise ValueError("No frames were extracted from the video")
    
    print("Processing frames...")
    lipsynced_frames = []
    total_frames = len(frames)
    
    for i, frame in enumerate(frames):
        try:
            h, w, _ = frame.shape
            lip_region_h = h // 3
            lip_region_w = w // 3
            
            start_h = h // 3
            start_w = w // 3
            
            lipsynced_area = frame[start_h:start_h+lip_region_h, start_w:start_w+lip_region_w]
            
            if lipsynced_area.size == 0:
                lipsynced_frames.append(frame)
                continue
                
            original_res = lipsynced_area.shape
            super_res_area = apply_superresolution(lipsynced_area, method=superres_method)
            
            if super_res_area is not None and super_res_area.size > 0:
                super_res_area = cv2.resize(super_res_area, (original_res[1], original_res[0]))
                frame[start_h:start_h+lip_region_h, start_w:start_w+lip_region_w] = super_res_area
            
            lipsynced_frames.append(frame)
            print(f"Processed frame {i+1}/{total_frames}")
            
        except Exception as e:
            lipsynced_frames.append(frame)

    print("Saving processed video...")
    original_video = cv2.VideoCapture(input_video)
    fps = original_video.get(cv2.CAP_PROP_FPS)
    original_video.release()
    
    save_frames_to_video(lipsynced_frames, "temp_output.mp4", fps)
    
    print("Combining video and audio...")
    try:
        subprocess.run(["ffmpeg", "-i", "temp_output.mp4", "-i", input_audio, "-c:v", "copy", "-c:a", "aac", output_video], check=True)
    except subprocess.CalledProcessError as e:
        raise
    finally:
        if os.path.exists("temp_output.mp4"):
            os.remove("temp_output.mp4")
    
    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Superresolution for lipsynced videos.")
    parser.add_argument("--superres", type=str, required=True, choices=["GFPGAN"], help="Superresolution method to use.")
    parser.add_argument("-iv", "--input_video", type=str, required=True, help="Input video file.")
    parser.add_argument("-ia", "--input_audio", type=str, required=True, help="Input audio file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output video file.")
    args = parser.parse_args()

    try:
        process_video(args.input_video, args.input_audio, args.output, args.superres)
    except Exception as e:
        print(f"Error: {str(e)}")