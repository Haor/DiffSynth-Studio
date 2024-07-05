import os
import requests
from tqdm import tqdm
import torch
import argparse
import cv2
import subprocess
import json
import math
from diffsynth import ModelManager, SDVideoPipeline, ControlNetConfigUnit, VideoData, save_video
from diffsynth.extensions.RIFE import RIFESmoother
from moviepy.editor import VideoFileClip, AudioFileClip

def download_model(url, file_path):
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Skipping download.")
        return
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'wb') as f, tqdm(
        desc=file_path,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            f.write(data)
            bar.update(len(data))
    print(f"Download complete: {file_path}")

def download_all_models():
    models = [
        ("https://civitai.com/api/download/models/266360?type=Model&format=SafeTensor&size=pruned&fp=fp16", "models/stable_diffusion/flat2DAnimerge_v45Sharp.safetensors"),
        ("https://hf-mirror.com/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt", "models/AnimateDiff/mm_sd_v15_v2.ckpt"),
        ("https://hf-mirror.com/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth", "models/ControlNet/control_v11p_sd15_lineart.pth"),
        ("https://hf-mirror.com/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth", "models/ControlNet/control_v11f1e_sd15_tile.pth"),
        ("https://hf-mirror.com/lllyasviel/Annotators/resolve/main/sk_model.pth", "models/Annotators/sk_model.pth"),
        ("https://hf-mirror.com/lllyasviel/Annotators/resolve/main/sk_model2.pth", "models/Annotators/sk_model2.pth"),
        ("https://civitai.com/api/download/models/25820?type=Model&format=PickleTensor&size=full&fp=fp16", "models/textual_inversion/verybadimagenegative_v1.3.pt"),
        ("https://r2.114514.pro/flownet.pkl", "models/RIFE/flownet.pkl"),
    ]
    
    for url, file_path in models:
        download_model(url, file_path)

def get_video_info(input_path):
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', input_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Error getting video info: {result.stderr}")
    probe = json.loads(result.stdout)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    if video_stream is None:
        raise Exception('No video stream found')
    return video_stream, probe['format']

def save_video_with_audio(frames, output_path, fps, original_video_path, keep_audio, start_frame, end_frame, save_original):
    # First, save the processed video without audio
    processed_video_path = output_path.rsplit('.', 1)[0] + '_processed.mp4'
    save_video(frames, processed_video_path, fps)
    print(f"Processed video (without audio) saved to: {processed_video_path}")
    
    # Calculate the duration of the processed segment
    segment_duration = (end_frame - start_frame) / fps
    
    try:
        if keep_audio:
            # Load the original video to extract its audio
            original_video = VideoFileClip(original_video_path)
            
            # Load the newly created video (without audio)
            new_video = VideoFileClip(processed_video_path)
            
            if original_video.audio is not None:
                # Extract audio from the original video for the processed segment
                audio = original_video.audio.subclip(start_frame / original_video.fps, end_frame / original_video.fps)
                
                # Set the audio of the new video
                new_video = new_video.set_audio(audio)
            else:
                print("Warning: The original video does not have an audio track. Output video will be silent.")
            
            # Ensure the new video duration matches the processed segment
            new_video = new_video.set_duration(segment_duration)
            
            # Write the final video with the original audio (if available)
            new_video.write_videofile(output_path, codec="libx264", audio_codec="aac" if original_video.audio else None, fps=fps)
            
            # Close the video clips
            original_video.close()
            new_video.close()
        else:
            # If not keeping audio, we still need to adjust the video duration
            video = VideoFileClip(processed_video_path)
            video = video.set_duration(segment_duration)
            video.write_videofile(output_path, codec="libx264", audio=False, fps=fps)
            video.close()
        
        print(f"Final video saved to: {output_path}")
        
        # Remove the processed video if we're not saving the original
        if not save_original:
            os.remove(processed_video_path)
            print(f"Removed intermediate processed video: {processed_video_path}")
    
    except Exception as e:
        print(f"Error occurred while processing the video: {str(e)}")
        if save_original:
            print(f"Using the processed video as the final output: {processed_video_path}")
            os.rename(processed_video_path, output_path)
        else:
            print("Error occurred while processing the video. No output was saved.")

def process_video(args):
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Get video information
    video_stream, format_info = get_video_info(args.input)
    
    # Get original width and height
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    
    # Calculate aspect ratio
    aspect_ratio = width / height
    
    # Calculate new width and height, maintaining aspect ratio, with long side as max_side
    max_side = args.max_side
    if width > height:  # Landscape video
        new_width = max_side
        new_height = int(new_width / aspect_ratio)
    else:  # Portrait video
        new_height = max_side
        new_width = int(new_height * aspect_ratio)
    
    # Ensure width and height are multiples of 64
    new_width = math.floor(new_width / 64) * 64
    new_height = math.floor(new_height / 64) * 64
    
    # Recalculate aspect ratio after rounding
    new_aspect_ratio = new_width / new_height
    
    # Adjust dimensions to maintain aspect ratio if needed
    if abs(new_aspect_ratio - aspect_ratio) > 0.01:  # Allow 1% tolerance
        if width > height:
            new_height = math.floor(new_width / aspect_ratio / 64) * 64
        else:
            new_width = math.floor(new_height * aspect_ratio / 64) * 64
    
    # Final check to ensure both dimensions are multiples of 64
    new_width = max(64, math.floor(new_width / 64) * 64)
    new_height = max(64, math.floor(new_height / 64) * 64)
    
    # Get original frame rate
    original_fps = eval(video_stream['r_frame_rate'])
    
    # Set frame rate to target_fps
    fps = args.fps or original_fps
    
    # Get video duration
    duration = float(format_info['duration'])
    
    # Use the specified max_duration or the full duration of the video
    max_duration = duration if args.max_duration == -1 else min(args.max_duration, duration)
    max_frames = int(fps * max_duration)
    
    # Calculate start and end frames
    start_frame = 0
    end_frame = max_frames

    model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
    model_manager.load_textual_inversions("models/textual_inversion")
    model_manager.load_models([
        "models/stable_diffusion/flat2DAnimerge_v45Sharp.safetensors",
        "models/AnimateDiff/mm_sd_v15_v2.ckpt",
        "models/ControlNet/control_v11p_sd15_lineart.pth",
        "models/ControlNet/control_v11f1e_sd15_tile.pth",
        "models/RIFE/flownet.pkl"
    ])
    
    pipe = SDVideoPipeline.from_model_manager(
        model_manager,
        [
            ControlNetConfigUnit(
                processor_id="lineart",
                model_path="models/ControlNet/control_v11p_sd15_lineart.pth",
                scale=0.5
            ),
            ControlNetConfigUnit(
                processor_id="tile",
                model_path="models/ControlNet/control_v11f1e_sd15_tile.pth",
                scale=0.5
            )
        ]
    )
    
    smoother = RIFESmoother.from_model_manager(model_manager)
    
    video = VideoData(video_file=args.input, height=new_height, width=new_width)
    input_video = [video[i] for i in range(start_frame, end_frame)]
    
    torch.manual_seed(args.seed)
    output_video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        cfg_scale=args.cfg_scale,
        clip_skip=args.clip_skip,
        controlnet_frames=input_video,
        num_frames=len(input_video),
        num_inference_steps=args.steps,
        height=new_height,
        width=new_width,
        animatediff_batch_size=args.batch_size,
        animatediff_stride=args.stride,
        vram_limit_level=args.vram_limit,
    )
    
    if args.use_rife:
        output_video = smoother(output_video)
    
    save_video_with_audio(output_video, args.output, fps, args.input, args.keep_audio, start_frame, end_frame, args.save_original)

    return new_width, new_height, fps, max_duration

def main():
    parser = argparse.ArgumentParser(description="Video Processing Script")
    parser.add_argument("-i", "--input", required=True, help="Path to input video")
    parser.add_argument("-o", "--output", required=True, help="Path for output video")
    parser.add_argument("--prompt", default="masterpiece, best quality, perfect anime illustration, perfect anime still frame, light", help="Prompt for video generation")
    parser.add_argument("--negative_prompt", default="verybadimagenegative_v1.3", help="Negative prompt for video generation")
    parser.add_argument("--cfg_scale", type=float, default=3, help="CFG scale")
    parser.add_argument("--clip_skip", type=int, default=2, help="Clip skip value")
    parser.add_argument("--steps", type=int, default=10, help="Number of inference steps")
    parser.add_argument("--batch_size", type=int, default=32, help="AnimateDiff batch size")
    parser.add_argument("--stride", type=int, default=16, help="AnimateDiff stride")
    parser.add_argument("--vram_limit", type=int, default=0, help="VRAM limit level")
    parser.add_argument("--fps", type=int, default=None, help="Output video FPS (default: same as input)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--use_rife", action="store_true", help="Use RIFE smoother")
    parser.add_argument("--keep_audio", action="store_true", default=True, help="Keep original audio (default: True)")
    parser.add_argument("--download_models", action="store_true", help="Download required models")
    parser.add_argument("--max_side", type=int, default=1024, help="Maximum side length for the output video")
    parser.add_argument("--max_duration", type=float, default=-1, help="Maximum duration of the output video in seconds (default: -1, use full video duration)")
    parser.add_argument("--save_original", action="store_true", help="Save the original processed video without audio")
    
    args = parser.parse_args()
    
    if args.download_models:
        download_all_models()
     
    try:
        new_width, new_height, new_fps, new_duration = process_video(args)
        print(f"Video processing complete! Output saved to {args.output}")
        print(f"New resolution: {new_width}x{new_height}, FPS: {new_fps}, Duration: {new_duration}s")
    except Exception as e:
        print(f"An error occurred during video processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
