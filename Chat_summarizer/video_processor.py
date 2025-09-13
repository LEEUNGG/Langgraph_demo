import os
import cv2
import numpy as np
import tempfile
import requests
import time
import random
import base64
from dotenv import load_dotenv
from openai import OpenAI
import openai
import io
from PIL import Image
import google.generativeai as genai

# Load .env file from parent directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

def resize_images(image_paths, max_width=512, max_height=512):
    """
    Resize images to avoid oversized files
    """
    resized_image_paths = []
    
    try:
        for image_path in image_paths:
            # Open image
            img = Image.open(image_path)
            
            # Get original dimensions
            width, height = img.size
            
            # Calculate resize ratio
            width_ratio = max_width / width
            height_ratio = max_height / height
            resize_ratio = min(width_ratio, height_ratio)
            
            # Resize if needed
            if resize_ratio < 1:
                new_width = int(width * resize_ratio)
                new_height = int(height * resize_ratio)
                
                # Resize image
                img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Create temporary file to save resized image
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_resized_file:
                    img.save(temp_resized_file.name)
                    resized_image_paths.append(temp_resized_file.name)
                
                print(f"Resized image: {image_path}, new dimensions: {new_width}x{new_height}")
            else:
                # No resizing needed, use original path
                resized_image_paths.append(image_path)
                print(f"Image doesn't need resizing: {image_path}")
        
        return resized_image_paths
    except Exception as e:
        print(f"Error during image resizing: {e}")
        # Return original image paths in case of error
        return image_paths

def analyze_images_with_ai(image_paths, max_retries=2):
    """
    Call AI to analyze image content
    """
    user_prompt = """
    You are a professional sensitive content reviewer working in a government department. 
    Your task is to analyze the provided screenshots (from the same video, given separately) 
    and determine whether they contain sensitive content. 

    Since only a few screenshots are available, you must **reasonably guess the possible storyline or context of the video** based on the images.

    ## Sensitive Content Definition
    Sensitive content includes, but is not limited to:
    - Pornography or sexually explicit material
    - Violence
    - Terrorism or extremist propaganda
    - Gore or graphic injury
    - Promotion of illegal activities

    ## Workflow
    1. **Content Review**: Examine the screenshots carefully to determine if sensitive content is present.
    2. **Storyline Guessing**: Based on visual details, infer the likely storyline or scenario of the video, not just a literal description of the screenshots.
    3. **Result Generation**:
    - If no sensitive content is found: Output an approximate storyline in ~30 words.
    - If sensitive content is found: Specify the sensitivity type, and output an approximate storyline in ~30 words using strictly non-sensitive vocabulary.

    ## Output Format
    Return results in **valid JSON** with the following structure:
    {
        "Sensitive Content Present": "Yes" or "No",
        "Sensitive Type": "Pornography / Violence / Terrorism / Gore / Illegal Activities / None",
        "Video Storyline Guess": "Approx. 30-word storyline guess based on sensitivity status"
    }

    Begin the analysis now.
    """
    
    # Configure API key
    try:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    except KeyError:
        return '{"error": "GOOGLE_API_KEY environment variable not set"}'

    # Prepare images
    image_parts = []
    try:
        for image_path in image_paths:
            img = Image.open(image_path)
            image_parts.append(img)

    except Exception as e:
        return f'{{"error": "Failed to process image: {str(e)}"}}'

    # Initialize model and generate content
    model = genai.GenerativeModel(model_name="gemini-1.5-flash-8b-latest")

    # Properly construct content for the model
    content_for_model = [user_prompt]
    # Add images to content list
    content_for_model.extend(image_parts[:3])

    response = model.generate_content(content_for_model)

    # Return response
    try:
        if hasattr(response, 'candidates') and len(response.candidates) > 0:
            if response.text:
                return response.text
            else:
                return '{"error": "The model returned an empty response"}'
        else:
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                    return f'{{"error": "Content blocked due to policy restrictions: {str(response.prompt_feedback)}"}}'
            return '{"error": "No candidates in the response"}'
    except Exception as e:
        return f'{{"error": "Error processing response: {str(e)}"}}'

def process_video(video_url: str) -> list:
    """
    Process video and return list of extracted frame paths
    """
    print(f"Starting video processing: {video_url}")
    start_time = time.time()
    
    temp_video_path = None
    frame_paths = []
    cap = None
    
    try:
        # Check if URL ends with .m3u8, if so change to .mp4
        if video_url.endswith('.m3u8'):
            video_url = video_url[:-5] + '.mp4'
            print(f"URL suffix changed from .m3u8 to .mp4: {video_url}")
        
        # Download video file
        print("Starting video download...")
        video_start_time = time.time()
        response = requests.get(video_url, stream=True, timeout=60)
        
        # Check for request errors
        if response.status_code != 200:
            return []
        
        # Create temporary file to save video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_video_file.write(chunk)
            temp_video_path = temp_video_file.name
        
        video_end_time = time.time()
        print(f"Video download completed, time taken: {video_end_time - video_start_time:.2f} seconds")
        
        # Use OpenCV to read video and extract frames
        print("Starting video frame extraction...")
        frame_start_time = time.time()
        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            print("Failed to read video frames")
            return []
        
        print(f"Video info: FPS={fps:.2f}, total frames={total_frames}")
        
        # Randomly select 3 different frames
        frame_indices = []
        for _ in range(3):
            index = random.randint(0, total_frames - 1)
            while index in frame_indices:
                index = random.randint(0, total_frames - 1)
            frame_indices.append(index)
        
        # Create unique identifier for current run
        timestamp = int(time.time())
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Save as temporary file for later processing
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_frame_file:
                    cv2.imwrite(temp_frame_file.name, frame)
                    frame_paths.append(temp_frame_file.name)
                
                print(f"Frame {i+1} (frame #{frame_idx}) processed successfully")
        
        frame_end_time = time.time()
        print(f"Video frame extraction completed, time taken: {frame_end_time - frame_start_time:.2f} seconds")
        
        total_end_time = time.time()
        print(f"Total video processing time: {total_end_time - start_time:.2f} seconds")
        
        return frame_paths
        
    except requests.RequestException as e:
        print(f"Video download failed: {e}")
        return []
    except Exception as e:
        print(f"Error during video processing: {e}")
        return []
    finally:
        # Release video capture resource
        if cap is not None:
            cap.release()
            print("Video capture resource released")
            
        # Clean up temporary video file (note: do not clean up frame files as they will be used by main function)
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
                print("Temporary video file cleaned up")
            except Exception as e:
                print(f"Failed to clean up temporary video file: {e}")

def main(video_url: str) -> str:
    """
    Main function: Process video URL and return AI analysis result
    """
    # 1. Process video and extract frames
    frame_paths = process_video(video_url)
    
    if not frame_paths:
        return "Failed to process video, unable to extract frames"
    
    try:
        # 2. Resize images
        resized_frame_paths = resize_images(frame_paths)
        
        # 3. Call AI to analyze images
        print("Starting Google Gemini API call to analyze video frames...")
        api_start_time = time.time()
        
        ai_response = analyze_images_with_ai(resized_frame_paths)
        
        api_end_time = time.time()
        print(f"API call completed, time taken: {api_end_time - api_start_time:.2f} seconds")
        
        return ai_response
    finally:
        # Clean up temporary frame files
        for frame_path in frame_paths:
            try:
                if os.path.exists(frame_path):
                    os.unlink(frame_path)
            except Exception as e:
                print(f"Failed to clean up temporary frame file: {e}")
        
        # Clean up resized temporary frame files
        if 'resized_frame_paths' in locals():
            for frame_path in resized_frame_paths:
                try:
                    if os.path.exists(frame_path) and frame_path not in frame_paths:
                        os.unlink(frame_path)
                except Exception as e:
                    print(f"Failed to clean up temporary resized frame file: {e}")
        
        print(f"Temporary files cleaned up")

if __name__ == "__main__":
    # 示例视频URL（用户可以替换为实际的视频URL）
    sample_video_url = "https://storage.googleapis.com/avatarai/upload_video/2025-09-11/fb3505fc-3807-4087-8c70-32c925c92fde/6fae4390-bee4-4ac2-a908-914feabc14ef.m3u8"
    
    try:
        # 调用主函数处理视频
        result = main(sample_video_url)
        
        # 打印结果到控制台
        print("\n===== AI 分析结果 =====")
        print(result)
        print("=====================")
    except Exception as e:
        print(f"程序执行过程中发生错误: {e}")