import os
import io
import json
import requests
import google.generativeai as genai
from google.oauth2 import service_account
from pydub import AudioSegment

def shrink_audio(audio_bytes: bytes, max_size: int = 10 * 1024 * 1024) -> bytes:
    """
    Shrinks audio to below max_size by lowering bitrate using pydub.
    
    Args:
        audio_bytes (bytes): Original audio data in bytes.
        max_size (int): Maximum allowed size in bytes.
    
    Returns:
        bytes: Compressed audio data not exceeding max_size.
    """
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    bitrate_kbps = 128  # start with 128 kbps

    while bitrate_kbps >= 32:  # don't go below 32 kbps
        out_io = io.BytesIO()
        audio.export(out_io, format="mp3", bitrate=f"{bitrate_kbps}k")
        compressed = out_io.getvalue()
        if len(compressed) <= max_size:
            return compressed
        bitrate_kbps //= 2  # reduce bitrate and try again

    # fallback: return smallest attempt
    return compressed


def generate_description_from_url(audio_url: str) -> str:
    """
    Generates a text description for an MP3 audio file from a public URL.
    If the file exceeds 10MB, it is compressed to fit under 10MB.
    """

    user_prompt = """
    You are a professional sensitive content identifier working in a government department, tasked with analyzing a given audio file and determining whether it contains sensitive content.

    ## Input:
    - Audio: (provided separately)

    ## Definition of Sensitive Content:
    Sensitive content includes but is not limited to explicit sexual content, hate speech, violence promotion, terror-related content, illegal activity promotion, and harmful content that could cause psychological distress.

    ## Processing Flow:
    1. **Content Identification**: Carefully analyze the audio to determine if it contains sensitive content.
    2. **Result Output**:
        - If the audio contains no sensitive content, output a description of the audio content, approximately 10 words in length.
        - If the audio contains sensitive content, output the sensitive category and describe the audio using non-sensitive vocabulary, approximately 10 words.

    ## Output Format:
    {
        "Sensitive Content Present": "Yes" or "No",
        "Sensitive Type": "If present, specify type; if absent, enter 'None'",
        "Audio Description": "Output corresponding description based on sensitivity status"
    }
    """

    # Configure authentication using service account
    try:
        service_account_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pulsebase-64987db2e79f.json')
        credentials = service_account.Credentials.from_service_account_file(
            service_account_path,
            scopes=[
                'https://www.googleapis.com/auth/cloud-platform',
                'https://www.googleapis.com/auth/generative-language'
            ]
        )

        genai.configure(credentials=credentials)
        print(f"Configured with scopes: {credentials.scopes}")
        
    except FileNotFoundError:
        return '{"error": "Service account file not found"}'
    except Exception as e:
        return f'{{"error": "Failed to configure authentication: {str(e)}"}}'

    # Download MP3 file
    try:
        response = requests.get(audio_url)
        response.raise_for_status()
        audio_data = response.content

        # Shrink if > 10MB
        MAX_AUDIO_SIZE = 10 * 1024 * 1024
        if len(audio_data) > MAX_AUDIO_SIZE:
            audio_data = shrink_audio(audio_data, MAX_AUDIO_SIZE)

        # Create MP3 audio part for the model
        audio_part = {
            "mime_type": "audio/mpeg",
            "data": audio_data
        }

    except requests.exceptions.RequestException as e:
        return f'{{"error": "Failed to download audio file: {str(e)}"}}'

    # Initialize the model
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    except Exception as e:
        return f'{{"error": "Failed to initialize model: {str(e)}"}}'

    # Generate content analysis
    try:
        content_for_model = [user_prompt, audio_part]
        response = model.generate_content(content_for_model)

    except Exception as e:
        return f'{{"error": "Failed to generate content: {str(e)}"}}'

    # Process and return the response
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


if __name__ == "__main__":
    audio_url = "https://storage.googleapis.com/avatarai/upload_file/2025-08-05/cafcff84-2706-4cb1-a00d-f2f6e3073ee6_.mp3"
    description = generate_description_from_url(audio_url)
    print(description)
