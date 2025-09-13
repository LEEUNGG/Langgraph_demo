import os
import requests
import io
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path)

def generate_description_from_url(image_url: str) -> str:
    """
    Generates a text description for an image from a public URL.

    Args:
        image_url (str): A public URL pointing to an image.
    
    Returns:
        str: JSON string containing the image analysis result or error message
    """

    user_prompt = """
    You are a professional sensitive content identifier that working in a goverment department, tasked with analyzing a given image and determining whether it contains sensitive content.

## Input:
- Image: (provided separately)

## Definition of Sensitive Content:
Sensitive content includes but is not limited to pornography, violence, terror, gore, and content promoting illegal activities.

## Processing Flow:
1. **Content Identification**: Carefully examine the image to determine if it contains sensitive content.
2. **Result Output**:
    - If the image contains no sensitive content, output a description of the image, approximately 30 words in length.
    - If the image contains sensitive content, output the sensitive category and describe the image using non-sensitive vocabulary, approximately 30 words.

## Output Format:
Please output your judgment results in the following JSON format:
{
    "Sensitive Content Present": "Yes" or "No",
    "Sensitive Type": "If present, specify type; if absent, enter 'None'",
    "Image Description": "Output corresponding description based on sensitivity status"
}

Begin analysis based on the provided images.
    """

    # --- 1. Configure API Key ---
    try:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    except KeyError:
        return '{"error": "GOOGLE_API_KEY environment variable not set"}'

    # --- 2. Download and Prepare Image ---
    image_parts = []
    try:
        response = requests.get(image_url)
        response.raise_for_status() 
        img = Image.open(io.BytesIO(response.content))
        
        MAX_WIDTH = 512
        MAX_HEIGHT = 512
        
        width, height = img.size
        
        if width > MAX_WIDTH or height > MAX_HEIGHT:
            width_ratio = MAX_WIDTH / width
            height_ratio = MAX_HEIGHT / height
            resize_ratio = min(width_ratio, height_ratio)
            
            new_width = int(width * resize_ratio)
            new_height = int(height * resize_ratio)
            
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        image_parts.append(img)

    except requests.exceptions.RequestException as e:
        return f'{"error": "Failed to download or process image: {str(e)}"}'

    # --- 3. Initialize the Model ---
    model = genai.GenerativeModel(model_name="gemini-1.5-flash-8b-latest")

    # --- 4. Generate Content ---
    content_for_model = [
        user_prompt,
        image_parts[0] 
    ]
    
    response = model.generate_content(content_for_model)

    # --- 5. Return the Response ---
    try:
        if hasattr(response, 'candidates') and len(response.candidates) > 0:
            if response.text:
                return response.text
            else:
                return '{"error": "The model returned an empty response"}'
        else:
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                    return f'{"error": "Content blocked due to policy restrictions: {str(response.prompt_feedback)}"}'
            return '{"error": "No candidates in the response"}'
    except Exception as e:
        return f'{"error": "Error processing response: {str(e)}"}'
