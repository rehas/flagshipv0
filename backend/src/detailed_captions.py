import os
import torch
from PIL import Image
import traceback
from transformers import AutoModelForCausalLM, AutoProcessor

# Global model and processor instances (loaded only once)
_model = None
_processor = None
_device = None

def _initialize_model():
    """Initialize the Florence-2 model and processor (if not already loaded)."""
    global _model, _processor, _device
    
    # Only initialize if not already done
    if _model is None or _processor is None:
        print("Initializing MiaoshouAI/Florence-2-large-PromptGen-v2.0 model...")
        
        # Determine device (use GPU if available)
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {_device}")
        
        try:
            # Load model and processor
            _model = AutoModelForCausalLM.from_pretrained(
                "MiaoshouAI/Florence-2-large-PromptGen-v2.0", 
                trust_remote_code=True
            )
            _processor = AutoProcessor.from_pretrained(
                "MiaoshouAI/Florence-2-large-PromptGen-v2.0", 
                trust_remote_code=True
            )
            
            # Move model to the appropriate device
            _model.to(_device)
            print("Florence-2 model initialized successfully.")
            
            return True
        except Exception as e:
            print(f"Error initializing Florence-2 model: {str(e)}")
            print(traceback.format_exc())
            return False
    
    # Model already initialized
    return True

def generate_detailed_caption(image_path, instruction="<MIXED_CAPTION_PLUS>", max_new_tokens=1024):
    """
    Generate a detailed caption for an image using the Florence-2 model.
    
    Args:
        image_path (str): Path to the image file
        instruction (str): One of the supported instruction prompts:
                          "<GENERATE_TAGS>" - generate danbooru style tags
                          "<CAPTION>" - one line caption
                          "<DETAILED_CAPTION>" - structured caption with positions
                          "<MIXED_CAPTION_PLUS>" - very detailed description
                          "<MIXED_CAPTION>" - mixed caption style with tags
        max_new_tokens (int): Maximum number of tokens to generate
        
    Returns:
        dict: Result containing the generated caption or error information
    """
    # Ensure model is initialized
    if not _initialize_model():
        return {"error": "Failed to initialize Florence-2 model"}
    
    try:
        # Open and process the image
        image = Image.open(image_path).convert('RGB')
        
        # Create inputs for the model
        inputs = _processor(text=instruction, images=image, return_tensors="pt").to(_device)
        
        # Generate caption
        print(f"Generating {instruction} caption...")
        with torch.no_grad():  # Disable gradient calculation for inference
            generated_ids = _model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=3
            )
        
        # Decode the generated text
        generated_text = _processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Post-process the generated text
        parsed_answer = _processor.post_process_generation(
            generated_text, 
            task=instruction, 
            image_size=(image.width, image.height)
        )
        
        # print(f"Caption generated successfully: {parsed_answer[:100]}...")
        return parsed_answer["<MIXED_CAPTION_PLUS>"] if "<MIXED_CAPTION_PLUS>" in parsed_answer else parsed_answer
        
    except Exception as e:
        error_message = f"Error generating caption: {str(e)}"
        print(error_message)
        print(traceback.format_exc())
        return {"error": error_message}

# Example usage
if __name__ == "__main__":
    # Test the module if run directly
    test_image = "path/to/test/image.jpg"
    if os.path.exists(test_image):
        result = generate_detailed_caption(test_image, "<MIXED_CAPTION_PLUS>")
        print(result)
    else:
        print(f"Test image not found: {test_image}") 