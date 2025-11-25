"""
Utility Functions for Image Processing
Handles preprocessing, augmentation, and validation
"""

import numpy as np
from PIL import Image
import io


def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for model input
    
    Args:
        image (PIL.Image): Input image
        target_size (tuple): Target dimensions (height, width)
    
    Returns:
        np.ndarray: Preprocessed image array ready for prediction
    
    Processing Steps:
        1. Convert to RGB format
        2. Resize to target dimensions
        3. Normalize pixel values to [0, 1]
        4. Add batch dimension
    
    Example:
        >>> from PIL import Image
        >>> img = Image.open('accident.jpg')
        >>> processed = preprocess_image(img)
        >>> print(processed.shape)  # (1, 224, 224, 3)
    """
    
    try:
        # Ensure RGB format (remove alpha channel if present)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Normalize pixel values to [0, 1]
        img_array = img_array / 255.0
        
        # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")


def validate_image(image):
    """
    Validate uploaded image meets requirements
    
    Args:
        image (PIL.Image): Image to validate
    
    Returns:
        tuple: (is_valid, error_message)
    
    Validation Checks:
        - Minimum resolution: 100x100 pixels
        - Maximum file size: 10MB
        - Valid formats: RGB, RGBA
    """
    
    # Check image dimensions
    width, height = image.size
    if width < 100 or height < 100:
        return False, "❌ Image too small. Minimum size: 100x100 pixels"
    
    # Check image format
    if image.mode not in ['RGB', 'RGBA', 'L']:
        return False, f"❌ Unsupported color mode: {image.mode}"
    
    # Check file size (if available)
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        size_mb = len(img_byte_arr.getvalue()) / (1024 * 1024)
        
        if size_mb > 10:
            return False, f"❌ File too large: {size_mb:.2f}MB (max: 10MB)"
    except:
        pass  # Skip size check if fails
    
    return True, "✅ Image validated successfully"


def get_image_metadata(image):
    """
    Extract metadata from uploaded image
    
    Args:
        image (PIL.Image): Input image
    
    Returns:
        dict: Image metadata
    """
    
    width, height = image.size
    
    metadata = {
        "width": width,
        "height": height,
        "format": image.format or "Unknown",
        "mode": image.mode,
        "aspect_ratio": round(width / height, 2),
        "megapixels": round((width * height) / 1_000_000, 2)
    }
    
    return metadata


def enhance_image(image, brightness=1.0, contrast=1.0, sharpness=1.0):
    """
    Apply image enhancements for better model performance
    
    Args:
        image (PIL.Image): Input image
        brightness (float): Brightness factor (0.5 = darker, 2.0 = brighter)
        contrast (float): Contrast factor
        sharpness (float): Sharpness factor
    
    Returns:
        PIL.Image: Enhanced image
    """
    
    from PIL import ImageEnhance
    
    # Apply brightness adjustment
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
    
    # Apply contrast adjustment
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
    
    # Apply sharpness adjustment
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness)
    
    return image


def create_thumbnail(image, max_size=(300, 300)):
    """
    Create thumbnail version of image for display
    
    Args:
        image (PIL.Image): Input image
        max_size (tuple): Maximum dimensions
    
    Returns:
        PIL.Image: Thumbnail image
    """
    
    image_copy = image.copy()
    image_copy.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image_copy


def calculate_image_stats(image):
    """
    Calculate statistical properties of image
    
    Args:
        image (PIL.Image): Input image
    
    Returns:
        dict: Statistical metrics
    """
    
    img_array = np.array(image)
    
    stats = {
        "mean_brightness": float(np.mean(img_array)),
        "std_brightness": float(np.std(img_array)),
        "min_value": int(np.min(img_array)),
        "max_value": int(np.max(img_array)),
        "median": float(np.median(img_array))
    }
    
    return stats
