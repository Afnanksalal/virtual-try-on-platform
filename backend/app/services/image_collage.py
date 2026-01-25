from PIL import Image, ImageDraw, ImageFont
import io
import base64
from typing import List, Optional
import numpy as np

class ImageCollageService:
    """Creates collages from multiple images for AI analysis."""
    
    @staticmethod
    def create_collage(
        images: List[Image.Image], 
        max_width: int = 1200,
        labels: Optional[List[str]] = None,
        add_borders: bool = True,
        border_width: int = 3,
        border_color: tuple = (200, 200, 200)
    ) -> Image.Image:
        """
        Create a grid collage from multiple images with labels and borders.
        Optimized for Gemini Vision API analysis.
        
        Args:
            images: List of PIL Images to combine
            max_width: Maximum width of the collage
            labels: Optional labels for each image (e.g., ["User Photo", "Wardrobe Item 1"])
            add_borders: Whether to add borders around each image
            border_width: Width of borders in pixels
            border_color: RGB tuple for border color
            
        Returns:
            PIL Image containing the collage
        """
        if not images:
            raise ValueError("No images provided")
        
        # Calculate grid dimensions
        num_images = len(images)
        if num_images == 1:
            cols, rows = 1, 1
        elif num_images == 2:
            cols, rows = 2, 1
        elif num_images <= 4:
            cols, rows = 2, 2
        elif num_images <= 6:
            cols, rows = 3, 2
        else:
            cols, rows = 3, 3
        
        # Calculate target size accounting for borders and labels
        label_height = 30 if labels else 0
        border_padding = border_width * 2 if add_borders else 0
        target_size = (max_width // cols) - border_padding
        
        # Resize all images to same size
        resized_images = []
        
        for idx, img in enumerate(images[:cols * rows]):  # Limit to grid size
            # Maintain aspect ratio
            img_copy = img.copy()
            img_copy.thumbnail((target_size, target_size - label_height), Image.Resampling.LANCZOS)
            
            # Create centered image on white background
            cell_height = target_size
            background = Image.new('RGB', (target_size, cell_height), 'white')
            
            # Center the image
            img_offset = (
                (target_size - img_copy.width) // 2,
                (cell_height - label_height - img_copy.height) // 2
            )
            background.paste(img_copy, img_offset)
            
            # Add label if provided
            if labels and idx < len(labels):
                draw = ImageDraw.Draw(background)
                label_text = labels[idx]
                
                # Try to use a nice font, fall back to default
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                # Calculate text position (centered at bottom)
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_x = (target_size - text_width) // 2
                text_y = cell_height - label_height + 5
                
                # Draw text with background
                draw.rectangle(
                    [(0, cell_height - label_height), (target_size, cell_height)],
                    fill=(240, 240, 240)
                )
                draw.text((text_x, text_y), label_text, fill=(50, 50, 50), font=font)
            
            # Add border if requested
            if add_borders:
                bordered = Image.new('RGB', 
                    (target_size + border_padding, cell_height + border_padding), 
                    border_color
                )
                bordered.paste(background, (border_width, border_width))
                resized_images.append(bordered)
            else:
                resized_images.append(background)
        
        # Calculate final collage dimensions
        cell_width = target_size + border_padding
        cell_height = target_size + border_padding
        collage_width = cols * cell_width
        collage_height = rows * cell_height
        
        # Create collage
        collage = Image.new('RGB', (collage_width, collage_height), 'white')
        
        for idx, img in enumerate(resized_images):
            row = idx // cols
            col = idx % cols
            x = col * cell_width
            y = row * cell_height
            collage.paste(img, (x, y))
        
        return collage
    
    @staticmethod
    def image_to_base64(image: Image.Image, format: str = 'PNG') -> str:
        """Convert PIL Image to base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/{format.lower()};base64,{img_str}"
    
    @staticmethod
    def base64_to_image(base64_str: str) -> Image.Image:
        """Convert base64 string to PIL Image."""
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        img_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(img_data))


collage_service = ImageCollageService()
