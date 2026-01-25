from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image, ImageOps
import io
import numpy as np
import cv2
import uuid
from datetime import datetime
from ..core.logging_config import get_logger
from ..core.file_validator import FileValidator
from ..services.supabase_storage import supabase_storage

router = APIRouter()
logger = get_logger("api.image_composition")

# Initialize file validator
file_validator = FileValidator(max_size_mb=10)

@router.post("/combine-head-body")
async def combine_head_body(
    head_image: UploadFile = File(...),
    body_image: UploadFile = File(...),
):
    """
    Combine a head-only photo with a generated body image.
    ALL files stored in Supabase ONLY.
    """
    try:
        # Validate uploaded files
        await file_validator.validate_file(head_image)
        await file_validator.validate_file(body_image)
        
        # Read images
        head_bytes = await head_image.read()
        body_bytes = await body_image.read()
        
        head_img = Image.open(io.BytesIO(head_bytes)).convert("RGB")
        body_img = Image.open(io.BytesIO(body_bytes)).convert("RGB")
        
        # Convert to numpy for OpenCV processing
        head_np = np.array(head_img)
        body_np = np.array(body_img)
        
        # Convert RGB to BGR for OpenCV
        head_bgr = cv2.cvtColor(head_np, cv2.COLOR_RGB2BGR)
        
        # Detect face in head image
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(head_bgr, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) == 0:
            raise HTTPException(400, "No face detected in head image")
        
        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Extract head region with some padding
        padding = int(h * 0.3)  # 30% padding
        y1 = max(0, y - padding)
        y2 = min(head_img.height, y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(head_img.width, x + w + padding)
        
        head_region = head_img.crop((x1, y1, x2, y2))
        
        # Resize body image to standard size
        body_img = body_img.resize((512, 768))  # Standard portrait size
        
        # Calculate head position on body (top 1/3)
        head_height = int(body_img.height * 0.25)
        head_width = int(head_height * head_region.width / head_region.height)
        
        # Resize head
        head_resized = head_region.resize((head_width, head_height))
        
        # Calculate position (centered horizontally, top portion)
        x_pos = (body_img.width - head_width) // 2
        y_pos = int(body_img.height * 0.05)  # 5% from top
        
        # Create alpha mask for smooth blending
        mask = Image.new('L', head_resized.size, 255)
        
        # Paste head onto body
        result = body_img.copy()
        result.paste(head_resized, (x_pos, y_pos), mask)
        
        # Upload to Supabase
        request_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"composed/{request_id}/combined_{timestamp}.png"
        
        public_url = supabase_storage.upload_image(
            result,
            bucket=supabase_storage.RESULTS_BUCKET,
            path=path
        )
        
        logger.info(f"Successfully combined head and body, uploaded to Supabase: {public_url}")
        
        return {
            "message": "Images combined successfully",
            "request_id": request_id,
            "image_url": public_url
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image composition failed: {e}", exc_info=True)
        raise HTTPException(500, f"Image composition failed: {str(e)}")
