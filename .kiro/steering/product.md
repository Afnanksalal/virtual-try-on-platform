# Product Overview

Virtual Try-On Platform - An AI-powered fashion technology application that enables users to virtually try on clothing using advanced machine learning models.

## Core Features

### 2D Virtual Try-On (Leffa)
- Upload user photos and garment images to generate realistic try-on results
- Supports upper body, lower body, and dresses
- Advanced options: adjustable inference steps, guidance scale, seed control
- Model variants: viton_hd (recommended) and dress_code
- Performance modes: ref_acceleration for speed, repaint for better edges
- Batch processing: try multiple garments at once

### Body Generation (SDXL + InstantID)
- Create synthetic body models based on user parameters
- Parameters: ethnicity, body type, height (140-220cm), weight (40-200kg)
- Generate multiple variations (up to 4 at once)
- **Identity-Preserving Generation (InstantID)**: Generates full-body images with user's actual face natively embedded (not stitched)
  - Uses InsightFace for face embedding extraction
  - ControlNet for facial keypoint guidance
  - IP-Adapter for identity injection
  - Gemini Vision for accurate skin tone and ethnicity detection
  - Graceful fallback to SDXL if InstantID unavailable

### AI Recommendations (Gemini + eBay)
- Personalized outfit recommendations using Gemini 2.5 Flash Vision
- Analyzes user photo, wardrobe, and try-on history
- Scientific color theory based on skin tone analysis
- eBay product search with buy links (up to 20 items)
- Considers user profile: height, weight, body type, style preferences

### Smart Onboarding
- Analyze uploaded photos to determine if head-only or full-body
- Automatic body generation for head-only photos
- Seamless head-body composition for complete avatars

### 3D Reconstruction
- Generate 3D meshes from 2D images
- TripoSR for fast, high-quality reconstruction
- SAM 2.1 for precise segmentation
- Depth Anything V2 for depth estimation
- Export formats: GLB, OBJ with textures

### Wardrobe Management
- Upload and store garment images
- Organize personal clothing collection
- Quick access for try-on sessions
- Database-backed with Supabase storage

### Try-On History
- Track all previous try-on results
- Review and download past results
- Use history for AI recommendations
- User data isolation (users only see their own data)

## User Flow

1. **Authentication**: Sign up/login via Supabase
2. **Onboarding**: Upload photo → system analyzes (head-only vs full-body)
3. **Body Generation** (if needed): Generate body models → combine with face
4. **Studio**: Try on garments from wardrobe or upload new ones
5. **Recommendations**: Get AI-powered outfit suggestions
6. **History**: Review and download previous try-ons
7. **3D Visualization**: Generate 3D models for advanced viewing

## Security & Privacy

- JWT-based authentication via Supabase
- User data isolation enforced at API level
- Users can only access their own wardrobe and history
- Secure file uploads with validation (max 10MB, image types only)
- CORS protection (configurable origins)

## Technical Highlights

- **GPU Optimized**: Runs on 4GB VRAM (tested on RTX 3050)
- **Memory Management**: Aggressive GPU memory cleanup between operations
- **Error Handling**: Comprehensive error classification and recovery
- **Performance Tracking**: Automatic metrics logging
- **Background Tasks**: Temp file cleanup every 15 minutes
- **Model Caching**: Singleton pattern for efficient model loading
