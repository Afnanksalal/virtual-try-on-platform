"""
Script to download IDM-VTON model weights from HuggingFace Hub.

This script downloads all required model components for IDM-VTON:
1. IDM-VTON checkpoint
2. IP-Adapter for SDXL
3. Image Encoder (CLIP ViT-H)
4. Human Parsing model (SCHP)

Usage:
    python download_weights.py [--skip-base-models]

Requirements:
    pip install huggingface_hub
"""

import os
import sys
import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError:
    print("Error: huggingface_hub not installed")
    print("Install with: pip install huggingface_hub")
    sys.exit(1)


def download_idm_vton_checkpoint(base_dir: Path):
    """Download IDM-VTON trained checkpoint."""
    print("\n" + "="*60)
    print("Downloading IDM-VTON Checkpoint...")
    print("="*60)
    
    checkpoint_dir = base_dir / "checkpoint"
    
    try:
        snapshot_download(
            repo_id="yisol/IDM-VTON",
            local_dir=str(checkpoint_dir),
            ignore_patterns=["*.md", "*.txt", "*.jpg", "*.png"],
        )
        print(f"✓ IDM-VTON checkpoint downloaded to: {checkpoint_dir}")
    except Exception as e:
        print(f"✗ Failed to download IDM-VTON checkpoint: {e}")
        print("  Trying alternative source...")
        try:
            snapshot_download(
                repo_id="imaginairy/idm-vton-safetensors",
                local_dir=str(checkpoint_dir),
                ignore_patterns=["*.md", "*.txt", "*.jpg", "*.png"],
            )
            print(f"✓ IDM-VTON checkpoint downloaded from alternative source")
        except Exception as e2:
            print(f"✗ Failed to download from alternative source: {e2}")
            return False
    
    return True


def download_ip_adapter(base_dir: Path):
    """Download IP-Adapter weights for SDXL."""
    print("\n" + "="*60)
    print("Downloading IP-Adapter for SDXL...")
    print("="*60)
    
    ip_adapter_dir = base_dir / "ip_adapter"
    ip_adapter_dir.mkdir(exist_ok=True)
    
    try:
        # Download IP-Adapter weights
        hf_hub_download(
            repo_id="h94/IP-Adapter",
            filename="sdxl_models/ip-adapter-plus_sdxl_vit-h.bin",
            local_dir=str(ip_adapter_dir),
        )
        print(f"✓ IP-Adapter weights downloaded to: {ip_adapter_dir}")
    except Exception as e:
        print(f"✗ Failed to download IP-Adapter: {e}")
        return False
    
    return True


def download_image_encoder(base_dir: Path):
    """Download CLIP image encoder."""
    print("\n" + "="*60)
    print("Downloading Image Encoder (CLIP ViT-H)...")
    print("="*60)
    
    encoder_dir = base_dir / "image_encoder"
    
    try:
        snapshot_download(
            repo_id="h94/IP-Adapter",
            allow_patterns=["models/image_encoder/*"],
            local_dir=str(base_dir),
        )
        
        # Move from models/image_encoder to image_encoder
        source = base_dir / "models" / "image_encoder"
        if source.exists():
            import shutil
            if encoder_dir.exists():
                shutil.rmtree(encoder_dir)
            shutil.move(str(source), str(encoder_dir))
            # Clean up empty models directory
            models_dir = base_dir / "models"
            if models_dir.exists() and not any(models_dir.iterdir()):
                models_dir.rmdir()
        
        print(f"✓ Image encoder downloaded to: {encoder_dir}")
    except Exception as e:
        print(f"✗ Failed to download image encoder: {e}")
        return False
    
    return True


def download_human_parsing(base_dir: Path):
    """Download human parsing model (SCHP)."""
    print("\n" + "="*60)
    print("Downloading Human Parsing Model (SCHP)...")
    print("="*60)
    
    schp_dir = base_dir / "schp"
    schp_dir.mkdir(exist_ok=True)
    
    try:
        # Download SCHP checkpoint
        hf_hub_download(
            repo_id="mattmdjaga/segformer_b2_clothes",
            filename="pytorch_model.bin",
            local_dir=str(schp_dir),
        )
        print(f"✓ Human parsing model downloaded to: {schp_dir}")
        print("  Note: Using Segformer as alternative to SCHP")
    except Exception as e:
        print(f"✗ Failed to download human parsing model: {e}")
        print("  This is optional - segmentation can use alternative models")
        return True  # Non-critical failure
    
    return True


def verify_downloads(base_dir: Path):
    """Verify all required files are downloaded."""
    print("\n" + "="*60)
    print("Verifying Downloads...")
    print("="*60)
    
    required_paths = [
        base_dir / "checkpoint",
        base_dir / "ip_adapter",
        base_dir / "image_encoder",
    ]
    
    all_present = True
    for path in required_paths:
        if path.exists() and any(path.iterdir()):
            print(f"✓ {path.name}: Present")
        else:
            print(f"✗ {path.name}: Missing or empty")
            all_present = False
    
    return all_present


def main():
    parser = argparse.ArgumentParser(
        description="Download IDM-VTON model weights"
    )
    parser.add_argument(
        "--skip-base-models",
        action="store_true",
        help="Skip downloading base models (SDXL will be auto-downloaded on first use)"
    )
    args = parser.parse_args()
    
    # Get base directory
    script_dir = Path(__file__).parent
    base_dir = script_dir
    
    print("="*60)
    print("IDM-VTON Model Weights Downloader")
    print("="*60)
    print(f"Download directory: {base_dir}")
    print(f"Skip base models: {args.skip_base_models}")
    
    # Create base directory if it doesn't exist
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Download components
    success = True
    
    # 1. IDM-VTON checkpoint
    if not download_idm_vton_checkpoint(base_dir):
        success = False
    
    # 2. IP-Adapter
    if not download_ip_adapter(base_dir):
        success = False
    
    # 3. Image Encoder
    if not download_image_encoder(base_dir):
        success = False
    
    # 4. Human Parsing (optional)
    download_human_parsing(base_dir)
    
    # Verify downloads
    print()
    if verify_downloads(base_dir):
        print("\n" + "="*60)
        print("✓ All required models downloaded successfully!")
        print("="*60)
        print("\nNote: SDXL base model will be downloaded automatically")
        print("on first use and cached in ~/.cache/huggingface/hub/")
        print("\nYou can now run the IDM-VTON pipeline.")
    else:
        print("\n" + "="*60)
        print("✗ Some downloads failed. Please check errors above.")
        print("="*60)
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
