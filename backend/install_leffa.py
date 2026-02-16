"""
Leffa Installation Script

This script downloads Leffa checkpoints from HuggingFace.
It will download several GB of model weights and dependencies.
"""

import os
import sys
from pathlib import Path

def check_leffa_directory():
    """Check if Leffa directory exists at project root."""
    project_root = Path(__file__).parent.parent
    leffa_path = project_root / "Leffa"
    
    if not leffa_path.exists():
        print("=" * 60)
        print("ERROR: Leffa directory not found!")
        print("=" * 60)
        print(f"Expected location: {leffa_path}")
        print()
        print("Please clone Leffa repository first:")
        print("  cd", project_root)
        print("  git clone https://github.com/franciszzj/Leffa")
        print()
        sys.exit(1)
    
    return leffa_path


def check_huggingface_hub():
    """Check if huggingface_hub is installed."""
    try:
        import huggingface_hub
        return True
    except ImportError:
        print("=" * 60)
        print("ERROR: huggingface_hub not installed!")
        print("=" * 60)
        print("Please install it first:")
        print("  pip install huggingface_hub")
        print()
        sys.exit(1)


def install_leffa(leffa_path):
    """Download Leffa checkpoints from HuggingFace."""
    from huggingface_hub import snapshot_download
    
    ckpts_dir = leffa_path / "ckpts"
    
    print()
    print("=" * 60)
    print("DOWNLOADING LEFFA CHECKPOINTS FROM HUGGINGFACE")
    print("=" * 60)
    print()
    print("Repository: franciszzj/Leffa")
    print()
    print("This will download:")
    print("- Model weights (virtual_tryon.pth, etc.)")
    print("- Stable Diffusion inpainting models")
    print("- 3rdparty dependencies (SCHP, densepose, detectron2)")
    print("- Preprocessor models (humanparsing, openpose)")
    print("- Example images")
    print()
    print("Size: Several GB")
    print("Time: 10-30 minutes depending on connection")
    print("=" * 60)
    print()
    
    # Check if HF_TOKEN is set
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("Warning: You are sending unauthenticated requests to the HF Hub.")
        print("Please set a HF_TOKEN to enable higher rate limits and faster downloads.")
        print()
    
    try:
        snapshot_download(
            repo_id="franciszzj/Leffa",
            local_dir=str(ckpts_dir),
            local_dir_use_symlinks=False,
            token=hf_token,
            resume_download=True,  # Resume if interrupted
        )
        
        print()
        print("=" * 60)
        print("✓ LEFFA CHECKPOINTS DOWNLOADED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print(f"Location: {ckpts_dir}")
        print()
        print("You can now use Leffa for virtual try-on!")
        print()
        
    except KeyboardInterrupt:
        print()
        print()
        print("=" * 60)
        print("DOWNLOAD INTERRUPTED")
        print("=" * 60)
        print()
        print("The download was interrupted. You can resume by running this script again.")
        print("Already downloaded files will be skipped.")
        print()
        sys.exit(1)
    except Exception as e:
        print()
        print()
        print("=" * 60)
        print("ERROR DURING DOWNLOAD")
        print("=" * 60)
        print()
        print(f"Error: {e}")
        print()
        print("You can try again by running this script again.")
        print("Already downloaded files will be skipped.")
        print()
        sys.exit(1)


def main():
    print("=" * 60)
    print("LEFFA INSTALLATION")
    print("=" * 60)
    
    # Check Leffa directory
    leffa_path = check_leffa_directory()
    print(f"✓ Leffa found at: {leffa_path}")
    
    # Check huggingface_hub
    check_huggingface_hub()
    print("✓ huggingface_hub available")
    
    # Check checkpoint directory
    ckpts_dir = leffa_path / "ckpts"
    print(f"Checkpoint directory: {ckpts_dir}")
    
    # Download checkpoints
    install_leffa(leffa_path)


if __name__ == "__main__":
    main()
