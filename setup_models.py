import os
import sys
import zipfile
import urllib.request
import shutil
import subprocess

# URLs for models and repository
WAV2LIP_REPO_URL = "https://github.com/Rudrabha/Wav2Lip/archive/refs/heads/master.zip"
WAV2LIP_GAN_URL = "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth"
S3FD_URL = "https://huggingface.co/rippertnt/wav2lip/resolve/main/s3fd.pth"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WAV2LIP_DIR = os.path.join(BASE_DIR, "wav2lip")

def download_file(url, dest_path):
    """Downloads a file with a visual progress bar."""
    print(f"Downloading {os.path.basename(dest_path)} from {url}...")
    
    # Custom progress reporter
    def reporthook(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = min(100, readsofar * 1e2 / totalsize)
            s = f"\r   [{percent:5.1f}%] {readsofar / (1024*1024):.2f} MB / {totalsize / (1024*1024):.2f} MB"
            sys.stdout.write(s)
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\r   Read {readsofar / (1024*1024):.2f} MB")
            sys.stdout.flush()

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    # Download
    urllib.request.urlretrieve(url, dest_path, reporthook)
    print("\n   Download complete!")

def setup_wav2lip_codebase():
    """Downloads the Wav2Lip repository zip, extracts it, and renames it to 'wav2lip'."""
    if os.path.exists(WAV2LIP_DIR):
        print("Wav2Lip directory already exists. Skipping codebase download.")
        return
    
    zip_path = os.path.join(BASE_DIR, "wav2lip_master.zip")
    
    try:
        download_file(WAV2LIP_REPO_URL, zip_path)
        
        print("Extracting Wav2Lip codebase...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(BASE_DIR)
            
        extracted_dir = os.path.join(BASE_DIR, "Wav2Lip-master")
        if os.path.exists(extracted_dir):
            os.rename(extracted_dir, WAV2LIP_DIR)
            print("Successfully set up 'wav2lip' directory.")
        else:
            raise FileNotFoundError("Could not find extracted folder 'Wav2Lip-master'.")
            
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)

def download_checkpoints():
    """Downloads wav2lip_gan.pth and s3fd.pth to their respective locations."""
    # wav2lip_gan.pth goes to wav2lip/checkpoints/wav2lip_gan.pth
    gan_path = os.path.join(WAV2LIP_DIR, "checkpoints", "wav2lip_gan.pth")
    if not os.path.exists(gan_path):
        download_file(WAV2LIP_GAN_URL, gan_path)
    else:
        print("wav2lip_gan.pth already exists. Skipping.")

    # s3fd.pth goes to wav2lip/face_detection/detection/sfd/s3fd.pth
    s3fd_path = os.path.join(WAV2LIP_DIR, "face_detection", "detection", "sfd", "s3fd.pth")
    if not os.path.exists(s3fd_path):
        download_file(S3FD_URL, s3fd_path)
    else:
        print("s3fd.pth already exists. Skipping.")

def patch_wav2lip_compatibility():
    """Patches Wav2Lip codebase to resolve Python 3.11 and modern library compatibility issues."""
    print("Patching Wav2Lip codebase for Python 3.11 / modern-library compatibility...")

    patched_count = 0
    for root, dirs, files in os.walk(WAV2LIP_DIR):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)

                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                original_content = content

                # np.int was removed in modern numpy - replace with the builtin int
                if "np.int" in content:
                    content = content.replace("np.int", "int")

                # librosa >=0.10 made librosa.filters.mel() keyword-only past the
                # first couple of positional args. Wav2Lip's audio.py still calls
                # it the old positional way, which raises:
                #   TypeError: mel() takes 0 positional arguments but 2 were given
                if "librosa.filters.mel(hp.sample_rate, hp.n_fft" in content:
                    content = content.replace(
                        "librosa.filters.mel(hp.sample_rate, hp.n_fft",
                        "librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft"
                    )

                if content != original_content:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"   Patched: {os.path.relpath(file_path, WAV2LIP_DIR)}")
                    patched_count += 1

    print(f"Patched {patched_count} files successfully.")

def install_dependencies():
    """Installs required python dependencies."""
    print("Installing python dependencies (librosa, filterpy, opencv-python, TTS)...")
    
    # We will run pip install
    # Wait, let's also check if they want PyTorch CUDA.
    # We will first install the basic libraries
    dependencies = ["librosa", "filterpy", "opencv-python", "TTS"]
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
        
    print("Dependencies installed successfully!")

def verify_pytorch_gpu():
    """Checks if PyTorch is configured with CUDA GPU support."""
    print("\n--- PyTorch GPU Verification ---")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        if cuda_available:
            print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
            print("CUDA is successfully configured! Running XTTS and Wav2Lip will be very fast.")
        else:
            print("WARNING: CUDA is NOT available to PyTorch. Models will run on CPU, which will be slow.")
            print("To enable GPU acceleration, please install PyTorch with CUDA using:")
            print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall")
    except ImportError:
        print("PyTorch is not installed.")

if __name__ == "__main__":
    print("Starting Wav2Lip and XTTS Model Setup...")
    try:
        setup_wav2lip_codebase()
        download_checkpoints()
        patch_wav2lip_compatibility()
        install_dependencies()
        verify_pytorch_gpu()
        print("\nSUCCESS: All models and dependencies are successfully set up!")
    except Exception as e:
        print(f"\nERROR: Setup failed: {e}", file=sys.stderr)
        sys.exit(1)