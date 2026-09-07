import os
import sys

def test_pipeline():
    print("=== Integration Verification ===")
    
    # 1. PyTorch & CUDA
    try:
        import torch
        print(f"[+] PyTorch version: {torch.__version__}")
        print(f"[+] CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[+] GPU Device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"[-] PyTorch error: {e}")

    # 2. OpenCV
    try:
        import cv2
        print(f"[+] OpenCV version: {cv2.__version__}")
    except Exception as e:
        print(f"[-] OpenCV error: {e}")

    # 3. Coqui TTS / XTTS
    try:
        from TTS.api import TTS
        print("[+] TTS library imported successfully!")
        # Let's check model availability or initialization (lazy loaded, but we can verify class works)
        print("[+] TTS API class is available.")
    except Exception as e:
        print(f"[-] TTS import error: {e}")

    # 4. Wav2Lip Checkpoints
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gan_path = os.path.join(base_dir, "wav2lip", "checkpoints", "wav2lip_gan.pth")
    s3fd_path = os.path.join(base_dir, "wav2lip", "face_detection", "detection", "sfd", "s3fd.pth")
    
    print(f"[?] Checking Wav2Lip GAN weight at: {gan_path}")
    if os.path.exists(gan_path):
        print(f"[+] Wav2Lip GAN weight exists! Size: {os.path.getsize(gan_path) / (1024*1024):.2f} MB")
    else:
        print("[-] Wav2Lip GAN weight is MISSING!")

    print(f"[?] Checking face detector weight at: {s3fd_path}")
    if os.path.exists(s3fd_path):
        print(f"[+] Face detector weight exists! Size: {os.path.getsize(s3fd_path) / (1024*1024):.2f} MB")
    else:
        print("[-] Face detector weight is MISSING!")

if __name__ == "__main__":
    test_pipeline()
