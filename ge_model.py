import os
import subprocess
import sys

# === EDIT THESE ===
FILE_ID = "1TDs3DUMqcP_s6AvmgnAVZ-jdhLM74StF"
OUT_PATH = "AlexLeNet-master/new_model.h5"

def have(cmd):
    from shutil import which
    return which(cmd) is not None

def download_with_gdown(file_id, out_path):
    cmd = ["gdown", "--id", file_id, "-O", out_path]
    print("Downloading with gdown:", " ".join(cmd))
    subprocess.check_call(cmd)

def download_with_curl(file_id, out_path):
    # Drive direct-download URL
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    cmd = ["curl", "-L", url, "-o", out_path]
    print("Downloading with curl:", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    if os.path.exists(OUT_PATH):
        print("Model already exists:", OUT_PATH)
        return

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    try:
        if have("gdown"):
            download_with_gdown(FILE_ID, OUT_PATH)
        elif have("curl"):
            download_with_curl(FILE_ID, OUT_PATH)
        else:
            print("Please install gdown (`pip install gdown`) or ensure curl is available.")
            sys.exit(1)
        print("Saved model to:", OUT_PATH)
    except subprocess.CalledProcessError as e:
        print("Download failed:", e)
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
