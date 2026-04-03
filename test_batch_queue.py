import time
import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"
TEST_FILE = r"C:\Users\Пользователь\Desktop\watermark\Араб.mp4"

def test_batch_queue():
    print(f"Starting test with file: {TEST_FILE}")
    if not Path(TEST_FILE).exists():
        print(f"Error: File {TEST_FILE} not found.")
        return

    # 1. Upload file
    print("Uploading file...")
    with open(TEST_FILE, "rb") as f:
        files = {"file": f}
        res = requests.post(f"{BASE_URL}/api/upload", files=files)
        res.raise_for_status()
        info = res.json()
        
    print("Upload successful. Info:", info)
    remote_path = info["path"]

    # 2. Add to queue
    print("Adding to queue...")
    payload = {
        "path": remote_path,
        "name": info["name"],
        "regions": [{"x": 10, "y": 10, "w": 100, "h": 50}],
        "duration": info["duration"],
        "fps": info["fps"],
        "width": info["width"],
        "height": info["height"],
        "mode": "delogo",
        "device": "cpu"
    }
    
    res = requests.post(f"{BASE_URL}/api/queue", json=payload)
    res.raise_for_status()
    queue_info = res.json()
    job_id = queue_info["job_id"]
    print(f"Added to queue. Job ID: {job_id}")

    # 3. Poll queue status
    print("Polling queue status...")
    while True:
        res = requests.get(f"{BASE_URL}/api/queue")
        res.raise_for_status()
        jobs = res.json()
        
        my_job = next((j for j in jobs if j["job_id"] == job_id), None)
        if not my_job:
            print("Job disappeared from queue!")
            return

        status = my_job["status"]
        progress = my_job.get("progress", 0)
        print(f"Status: {status}, Progress: {progress}%")
        
        if status == "done":
            download_url = my_job.get("download_url")
            print(f"Job completed successfully. Download URL: {download_url}")
            # Verify file exists locally
            filename = download_url.split("/")[-1]
            output_file = Path("temp_web") / filename
            if output_file.exists():
                print(f"SUCCESS: Output file {output_file} exists and size is {output_file.stat().st_size} bytes.")
            else:
                print(f"ERROR: Output file {output_file} not found locally.")
            break
        elif status == "error":
            print(f"Job failed with error: {my_job.get('error')}")
            break
            
        time.sleep(2)

if __name__ == "__main__":
    test_batch_queue()