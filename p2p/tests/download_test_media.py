import os
import requests
import cv2

# 创建文件夹
os.makedirs("p2p/tests/test_images", exist_ok=True)
os.makedirs("p2p/tests/test_videos", exist_ok=True)

# 图片和视频的 URL 列表
test_images = [
    "https://upload.wikimedia.org/wikipedia/commons/3/3f/Fronalpstock_big.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
]

test_videos = [
    "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4"
]

# 下载函数，添加 User-Agent 请求头来避免 403 错误
def download_file(url: str, file_path: str):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        r = requests.get(url, headers=headers, stream=True)
        r.raise_for_status()  # 如果请求失败会抛出异常

        with open(file_path, "wb") as f:
            total_length = int(r.headers.get('content-length', 0))
            print(f"Downloading {url}... Total size: {total_length} bytes")

            downloaded = 0
            for chunk in r.iter_content(chunk_size=128):
                f.write(chunk)
                downloaded += len(chunk)
                print(f"Downloaded {downloaded}/{total_length} bytes", end='\r')

        print(f"\nDownloaded {url} to {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
    except Exception as e:
        print(f"Error while downloading {url}: {e}")

def download_file_with_retries(url: str, file_path: str, retries: int = 3):
    for attempt in range(retries):
        try:
            download_file(url, file_path)
            break  # 成功则跳出循环
        except Exception as e:
            if attempt < retries - 1:
                print(f"Attempt {attempt+1} failed. Retrying...")
            else:
                print(f"All attempts failed for {url}: {e}")


# 下载测试图片
for i, img_url in enumerate(test_images):
    download_file_with_retries(img_url, f"p2p/tests/test_images/test_image_{i + 1}.jpg")

# 下载测试视频
for i, vid_url in enumerate(test_videos):
    download_file_with_retries(vid_url, f"p2p/tests/test_videos/test_video_{i + 1}.mp4")

# 验证图片文件是否可用
def validate_image(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
    else:
        print(f"Image loaded successfully: {image_path}")

# 验证视频文件是否可用
def validate_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
    else:
        print(f"Video opened successfully: {video_path}")
    cap.release()

# 测试下载的图片和视频
validate_image("tests/test_images/test_image_1.jpg")
validate_image("tests/test_images/test_image_2.jpg")
validate_video("tests/test_videos/test_video_1.mp4")
