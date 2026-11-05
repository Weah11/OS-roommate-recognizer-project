import os
import cv2

def open_rtsp(url):
    # TCP is usually more stable than UDP on Wi-Fi
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError("Could not open the RTSP stream")
    return capture