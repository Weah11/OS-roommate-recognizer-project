import os
import time
import cv2
from dotenv import load_dotenv
from roommate_recognizer.sources.rtsp import open_rtsp

load_dotenv()
capture = open_rtsp(os.environ["RTSP_URL"])
count, start = 0, time.time()

while True:
    ok, frame = capture.read()
    if not ok:
        print("Frame read failed")
        break
    count += 1
    if time.time() - start >= 1:
        print(f"{count} FPS, frame shape {frame.shape}")
        count, start = 0, time.time()
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()