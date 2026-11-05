import time
from roommate_recognizer.messages import Frame
from roommate_recognizer.sources.rtsp import open_rtsp

def run_camera(url, buffer, stop, dropped):
    # Lets this process exit even if frames remain unread
    buffer.frames.cancel_join_thread()
    capture = open_rtsp(url)
    frame_id = 0
    try:
        while not stop.is_set():
            ok, image = capture.read()
            if not ok:
                time.sleep(0.1)
                continue
            frame = Frame(frame_id, time.time(), image)
            if not buffer.put_or_drop(frame):
                with dropped.get_lock():
                    dropped.value += 1
            frame_id += 1
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()