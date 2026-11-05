import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from roommate_recognizer.ipc.queue_buffer import QueueBuffer
from roommate_recognizer.processes.camera import run_camera
from roommate_recognizer.processes.recognition import run_recognition

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-size", type=int, default=5)
    parser.add_argument("--delay-ms", type=int, default=0)
    parser.add_argument("--duration", type=int, default=120)
    args = parser.parse_args()

    load_dotenv()
    run_dir = Path("data/runs") / f"{int(time.time())}_buffer{args.buffer_size}_delay{args.delay_ms}"
    run_dir.mkdir(parents=True)

    buffer = QueueBuffer(mp.Queue(maxsize=args.buffer_size))
    stop = mp.Event()
    dropped = mp.Value("Q", 0)

    camera = mp.Process(target=run_camera, args=(os.environ["RTSP_URL"], buffer, stop, dropped))
    recognition = mp.Process(target=run_recognition, args=(buffer, stop, dropped, run_dir / "results.csv", args.delay_ms))
    camera.start()
    recognition.start()

    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for process in (camera, recognition):
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        print(f"Dropped frames: {dropped.value}")
        print(f"Results: {run_dir}")

if __name__ == "__main__":
    main()