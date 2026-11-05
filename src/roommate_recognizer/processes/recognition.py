import csv
import time

def run_recognition(buffer, stop, dropped, csv_path, delay_ms):
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame_id", "capture_time", "finish_time", "latency_ms", "result", "dropped_so_far"])
        try:
            while not stop.is_set():
                frame = buffer.get()
                if frame is None:
                    continue
                if delay_ms:
                    time.sleep(delay_ms / 1000)
                finish = time.time()
                latency = (finish - frame.capture_time) * 1000
                writer.writerow([frame.frame_id, frame.capture_time, finish, f"{latency:.1f}", "placeholder", dropped.value])
                file.flush()
                print(f"frame={frame.frame_id} latency={latency:.0f}ms dropped={dropped.value}")
        except KeyboardInterrupt:
            pass