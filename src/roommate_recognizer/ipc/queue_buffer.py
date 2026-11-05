import queue

class QueueBuffer:
    def __init__(self, frames):
        self.frames = frames

    def put_or_drop(self, item):
        try:
            self.frames.put_nowait(item)
            return True
        except queue.Full:
            return False

    def get(self, timeout=1):
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty:
            return None