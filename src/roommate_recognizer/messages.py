from dataclasses import dataclass
import numpy as np

@dataclass
class Frame:
    frame_id: int
    capture_time: float
    image: np.ndarray