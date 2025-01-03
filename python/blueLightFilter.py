import datetime
from datetime import date, datetime, time

import numpy as np
import PIL.Image as Image
from math import exp
from datetime import time

MIN_LEVEL_BLUE: float = 0.5
assert 0 <= MIN_LEVEL_BLUE <= 1, "MIN_LEVEL_BLUE must be between 0 and 1"

W = 0.1
assert 0 <= W <= 1, "W must be between 0 and 1"
P = 0.5
assert P > 0, "P must be greater than 0"

STARTING_TIME: datetime = datetime.combine(date.today(), time(hour=11, minute=0, second=0)) # 5:00 PM
ENDING_TIME: datetime = datetime.combine(date.today(), time(hour=19, minute=0, second=0)) # 7:00 PM
assert STARTING_TIME < ENDING_TIME, "STARTING_TIME must be before ENDING_TIME"
TIME_WINDOW: int = (ENDING_TIME - STARTING_TIME).seconds

def apply_blue_light_filter(image: Image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    return Image.fromarray(_blue_light_filter(np.array(image), W, P))

def _sigmoid(x: float, w: float, p: float):
    return 1 / (1 + exp(-(x - p) / w))

def _filter(w: float, p: float):
    current_time = datetime.now()
    if current_time < STARTING_TIME:
        return 1
    if current_time >= ENDING_TIME:
        return MIN_LEVEL_BLUE
    x = (current_time - STARTING_TIME).seconds / TIME_WINDOW # NOTE: if the window becomes too small, the transition won't be smooth.
    return -_sigmoid(x, w, p) * (1 - MIN_LEVEL_BLUE) + 1

def _blue_light_filter(image: np.array, w:float, p: float):
    # Vectorize the filter function so that it will be applied to the blue channel
    mapping_function = np.vectorize(lambda x: x * _filter(w, p))
    # slice the blue channel
    blue_channel = image[:,:,2]
    # apply the filter
    blue_channel = mapping_function(blue_channel)
    # replace the blue channel
    image[:,:,2] = blue_channel
    return image

def _test():
    image = Image.open("../images/default.png")
    image = image.convert("RGB")
    image.show()
    image = _blue_light_filter(np.array(image), W, P)
    image = Image.fromarray(image)
    image.show()

if __name__ == "__main__":
    _test()
