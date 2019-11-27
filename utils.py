import time
import colorsys

class Timer():
    def __init__(self):
        self.start = time.process_time()

    def stop(self):
        return time.process_time() - self.start


def get_colour(x, N):
    return (x/N, 0.5, 0.5)
    #return colorsys.hsv_to_rgb(x*1.0/N, 0.5, 0.5) 
