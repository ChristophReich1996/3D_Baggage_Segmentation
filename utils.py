import time


class Timer():
    def __init__(self):
        self.start = time.process_time()

    def stop(self):
        return time.process_time() - self.start