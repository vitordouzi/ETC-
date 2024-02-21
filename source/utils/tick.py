from time import time
class Tick(object):
    def __init__(self):
        super(Tick, self).__init__()
        self.timer = []
    @property
    def tick(self):
        self.timer.append( time() )
    @property
    def time(self):
        return (self.timer[-1] - self.timer[0])