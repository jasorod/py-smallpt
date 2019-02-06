from random import Random

class RNG(object):
    
    def __init__(self, s = 606418532):
        self.rnd = Random()
        self.seed(s)
    
    def seed(self, s):
        self.rnd.seed(s)

    def uniform_float(self):
        return self.rnd.random()

    def uniform_range_float(self, a, b):
        return self.rnd.uniform(a, b)