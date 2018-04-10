class TrustRegion:
    def __init__(self, radius):
        self.radius = radius
        # or whatever other structure for storing data
        self.cache = {}

    def update(self, x, step, model):
        # change self.radius
        pass

    def check_step(self, step):
        # check if step should be accepted
        return True
