class ObjectAnnotation:
    def __init__(self, bound_box, object_type):
        self.type = object_type
        self.bound_box = bound_box
        self.p1 = (bound_box[0],bound_box[1])
        self.p2 = (bound_box[2],bound_box[3])
    def __repr__(self):
        return self.type + " " + str(self.bound_box)