class Token:
    def __init__(self, value, id=-1, document=-1, offset=-1, **kwargs):
        self.value = value
        self.id = id
        self.attr = {}
        self.attr['document'] = document
        self.attr['offset'] = offset
        self.attr['length'] = value.__len__()
        self.attr.update(**kwargs)

    def __str__(self):
        s = "id: {0}, value: {1}, attributes: {2}".format(
            self.id, self.value, self.attr)
        return s
