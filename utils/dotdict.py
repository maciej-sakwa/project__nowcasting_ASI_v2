class DotDict(dict):

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dictionary: dict):
        for key, value in dictionary.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value