def complex2json(obj: complex):
    return {"__complex__": [obj.real, obj.imag]}


def json2complex(dct):
    key = "__complex__"
    return complex(*dct[key]) if key in dct else dct


def default(obj):
    if isinstance(obj, complex):
        return handle_complex(obj)
    return json.JSONEncoder.default(self, obj)
