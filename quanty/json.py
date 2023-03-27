import collections
import json
import warnings
from dataclasses import asdict, is_dataclass


def default(obj):
    data = {}

    if obj.__class__.__name__ == "function":
        data = {
            "__class__": {
                "__module__": obj.__module__,
                "__name__": obj.__name__,
            }
        }
        warnings.warn(f"function '{obj.__name__}' from '{obj.__module__}' dumped")
        return data

    data = {
        "__class__": {
            "__module__": obj.__class__.__module__,
            "__name__": obj.__class__.__name__,
        }
    }

    args, kwargs = None, None
    if isinstance(obj, complex):
        args = obj.real, obj.imag

    if isinstance(obj, (set, frozenset)):
        args = [list(obj)]

    if isinstance(obj, collections.deque):
        if len(obj) != 0 and obj.maxlen is not None:
            args = [list(obj), obj.maxlen]

        elif len(obj) == 0:
            kwargs = {"maxlen": obj.maxlen}

        elif obj.maxlen is None:
            args = [list(obj)]

    if is_dataclass(obj):
        kwargs = {k: getattr(obj, k) for k in asdict(obj).keys()}

    if hasattr(obj, "as_args_kwargs"):
        args, kwargs = obj.as_args_kwargs()

    if args is not None or kwargs is not None:
        data["__init__"] = [[], {}]

        if args is not None and len(args) != 0:
            data["__init__"][0] = args

        if kwargs is not None and len(kwargs.keys()) != 0:
            data["__init__"][1] = kwargs

    # elif hasattr(obj, "__dict__"):
    #     data["__dict__"] = obj.__dict__

    else:
        raise TypeError(f"can not serialize: {type(obj)}")

    return data


def dumps(obj, **kwargs) -> str:
    return json.dumps(obj, default=default, **kwargs)


def object_hook(dct: dict):
    if cls_dct := dct.get("__class__"):
        module_name = cls_dct["__module__"]
        class_name = cls_dct["__name__"]
        module = __import__(module_name, globals(), locals(), [class_name], 0)
        cls = getattr(module, class_name)

        if "__init__" in dct:
            args, kwargs = dct["__init__"]
            return cls(*args, **kwargs)

        return cls
        # if data := dct.get("__dict__"):
        #     instance = object.__new__(cls)
        #     instance.__dict__ = data
        #     return instance

    return dct


def loads(data: str):
    return json.loads(data, object_hook=object_hook)
