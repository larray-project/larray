from larray.inout.common import FileHandler


handler_classes = {}
ext_default_engine = {}


def register_file_handler(engine, extensions=None):
    r"""Class decorator to register new file handler class.

    Parameters
    ----------
    engine : str
        Engine name associated with the file handler.
    extensions : str or list of str, optional
        Extension(s) associated with the file handler.
    """
    def decorate_class(cls: FileHandler) -> FileHandler:
        if engine not in handler_classes:
            handler_classes[engine] = cls
        if extensions is None:
            exts = []
        elif isinstance(extensions, str):
            exts = [extensions]
        else:
            exts = extensions
        for ext in exts:
            ext_default_engine[ext] = engine
        return cls
    return decorate_class


def get_file_handler(engine) -> FileHandler:
    if engine not in handler_classes:
        raise TypeError(f"Engine {engine} is currently not implemented")
    file_handler_cls = handler_classes[engine]
    return file_handler_cls
