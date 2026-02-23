def is_nonempty(data):
    try:
        return data is not None and len(data) > 0
    except TypeError:
        try:
            return data is not None and not data.empty
        except AttributeError:
            return data is not None
