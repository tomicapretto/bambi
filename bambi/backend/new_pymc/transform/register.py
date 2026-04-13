TRANSFORMATIONS = {}
PARAMETERS_MANIPULATIONS = {}
DATA_MANIPULATIONS = {}


def transform_additive_predictor(family, parameter):
    def decorator(function):
        TRANSFORMATIONS[(family, parameter)] = function

        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)

        return wrapper

    return decorator


def manipulate_parameters(family):
    def decorator(function):
        PARAMETERS_MANIPULATIONS[(family,)] = function

        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)

        return wrapper

    return decorator


def manipulate_data(family):
    def decorator(function):
        DATA_MANIPULATIONS[(family,)] = function

        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)

        return wrapper

    return decorator
