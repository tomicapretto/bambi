class TransformsRegistry:
    def __init__(self):
        self.additive_predictors = {}
        self.parameters = {}
        self.data = {}

    def transform_predictor(self, family, parameter):
        """Register transformation function for additive predictors."""

        def decorator(function):
            self.additive_predictors[(family, parameter)] = function

            def wrapper(*args, **kwargs):
                return function(*args, **kwargs)

            return wrapper

        return decorator

    def transform_parameters(self, family):
        """Register transformation function for parameters of the observational model."""

        def decorator(function):
            self.parameters[(family,)] = function

            def wrapper(*args, **kwargs):
                return function(*args, **kwargs)

            return wrapper

        return decorator

    def transform_data(self, family):
        """Register transformation function for observational model data."""

        def decorator(function):
            self.data[(family,)] = function

            def wrapper(*args, **kwargs):
                return function(*args, **kwargs)

            return wrapper

        return decorator

    def get_transform_predictor(self, family, parameter):
        return self.additive_predictors.get((self._family_key(family), parameter), None)

    def get_transform_parameters(self, family):
        return self.parameters.get((self._family_key(family),), None)

    def get_transform_data(self, family):
        return self.data.get((self._family_key(family),), None)

    def _family_key(self, family):
        return family if isinstance(family, type) else type(family)


transforms_registry = TransformsRegistry()
