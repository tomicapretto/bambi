# pylint: disable=unused-argument
from bambi.families.family import Family
from bambi.transformations import transformations_namespace
from bambi.utils import extract_argument_names, get_aliased_name

# NOTE: How do we go from reduced to complete dims?
#       This is the case for models such as Categorical, Multinomial, etc.
#       Basically, every place where we use a reference encoding in the response
#       (constrained multivariate responses)
#       Should we pad directly in the pymc model?
#       Well, we actually do, but we're not registering it as a deterministic

# NOTE: How to fully decouple frontend from backend?
# Visitor pattern?
# I think it's impossible. The backend needs pytensor operations.
# We can't write them there without making Bambi even harder to be extended
# (unless we want to pay that price)


class MultivariateFamily(Family):
    KIND = "Multivariate"


class Multinomial(MultivariateFamily):
    SUPPORTED_LINKS = {"p": ["softmax"]}
    INVLINK_KWARGS = {"axis": -1}

    def get_coords(self, response):
        # For the moment, it always uses the first column as reference.
        name = get_aliased_name(response) + "_reduced_dim"
        labels = self.get_levels(response)
        return {name: labels[1:]}

    def get_levels(self, response):
        labels = extract_argument_names(response.name, list(transformations_namespace))
        if labels:
            return labels
        return [str(level) for level in range(response.data.shape[1])]


class DirichletMultinomial(MultivariateFamily):
    SUPPORTED_LINKS = {"a": ["log"]}

    def get_coords(self, response):
        name = get_aliased_name(response) + "_dim"
        labels = self.get_levels(response)
        return {name: labels}

    def get_levels(self, response):
        labels = extract_argument_names(response.name, list(transformations_namespace))
        if labels:
            return labels
        return [str(level) for level in range(response.data.shape[1])]
