"""Model registry: every member exposes get_models(fast) -> {name: (pipeline, param_dist)}."""
from . import rajeshwari, varnika, aditya

MEMBERS = {"rajeshwari": rajeshwari, "varnika": varnika, "aditya": aditya}


def get_all(fast: bool = False) -> dict:
    """{model_name: (member, pipeline, param_distributions)} for all 13 models."""
    out = {}
    for member, mod in MEMBERS.items():
        for name, (pipe, params) in mod.get_models(fast).items():
            out[name] = (member, pipe, params)
    return out
