from .vector import VectorModel
from .polynomial import PolynomialModel




def get_filter_model(type: str, **kwargs):
    """
    Returns the model with the given name
    Note: For unet the n_projections is not used
    """
    print(type)
    if type == 'vector':
        return VectorModel(**kwargs)
    if type == 'polynomial':
        return PolynomialModel(**kwargs)
    else:
        raise NotImplementedError(f"Model {type} not implemented")