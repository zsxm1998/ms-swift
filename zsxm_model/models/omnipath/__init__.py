from typing import TYPE_CHECKING

from transformers.utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available


_import_structure = {
    "configuration_llava": ["LlavaConfig"],
    "processing_llava": ["LlavaProcessor"],
}


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_llava"] = [
        "LlavaForConditionalGeneration",
        "LlavaPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_omnipath import LlavaConfig
    from .processing_omnipath import LlavaProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_omnipath import (
            LlavaForConditionalGeneration,
            LlavaPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
