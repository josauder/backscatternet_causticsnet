from torch import nn


def set_bn_momentum(module: nn.Module, momentum: float) -> None:
    """PyTorch convention: running stats update as ``(1 - momentum) * old + momentum * batch``."""
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.momentum = momentum
