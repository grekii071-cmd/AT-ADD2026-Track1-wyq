import copy
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear expects an nn.Linear base layer")

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.scaling = alpha / r if r > 0 else 1.0
        self.base_layer = copy.deepcopy(base_layer)
        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if r > 0:
            self.lora_A = nn.Linear(self.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, self.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        out = self.base_layer(x)
        if self.r > 0:
            out = out + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return out


def _replace_target_modules(module: nn.Module, r: int, alpha: int, target_names=None):
    if target_names is None:
        target_names = {'q_proj', 'v_proj'}

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in target_names:
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=0.1))
        else:
            _replace_target_modules(child, r=r, alpha=alpha, target_names=target_names)


def build_lora_xlsr(pretrain_path: str = None, r: int = 8, alpha: int = 16, base_model: nn.Module = None) -> nn.Module:
    """Self-contained LoRA wrapper for XLS-R/Wav2Vec2 backbones.

    Args:
        pretrain_path: local path or HF id for the base model.
        r: LoRA rank.
        alpha: LoRA scaling factor.
        base_model: optionally pass an already loaded Wav2Vec2Model.
    """
    if base_model is None:
        if pretrain_path is None:
            raise ValueError("Either pretrain_path or base_model must be provided")
        base_model = Wav2Vec2Model.from_pretrained(pretrain_path)

    model = base_model
    _replace_target_modules(model, r=r, alpha=alpha)

    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, LoRALinear):
            for param in module.lora_A.parameters():
                param.requires_grad = True
            for param in module.lora_B.parameters():
                param.requires_grad = True

    return model
