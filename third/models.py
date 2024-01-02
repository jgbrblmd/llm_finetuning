import re
import torch
import warnings

from peft.tuners import lora
from peft.tuners.lora import Linear, LoraLayer
from peft.utils import _get_submodules
from transformers.pytorch_utils import Conv1D

from alpaca_lora_4bit.autograd_4bit import Autograd4bitQuantLinear


class Linear4bitLt(Autograd4bitQuantLinear, LoraLayer):

    # Lora implemented in a dense layer
    def __init__(
            self,
            adapter_name,
            base_layer,
            in_features,
            out_features,
            groupsize: int = -1,
            is_v1_model: bool = False,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
    ):
        Autograd4bitQuantLinear.__init__(
            self,
            in_features,
            out_features,
            groupsize,
            is_v1_model
        )
        #LoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        LoraLayer.__init__(self, base_layer)

        # Freezing the pre-trained weight matrix
        self.qweight.requires_grad = False
        self.scales.requires_grad = False
        if self.is_v1_model:
            self.zeros.requires_grad = False
        else:
            self.qzeros.requires_grad = False
            self.g_idx.requires_grad = False
        self.bias.requires_grad = False

        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.set_adapter(adapter_names=[adapter_name])

    def forward(self, x: torch.Tensor):
        result = super().forward(x)
        previous_dtype = result.dtype

        if self.disable_adapters:
            return result
        else:
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)
                result += lora_B(lora_A(dropout(x))) * scaling

        result = result.to(previous_dtype)
        return result

    @property
    def weight(self):
        class WeightDeviceClass:
            device = self.qweight.device
            dtype = self.qweight.dtype
        return WeightDeviceClass()


class GPTQLoraModel(lora.LoraModel):
    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        bias = kwargs.pop("bias", False)

        if isinstance(target, Autograd4bitQuantLinear):
            new_module = Linear4bitLt(adapter_name, target, target.in_features, target.out_features, target.groupsize, target.is_v1_model, bias=bias, **kwargs)
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = Linear(adapter_name, in_features, out_features, bias=bias, **kwargs)

        return new_module


    @staticmethod
    def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        if isinstance(child, Autograd4bitQuantLinear) and isinstance(new_module, Linear4bitLt):
            new_module.qweight = child.qweight
            new_module.scales = child.scales
            if child.is_v1_model:
                new_module.zeros = child.zeros
            else:
                new_module.qzeros = child.qzeros
                new_module.g_idx = child.g_idx
            new_module.bias = child.bias
            if getattr(child, "state", None) is not None:
                new_module.state = child.state
                new_module.to(child.qweight.device)

            # dispatch to correct device
            for name, module in new_module.named_modules():
                if "lora_" in name:
                    module.to(child.qweight.device)
                if "ranknum" in name:
                    module.to(child.qweight.device)
        else:
            # child layer wraps the original module, unpack it
            if hasattr(child, "base_layer"):
                child = child.base_layer
            elif hasattr(child, "quant_linear_module"):
                child = child.quant_linear_module

            # TODO: layers with base_layer don't need the weight to be copied, as they have a reference already
            if not hasattr(new_module, "base_layer"):
                new_module.weight = child.weight
                if hasattr(child, "bias"):
                    new_module.bias = child.bias

            if getattr(child, "state", None) is not None:
                if hasattr(new_module, "base_layer"):
                    new_module.base_layer.state = child.state
                else:
                    new_module.state = child.state
                new_module.to(child.weight.device)

            # dispatch to correct device
            for name, module in new_module.named_modules():
                if "lora_" in name:
                    module.to(child.weight.device)
                if "ranknum" in name:
                    module.to(child.weight.device)
