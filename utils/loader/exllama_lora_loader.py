import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from exllamav2 import ExLlamaV2Lora
from exllamav2 import ExLlamaV2, ExLlamaV2Cache_8bit, ExLlamaV2Config
from exllamav2.linear import  ExLlamaV2Linear
from transformers import (
    GenerationConfig,
    LlamaTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from peft import PeftModel, get_peft_model, LoraConfig


class WrapParent(object):
    def __init__(self, name) -> None:
        self.name = name
        self.dict = {}

class WrapExLlamaV2(ExLlamaV2):

    def __init__(self, config: ExLlamaV2Config, lora_config: LoraConfig, lazy_load = False):
        super().__init__(config, lazy_load)

        self.lora_config = lora_config
        self.named_dict = {}
        self.named_parents = {}

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    def init_dict(self):

        m_dict = self.modules_dict

        for m_key in m_dict.keys():
            m = m_dict[m_key]
            c_name =  m.__class__.__name__

            print ( c_name )

            self.named_dict[m_key] = m

            """
            if isinstance(m, ExLlamaV2Linear):
                dq_weights = m.get_weight_tensor_dq()
                #new_m = torch.nn.Linear(m.in_features, m.out_features, m.has_bias, device=m.device(), dtype=dq_weights.dtype)
                new_m = torch.nn.Linear(m.in_features, m.out_features, m.has_bias, device='cpu', dtype=dq_weights.dtype)
                with torch.no_grad():
                    new_m.weight.copy_(dq_weights.t())

                del dq_weights

                self.named_dict[m_key] = new_m
            """


        #`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D` 


    def named_modules(self):
        return self.named_dict.items()


    def init_lora_loftq(self, orig_model_name: str):
        pass

    """
    def _get_submodules(model, key):
        parent = model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = model.get_submodule(key)
        return parent, target, target_name
    """
    def get_submodule(self, target: str) -> torch.nn.Module:
        if target in self.named_dict:
            m = self.named_dict[target]
            if self.last_fake_parent is not None:
                target_name = target.split(".")[-1]
                if target_name in self.lora_config.target_modules:
                    if isinstance(m, ExLlamaV2Linear):
                        dq_weights = m.get_weight_tensor_dq()
                        #new_m = torch.nn.Linear(m.in_features, m.out_features, m.has_bias, device=m.device(), dtype=dq_weights.dtype)
                        new_m = torch.nn.Linear(m.in_features, m.out_features, m.has_bias, device='cpu', dtype=dq_weights.dtype)
                        with torch.no_grad():
                            new_m.weight.copy_(dq_weights.t())

                        del dq_weights
                        setattr(self.last_fake_parent, target_name, new_m)

                        self.named_parents[target]= self.last_fake_parent

                        self.last_fake_parent = None
                        return new_m
            return m
        else:
            # fake parent
            fake_parent = WrapParent(target)
            self.last_fake_parent = fake_parent
            return fake_parent

    def named_parameters(self):

        return {}

class ExllamaHF(PreTrainedModel):
    def __init__(self, config: ExLlamaV2Config):
        super().__init__(PretrainedConfig())
        print(config)
        self.ex_config = config
        self.ex_model = ExLlamaV2(self.ex_config)
        self.generation_config = GenerationConfig()
        self.lora = None

    def _validate_model_class(self):
        pass

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        pass

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    @property
    def device(self) -> torch.device:
        return torch.device(0)

    def __call__(self, *args, **kwargs):
        # TODO: Some decoding methods (such as Contrastive Search) may not work at this time
        assert len(args) == 0, "no *args should be passed to forward"
        use_cache = kwargs.get("use_cache", True)
        seq = kwargs["input_ids"][0].tolist()
        cache = kwargs["past_key_values"] if "past_key_values" in kwargs else None
        if cache is None:
            cache = ExLlamaV2Cache_8bit(self.ex_model)
            self.ex_model.forward(
                torch.tensor([seq[:-1]], dtype=torch.long),
                cache,
                preprocess_only=True,
                lora=self.lora,
            )

        logits = self.ex_model.forward(
            torch.tensor([seq[-1:]], dtype=torch.long), cache, lora=self.lora
        ).to(kwargs["input_ids"].device)

        return CausalLMOutputWithPast(
            logits=logits, past_key_values=cache if use_cache else None
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        lora_config: LoraConfig,
        *model_args,
        **kwargs,
    ):
        assert (
            len(model_args) == 0 and len(kwargs) == 0
        ), "extra args is currently not supported"
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        #config = ExLlamaV2Config(pretrained_model_name_or_path / "config.json")
        config = ExLlamaV2Config()
        config.model_dir = pretrained_model_name_or_path
        config.prepare()
        
        model = WrapExLlamaV2(config, lora_config)
        model.load()
        model.init_dict()

        """
        # from 'oobabooga/text-generation-webui/modules/exllama.py'
        weight_path = None
        for ext in [".safetensors", ".pt", ".bin"]:
            found = list(pretrained_model_name_or_path.glob(f"*{ext}"))
            if len(found) > 0:
                weight_path = found[-1]
                break
        assert (
            weight_path is not None
        ), f'could not find weight in "{pretrained_model_name_or_path}"'

        config.model_path = str(weight_path)
        """

        config.max_seq_len = 2048
        config.compress_pos_emb = 1

        if torch.version.hip:
            config.rmsnorm_no_half2 = True
            config.rope_no_half2 = True
            config.matmul_no_half2 = True
            config.silu_no_half2 = True

        # This slowes down a bit but align better with autogptq generation.
        # TODO: Should give user choice to tune the exllama config
        # config.fused_attn = False
        # config.fused_mlp_thd = 0

        return model


def load_lora(model, lora_path):
    lora_config_path = os.path.join(lora_path, "adapter_config.json")
    lora_path = os.path.join(lora_path, "adapter_model.bin")
    lora = ExLlamaV2Lora(model.ex_model, lora_config_path, lora_path)
    model.lora = lora

    return model


def load_model_exllama(base, lora_config, delta):
    model = ExllamaHF.from_pretrained(base, lora_config)

    # load normal HF tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(base, use_fast=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # setup PEFT model
    if os.path.exists(delta):
        model = load_lora(model, delta)

    return model, tokenizer

def get_lora_exllama(
    base,
    lora_config,
    group_size=128,
    gradient_checkpointing=False,
    lora_path=None,
    load_lora=True,
    lora_trainable=False,
    backend="cuda",
    device_map="auto",
):

    lora_ck_path = ""
    model, tokenizer = load_model_exllama(base, lora_config, lora_ck_path)

    if load_lora:
        if lora_path:
            model = PeftModel.from_pretrained(
                model, lora_path, is_trainable=lora_trainable
            )
        else:
            model = get_peft_model(model, lora_config)

    # Scales to half
    print("Fitting x bits scales and zeros to half")
    for _, m in model.named_modules():
        print(str(type(m)), m)
        if "Autograd4bitQuantLinear" in str(type(m)) or "Linear4bitLt" in str(type(m)):
            if hasattr(m, "is_v1_model") and m.is_v1_model:
                m.zeros = m.zeros.half()
            m.scales = m.scales.half()

    if gradient_checkpointing:
        # Use gradient checkpointing
        print("not Applying gradient checkpointing ...")
        #apply_gradient_checkpointing(model, checkpoint_ratio=1.0)

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
