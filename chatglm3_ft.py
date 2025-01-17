import argparse
import os
import sys
from typing import Dict, List, Union

import fire
import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    LlamaTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from utils.prompter import AlpacaPrompter, PromptSelector
from utils.text import load_text_file

def parse_args():
    parser = argparse.ArgumentParser(description='ChatGLM-6B QLoRA')
    #parser.add_argument('--train_args_json', type=str, required=True, help='TrainingArguments的json文件')
    parser.add_argument('--train_args_json', type=str, required=False, help='TrainingArguments的json文件')
    parser.add_argument('--model_name_or_path', type=str, default='THUDM/chatglm-6b', help='模型id或local path')
    #parser.add_argument('--train_data_path', type=str, required=True, help='训练数据路径')
    parser.add_argument('--train_data_path', type=str, required=False, help='训练数据路径')
    parser.add_argument('--eval_data_path', type=str, default=None, help='验证数据路径')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_input_length', type=int, default=512, help='instruction + input的最大长度')
    parser.add_argument('--max_output_length', type=int, default=1536, help='output的最大长度')
    parser.add_argument('--lora_rank', type=int, default=4, help='lora rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='lora_alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='恢复训练的checkpoint路径')
    parser.add_argument('--prompt_text', type=str, default='', help='统一添加在所有数据前的指令文本')
    parser.add_argument('--base_model', type=str, default='', help='统一添加在所有数据前的指令文本')
    parser.add_argument('--data_path', type=str, default='', help='统一添加在所有数据前的指令文本')
    parser.add_argument('--output_dir', type=str, default='', help='统一添加在所有数据前的指令文本')
    parser.add_argument('--compute_dtype', type=str, default='fp32',
                        choices=['fp32', 'fp16', 'bf16'], help='计算数据类型')
    return parser.parse_args()

def tokenize_func(example, tokenizer, global_args, ignore_label_id=-100):
    """单样本tokenize处理"""
    question = global_args.prompt_text + example['instruction']
    if example.get('input', None):
        if example['input'].strip():
            question += f'''\n{example['input']}'''
    answer = example['output']
    q_ids = tokenizer.encode(text=question, add_special_tokens=False)
    a_ids = tokenizer.encode(text=answer, add_special_tokens=False)
    if len(q_ids) > global_args.max_input_length - 2:  # 2 - gmask, bos
        q_ids = q_ids[: global_args.max_input_length - 2]
    if len(a_ids) > global_args.max_output_length - 1:  # 1 - eos
        a_ids = a_ids[: global_args.max_output_length - 1]
    input_ids = tokenizer.build_inputs_with_special_tokens(q_ids, a_ids)
    # question_length = input_ids.index(tokenizer.bos_token_id)
    question_length = len(q_ids) + 2  # chatglm1 - gmask, bos, chatglm2 - gmask, sop
    labels = [ignore_label_id] * question_length + input_ids[question_length:]
    return {'input_ids': input_ids, 'labels': labels}


def get_datset(data_path, tokenizer, global_args):
    """读取本地数据文件，并tokenize，shuffle，返回datasets.dataset"""
    data = load_dataset('json', data_files=data_path)
    column_names = data['train'].column_names
    dataset = data['train'].map(lambda example: tokenize_func(example, tokenizer, global_args),
                                batched=False, remove_columns=column_names)
    dataset = dataset.shuffle(seed=global_args.seed)
    dataset = dataset.flatten_indices()
    return dataset


class DataCollatorForChatGLM:
    def __init__(self,
                 pad_token_id: int,
                 max_length: int = 2048,
                 ignore_label_id: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_label_id = ignore_label_id
        self.max_length = max_length

    def __call__(self, batch_data: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        """根据batch最大长度做padding"""
        len_list = [len(d['input_ids']) for d in batch_data]
        batch_max_len = max(len_list)
        input_ids, labels = [], []
        for len_of_d, d in sorted(zip(len_list, batch_data), key=lambda x: -x[0]):
            pad_len = batch_max_len - len_of_d
            ids = d['input_ids'] + [self.pad_token_id] * pad_len
            label = d['labels'] + [self.ignore_label_id] * pad_len
            if batch_max_len > self.max_length:
                ids = ids[: self.max_length]
                label = label[: self.max_length]
            input_ids.append(torch.LongTensor(ids))
            labels.append(torch.LongTensor(label))
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        return {'input_ids': input_ids, 'labels': labels}


class PeftTrainer(Trainer):
    def _save_checkpoint(self, _, trial, metrics=None):
        """Don't save base model, optimizer etc.
        but create checkpoint folder (needed for saving adapter)"""
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        os.makedirs(output_dir, exist_ok=True)

        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)


class PeftSavingCallback(TrainerCallback):
    """Correctly save PEFT model and not full model"""

    def _save(self, model, folder):
        peft_model_path = os.path.join(folder, "adapter_model")
        model.save_pretrained(peft_model_path)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Save final best model adapter"""
        pass

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        self._save(kwargs["model"], folder)


def load_hf_model(
    base_model,
    lora_config=None,
    mode=8,
    gradient_checkpointing=False,
    device_map="auto",
):
    from peft import prepare_model_for_kbit_training

    kwargs = {"device_map": device_map}
    if mode == 8:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=0.0,
        )
    elif mode == 4:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif mode == 16:
        kwargs["torch_dtype"] = torch.float16

    kwargs['trust_remote_code'] = True
    #model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    model = AutoModel.from_pretrained(base_model, **kwargs)

    #for i in model.named_parameters():
    #    print(f"{i[0]} -> {i[1].device}")

    # setup tokenizer
    #tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    #if gradient_checkpointing:
    #    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    if lora_config:
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    return model, tokenizer


# noinspection PyTypeChecker
def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "data/alpaca_data.json",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 64,
    micro_batch_size: int = 16,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    #eval_steps: int = 100,
    eval_steps: int = 0,
    save_steps: int = 10,
    logging_steps: int = 10,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ( "q_proj", "v_proj",),
    #lora_target_modules: List[str] = ( "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h",),
    #lora_target_modules: List[str] = ["query_key_value"],
    #target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template: str = "vicuna",  # The prompt template to use, will default to alpaca.
    # memory optimization params
    mode: Union[int, str] = 8,  # training floating point mode
    gradient_checkpointing: bool = False,
    # GPTQ specific params
    gptq_backend: str = "cuda",  # GPTQ backend "cuda" or "triton"
    gptq_groupsize: int = 128,
    # evaluation flag
    eval: bool = False,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"mode: {mode}\n"
            f"eval: {eval}\n"
            f"gradient_checkpointing: {gradient_checkpointing}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"eval_steps: {eval_steps}\n"
            f"logging_steps: {logging_steps}\n"
            f"save_steps: {save_steps}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt_template: {prompt_template}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = PromptSelector.from_template_name(prompt_template, verbose=False)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    print("use_wandb", use_wandb)
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # setup model and tokenizer
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if isinstance(mode, int):
        # use HF loader for normal model loading with bitsandbytes quantization
        model, tokenizer = load_hf_model(
            base_model,
            lora_config,
            mode=mode,
            gradient_checkpointing=gradient_checkpointing,
            device_map=device_map,
        )

        # setup model checkpoint if neeeded
        if resume_from_checkpoint:
            # Check the available weights and load them
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "pytorch_model.bin"
            )  # Full checkpoint
            if not os.path.exists(checkpoint_name):
                checkpoint_name = os.path.join(
                    resume_from_checkpoint, "adapter_model.bin"
                )  # only LoRA model - LoRA config above has to fit
            # The two files above have a different name depending on how they were saved, but are actually the same.
            if os.path.exists(checkpoint_name):
                print(f"Restarting from {checkpoint_name}")
                adapters_weights = torch.load(checkpoint_name, map_location="cpu")
                set_peft_model_state_dict(model, adapters_weights)
            else:
                print(f"Checkpoint {checkpoint_name} not found")

    elif mode == "exl2":
        from utils.loader.exllama_hf_loader import get_lora_exllama

        kwargs = {
            "gradient_checkpointing": gradient_checkpointing,
            "device_map": device_map,
            "group_size": gptq_groupsize,
            "backend": gptq_backend,
        }
        if resume_from_checkpoint:
            kwargs.update(
                {
                    "lora_path": resume_from_checkpoint,
                    "load_lora": True,
                    "lora_trainable": True,
                }
            )
            print(f"Restarting from {resume_from_checkpoint}")

        model, tokenizer = get_lora_exllama(
            base_model,
            lora_config,
            **kwargs,
        )
    else:
        raise NotImplementedError(f"Mode '{mode}' is not supported.")

    """
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(**data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            assert isinstance(prompter, AlpacaPrompter)
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    # data preparation
    # check if using raw text format (prompter is None)
    if prompter is None:
        train_data = load_text_file(
            data_path, tokenizer, cutoff_len=cutoff_len, overlap_len=cutoff_len // 2
        )

        if val_set_size > 0:
            train_val = train_data.train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            train_data = train_val["train"].shuffle()
            val_data = train_val["test"]
        else:
            val_data = None
    else:
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            data = load_dataset("json", data_files=data_path)
        else:
            if os.path.exists(data_path):
                data = load_dataset(
                    "json",
                    data_files={
                        "train": data_path + "/train.json",
                        "test": data_path + "/test.json",
                    },
                )
            else:
                data = load_dataset(data_path)

        if val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        else:
            train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
            val_data = data["test"].map(generate_and_tokenize_prompt)
    """
    # data
    global_args = parse_args()
    train_data = get_datset(data_path, tokenizer, global_args)
    val_data = None
    if global_args.eval_data_path:
        eval_dataset = get_datset(global_args.eval_data_path, tokenizer, global_args)

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    model.config.use_cache = False
    # sanity check of model saving process
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        model.save_pretrained(output_dir)

    trainer = PeftTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=logging_steps,
            optim="paged_adamw_8bit" if mode in [4, 8] else "adamw_torch",
            evaluation_strategy="steps" if eval_steps > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_steps if eval_steps > 0 else None,
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        #data_collator=transformers.DataCollatorForSeq2Seq(
        #    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        #),
        data_collator = DataCollatorForChatGLM(pad_token_id=tokenizer.pad_token_id,
                                      max_length=2048
        ),

        callbacks=[PeftSavingCallback],
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    if eval:
        eval_results = trainer.evaluate()
        print(eval_results)
    else:
        trainer.train()

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            model.save_pretrained(output_dir)
            print(
                "\n If there's a warning about missing keys above, please disregard :)"
            )


if __name__ == "__main__":
    fire.Fire(train)
