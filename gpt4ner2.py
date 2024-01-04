'''
    GPT2微调完成NER任务  gpt2-xlarge-chinese-cluecorpussmall
    Qlora -4bit + 半精度float16
'''

from torch.utils.data import Dataset, DataLoader, random_split
from transformers.data.data_collator import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from transformers import BitsAndBytesConfig
from transformers import GPT2LMHeadModel, BertTokenizer, pipeline, AutoModelForCausalLM
from transformers import get_scheduler, TrainingArguments, Trainer, TrainerCallback
from torch.nn.utils import clip_grad_norm_
import json, os
import torch
from tqdm import tqdm
import argparse
import time
from time import strftime, gmtime
import warnings
import logging
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, prepare_model_for_kbit_training
from bitsandbytes.optim import AdamW8bit
import random

warnings.filterwarnings("ignore")

device = "cuda"


class MyDataset(Dataset):
    '''
    从文件读取医疗文本数据
    '''

    def __init__(self, tokenizer):
        filepath = "D:\\pythonwork\\W2NER\\data\\OriginalFiles\\train_span.txt"
        prompt_prefix = "你是一个命名实体识别模型，请用Python字典的形式列出后面文本中类别为\"治疗\"、\"身体部位\"、\"症状和体征\"、\"检查和检验\"和 \"疾病和诊断\"的实体的位置和类别，实体与类别之间用\"_\"连接。文本："
        prompt_suffix = "实体有："

        zy = {
            "TREATMENT": "治疗",
            "BODY": "身体部位",
            "SIGNS": "症状和体征",
            "CHECK": "检查和检验",
            "DISEASE": "疾病和诊断",
        }

        with open(filepath, "r", encoding="utf-8") as f:
            all_datas = json.load(f)

        self.nerdata = []
        for data in tqdm(all_datas):
            text = data["context"][:pargs.max_length // 2]
            seqlen = len(text)

            ask = prompt_prefix + text + prompt_suffix
            ask = '[CLS]' + ask

            ask = tokenizer(ask, add_special_tokens=False)

            answer = '{'
            for se, label in data['span_posLabel'].items():
                startid, endid = se.split(";")
                if int(endid) < seqlen:
                    entity = text[int(startid):int(endid) + 1]
                    answer += "\"" + se + "\":\"" + entity + "_" + zy[label] + "\","

            answer = answer.rstrip(",") + "}[SEP]"
            answer = tokenizer(answer, add_special_tokens=False)

            input_ids = ask["input_ids"] + answer["input_ids"]
            attention_mask = ask["attention_mask"] + answer["attention_mask"]
            labels = [-100] * len(ask['input_ids']) + answer["input_ids"]

            self.nerdata.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

        random.shuffle(self.nerdata)

    def __len__(self):
        return len(self.nerdata)

    def __getitem__(self, idx):
        data = self.nerdata[idx]

        return data


def compute_metrics(data):
    preds, labels = data
    preds = torch.tensor(preds).to(device)
    labels = torch.tensor(labels).to(device)

    preds_shift = preds[:, :-1]
    preds = torch.argmax(preds_shift, dim=-1)
    labels_shift = labels[:, 1:]
    keeplabels = torch.logical_not(torch.eq(labels_shift, -100))

    rightlabel = torch.sum(torch.eq(preds, labels_shift)).item()
    total_label = torch.sum(keeplabels).item()

    eval_acc = rightlabel / total_label

    return {'acc': eval_acc}


def format_time(time):
    if time >= 3600:
        return strftime("%H:%M:%S", gmtime(time))
    else:
        return strftime("%M:%S", gmtime(time))


def create_logger(name, filename):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)

    # consoleHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(filename=filename, mode="a", encoding="utf-8")

    simple_formatter = logging.Formatter(fmt="%(asctime)s %(message)s",
                                         datefmt="%Y-%m-%d %H:%M:%S",
                                         )

    # consoleHandler.setFormatter(simple_formatter)
    fileHandler.setFormatter(simple_formatter)

    # logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)

    return logger


class MyTrainCallback(TrainerCallback):
    def __init__(self, mylogger):
        self.mylogger = mylogger

    def on_epoch_begin(self, args, state, control, **kwargs):
        history = state.log_history

        if len(history) > 0:
            loghis = history[-1]
            self.mylogger.info(
                'epoch: ' + str(loghis['epoch']) + '  learning_rate: ' + str(
                    loghis['learning_rate']) + '  loss: ' + str(
                    loghis['loss']))

    def on_train_end(self, args, state, control, **kwargs):
        history = state.log_history
        loghis = history[-2]
        self.mylogger.info(
            'epoch: ' + str(loghis['epoch']) + '  learning_rate: ' + str(
                loghis['learning_rate']) + '  loss: ' + str(
                loghis['loss']))

        loghis = history[-1]
        self.mylogger.info("耗时：" + format_time(loghis['train_runtime']))


def train():
    mycheckpoint = "models/gpt4ner2"
    if not os.path.exists(mycheckpoint):
        os.makedirs(mycheckpoint)

    logger = create_logger(name="train_log",
                           filename=mycheckpoint + "/gpt4ner2.log")
    logger.info(
        "------------------------------------------------------------------------------------------------------------------------------------------")

    logger.info(
        "基于QLora 4-bit微调GPT2实现ner任务，超参数有--num_samples %d --max_length %d --num_epochs %d --lr %e --batch_size %d --accum_steps %d" % (
            pargs.num_samples, pargs.max_length, pargs.num_epochs, pargs.lr, pargs.batch_size, pargs.accum_steps))

    logger.info("开始创建分词器...")

    pretrained_checkpoint = "uer/gpt2-xlarge-chinese-cluecorpussmall"
    tokenizer = BertTokenizer.from_pretrained(pretrained_checkpoint,
                                              )

    logger.info("开始读取数据...")
    dataset = MyDataset(tokenizer)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
    )

    logger.info("开始创建模型...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_checkpoint,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )

    model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    logger.info(lora_config)

    model = get_peft_model(model, lora_config)
    logger.info(lora_config)

    model.print_trainable_parameters()

    # 计算参数量和 trainable 参数量
    trainable_param_count, param_count = model.get_nb_trainable_parameters()
    logger.info("trainable params: %d || all params: %d  || trainable%%: %f" % (
        trainable_param_count, param_count, (100.0 * trainable_param_count) / param_count))

    model.to(device)

    logger.info("开始设置训练参数TrainingArguments...")
    # 半精度eps重新设置，否则会导致loss上溢出或下溢出
    training_args = TrainingArguments(
        output_dir=mycheckpoint,
        overwrite_output_dir=True,
        logging_strategy="epoch",
        per_device_train_batch_size=pargs.batch_size,
        gradient_accumulation_steps=pargs.accum_steps,
        num_train_epochs=pargs.num_epochs,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        dataloader_drop_last=False,
        learning_rate=pargs.lr,
        weight_decay=1e-2,
        adam_epsilon=1e-4,
        max_grad_norm=1.0,
        save_strategy="no",
        optim="paged_adamw_8bit",
        fp16=True,
    )

    mytraincallback = MyTrainCallback(logger)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset[:pargs.num_samples],
        data_collator=data_collator,
        callbacks=[mytraincallback],
    )

    logger.info("开始训练...")
    model.config.use_cache = False

    trainer.train()

    logger.info("保存模型")

    trainer.model.save_pretrained(mycheckpoint)


def generator(checkpoint):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    lora_config = LoraConfig.from_pretrained(checkpoint)

    tokenizer = BertTokenizer.from_pretrained(lora_config.base_model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(
        lora_config.base_model_name_or_path,
        quantization_config=bnb_config,
        load_in_8bit=True,
        torch_dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(model, checkpoint)

    model.half()
    model.eval()

    prompt_prefix = "你是一个命名实体识别模型，请用Python字典的形式列出后面文本中类别为\"治疗\"、\"身体部位\"、\"症状和体征\"、\"检查和检验\"和 \"疾病和诊断\"的实体的位置和类别，实体与类别之间用\"_\"连接。文本："
    prompt_suffix = "实体有："

    intent = True

    while intent:
        query = input("\n文本：")
        if query == '':
            intent = False
            continue

        ask = prompt_prefix + query + prompt_suffix

        askprompt = '[CLS]' + ask
        res = tokenizer(askprompt,
                        add_special_tokens=False,
                        )

        inputs = torch.tensor([res["input_ids"]]).to(device)
        attention_mask = torch.tensor([res["attention_mask"]]).to(device)

        generated_ids = model.generate(
            inputs=inputs,
            attention_mask=attention_mask,
            min_length=3,
            max_new_tokens=pargs.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=3.5,
            length_penalty=0.5,
            early_stopping=True,
            num_beams=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.95,
        )

        decoded_pres = tokenizer.batch_decode(generated_ids,
                                              skip_special_tokens=True,
                                              )[0].replace(" ", "")
        print(decoded_pres)

        # entities = decoded_pres.split("实体有：")[1].strip().split("。")[0].split("；")
        # print("\n实体有：")
        # for i, entity in enumerate(entities):
        #     print(str(i + 1) + ". " + entity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default="train", type=str, required=True)
    parser.add_argument("--num_samples", default=5000, type=int)
    parser.add_argument("--max_length", default=600, type=int)
    parser.add_argument("--max_new_tokens", default=300, type=int)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--accum_steps", default=4, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)

    pargs = parser.parse_args()

    if pargs.mode == "train":
        train()
    elif pargs.mode == "infer":
        generator("models/gpt4ner2")
