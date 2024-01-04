'''
    基于chatGLM3完成NER任务 chatglm3-6B-base
    作为没有经过人类意图对齐的模型，ChatGLM3-6B-Base 不能用于多轮对话。但是可以进行文本续写。
    Qlora -4bit + 半精度float16
'''

from torch.utils.data import Dataset, DataLoader, random_split
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import get_scheduler, TrainingArguments, Trainer
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
        prompt_prefix = "你现在是一个命名实体识别模型，请你帮我找出类别为\"治疗\"、\"身体部位\"、\"症状和体征\"、\"检查和检验\"和 \"疾病和诊断\"的实体，实体与类别之间用\"_\"连接，实体之间用\\n分割。文本："

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
            text = data["context"][:args.max_length // 2]
            seqlen = len(text)

            ask = prompt_prefix + text
            # print(ask)
            ask = tokenizer.build_chat_input(ask, history=[], role='user')

            answer = '\n'
            for se, label in data['span_posLabel'].items():
                startid, endid = se.split(";")
                if int(endid) < seqlen:
                    entity = text[int(startid):int(endid) + 1]
                    answer += entity + "_" + zy[label] + "\n"
            # print(answer)

            answer = tokenizer(answer, add_special_tokens=False)

            input_ids = ask["input_ids"][0].numpy().tolist() + answer["input_ids"] + [tokenizer.eos_token_id]
            attention_mask = ask['attention_mask'][0].numpy().tolist() + answer['attention_mask'] + [1]

            self.nerdata.append({'input_ids': input_ids,
                                 'attention_mask': attention_mask})
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


def train():
    mycheckpoint = "models/chatglm1"
    if not os.path.exists(mycheckpoint):
        os.makedirs(mycheckpoint)

    logger = create_logger(name="train_log", filename=mycheckpoint + "/llama2.log")
    logger.info(
        "------------------------------------------------------------------------------------------------------------------------------------------")

    logger.info(
        "基于QLora 4-bit微调chatglm3，超参数有--num_samples %d --max_length %d --num_epochs %d --lr %e --batch_size %d" % (
            args.num_samples, args.max_length, args.num_epochs, args.lr, args.batch_size))

    logger.info("开始创建分词器...")

    pretrained_checkpoint = "uer/ZhipuAI/chatglm3-6b-base"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint,
                                              trust_remote_code=True,
                                              )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("开始读取数据...")
    dataset = MyDataset(tokenizer)
    dataset_train, dataset_eval = random_split(dataset, [0.9, 0.1])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    data_train_loader = DataLoader(dataset_train,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   collate_fn=data_collator,
                                   drop_last=False)
    data_eval_loader = DataLoader(dataset_eval,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  collate_fn=data_collator,
                                  drop_last=False)

    logger.info("开始创建模型...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_checkpoint,
        quantization_config=bnb_config,
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

    model.half()

    # 计算参数量和 trainable 参数量
    param_count = sum([p.numel() for p in model.parameters()])
    trainable_param_count = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info("trainable params: %d || all params: %d  || trainable%%: %f" % (
        trainable_param_count, param_count, (100.0 * trainable_param_count) / param_count))

    # 半精度eps重新设置，否则会导致loss上溢出或下溢出
    optimizer = AdamW8bit(model.parameters(),
                          lr=args.lr,
                          weight_decay=0.0,
                          )

    num_steps_per_epoch = len(data_train_loader)
    num_training_steps = args.num_epochs * num_steps_per_epoch

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_training_steps // 10,
    )

    model.to(device)
    model.save_pretrained(mycheckpoint)
    logger.info("训练前保存模型")

    logger.info("开始训练...")

    for epoch in range(args.num_epochs):
        model.train()

        total_loss = 0.

        start_time = time.time()

        for step, batch in enumerate(data_train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss

            total_loss += loss.item()

            loss.backward()

            clip_grad_norm_(
                parameters=model.parameters(),
                max_norm=1.0,
                norm_type=2,
            )

            optimizer.step()
            lr_scheduler.step()

            end_time = time.time()

            run_time = end_time - start_time
            time_per_step = run_time / (step + 1)
            steps_per_second = (step + 1) / run_time

            rest_time = time_per_step * (num_steps_per_epoch - step - 1)

            print("\repoch: %d/%d  batch: %d/%d  %d%% [%s<%s, %.1fit/s]  loss: %f" % (
                epoch + 1, args.num_epochs, step + 1, num_steps_per_epoch,
                int(100 * (step + 1.) / num_steps_per_epoch),
                format_time(run_time), format_time(rest_time), steps_per_second,
                loss.item()), end="")

        avg_train_loss = total_loss / num_steps_per_epoch
        print("  avg_loss: %f  val_acc: " % (avg_train_loss), end="")

        model.eval()

        rightlabel = 0.
        total_label = 0.

        for batch in data_eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                )
            logits = outputs.logits
            logits_shift = logits[:, :-1]
            preds = torch.argmax(logits_shift, dim=-1)
            labels_shift = labels[:, 1:]
            keeplabels = torch.logical_not(torch.eq(labels_shift, -100))

            rightlabel += torch.sum(torch.eq(preds, labels_shift)).item()
            total_label += torch.sum(keeplabels).item()

        eval_acc = rightlabel / total_label
        print("%f\n" % (eval_acc))

        logger.info("epoch: %d/%d  avg_loss: %f  val_acc: %f" % (epoch + 1, args.num_epochs, avg_train_loss, eval_acc))

        model.save_pretrained(mycheckpoint)
        logger.info("保存模型")

    logger.info(
        "------------------------------------------------------------------------------------------------------------------------------------------")


def trainnew():
    mycheckpoint = "models/chatglm4ner"
    if not os.path.exists(mycheckpoint):
        os.makedirs(mycheckpoint)

    logger = create_logger(name="train_log", filename=mycheckpoint + "/chatglm4ner.log")
    logger.info(
        "------------------------------------------------------------------------------------------------------------------------------------------")

    logger.info(
        "基于QLora 4-bit微调chatglm3实现ner任务，超参数有--num_samples %d --max_length %d --num_epochs %d --lr %e --batch_size %d --accum_steps %d" % (
            args.num_samples, args.max_length, args.num_epochs, args.lr, args.batch_size, args.accum_steps))

    logger.info("开始创建分词器...")

    pretrained_checkpoint = "uer/ZhipuAI/chatglm3-6b-base"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint,
                                              trust_remote_code=True,
                                              )

    logger.info("开始读取数据...")
    dataset = MyDataset(tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    logger.info("开始创建模型...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_checkpoint,
        trust_remote_code=True,
        quantization_config=bnb_config,
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
    # param_count = sum([p.numel() for p in model.parameters()])
    # trainable_param_count = sum([p.numel() for p in model.parameters() if p.requires_grad])
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
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum_steps,
        num_train_epochs=args.num_epochs,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        dataloader_drop_last=False,
        learning_rate=args.lr,
        weight_decay=1e-2,
        adam_epsilon=1e-4,
        max_grad_norm=1.0,
        save_strategy="no",
        optim="paged_adamw_8bit",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset[:args.num_samples],
        data_collator=data_collator,
    )

    logger.info("开始训练...")
    model.config.use_cache = False

    trainer.train()

    for loghis in trainer.state.log_history[:args.num_epochs]:
        logger.info(
            'epoch: ' + str(loghis['epoch']) + '  learning_rate: ' + str(loghis['learning_rate']) + '  loss: ' + str(
                loghis['loss']))

    logger.info("耗时：" + format_time(trainer.state.log_history[-1]['train_runtime']))

    logger.info("保存模型")

    trainer.model.save_pretrained(mycheckpoint)

    logger.info(
        "------------------------------------------------------------------------------------------------------------------------------------------")


def generator(checkpoint, text):
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
    )

    tokenizer = BertTokenizer.from_pretrained(checkpoint)

    indexed_tokens = tokenizer.encode(text)

    tokens_tensor = torch.tensor([indexed_tokens])

    if args.cuda:
        model.to(device)
        tokens_tensor = tokens_tensor.to(device)

    model.eval()
    start_time = time.time()
    past_key_values = None

    for i in range(args.max_new_tokens):
        # print(tokens_tensor.shape)
        with torch.no_grad():
            output = model(tokens_tensor,
                           past_key_values=past_key_values,
                           )

        past_key_values = output.past_key_values
        # print(len(past_key_values))
        # print(past_key_values[0][0].shape)

        token = torch.argmax(output.logits[..., -1, :])

        indexed_tokens += [token.tolist()]

        tokens_tensor = token.unsqueeze(0)

        sequence = tokenizer.decode(indexed_tokens,
                                    skip_special_tokens=True,
                                    )
        # print("答：" + sequence.replace(" ", ""))
    end_time = time.time()
    print("\n耗时 " + format_time(end_time - start_time))

    lth = len(text)
    print("\n问：" + text)
    print("答：" + sequence.replace(" ", "")[lth:])


def inference_pipeline(checkpoint, text):
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
    )
    tokenizer = BertTokenizer.from_pretrained(checkpoint)

    prompt = text

    mydevice = "cpu"
    if args.cuda:
        model.to(device)
        mydevice = device

    model.eval()

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=mydevice
    )
    # generator = TextGenerationPipeline(
    #     model=model,
    #     tokenizer=tokenizer,
    #     device=device
    # )

    start_time = time.time()

    sequence = generator(
        prompt,
        min_length=3,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=3.5,
        length_penalty=2.5,
        early_stopping=True,
        num_beams=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )[0]['generated_text']

    end_time = time.time()

    print("\n耗时 " + format_time(end_time - start_time))

    lth = len(text)
    print("\n问：" + text)
    print("答：" + sequence.replace(" ", "")[lth:])


def generatornew(checkpoint):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig.from_pretrained(checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(lora_config.base_model_name_or_path,
                                              trust_remote_code=True,
                                              )

    model = AutoModelForCausalLM.from_pretrained(
        lora_config.base_model_name_or_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        load_in_8bit=True,
        torch_dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(model, checkpoint)

    model.half()
    model.eval()

    prompt_prefix = "你现在是一个命名实体识别模型，请你帮我找出类别为\"治疗\"、\"身体部位\"、\"症状和体征\"、\"检查和检验\"和 \"疾病和诊断\"的实体，实体与类别之间用\"_\"连接，实体之间用\\n分割。文本："
    intent = True

    while intent:
        query = input("\n文本：")
        if query == '':
            intent = False
            continue

        prompt = query

        # 方法一：直接使用模型的chat函数
        output = model.chat(tokenizer,
                            prompt_prefix + prompt,
                            history=[])
        entities = output[0].strip()

        print("\n实体有：\n")
        for entity in entities.split("\n"):
            print(" " + entity)

        # 方法二：使用generate函数
        res = tokenizer.build_chat_input(prompt_prefix + prompt, history=[], role="user")

        inputs = res["input_ids"].cuda()
        attention_mask = res["attention_mask"].cuda()

        generated_ids = model.generate(
            inputs=inputs,
            attention_mask=attention_mask,
            min_length=3,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=3.5,
            length_penalty=2.5,
            early_stopping=True,
            num_beams=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )

        decoded_pres = tokenizer.batch_decode(generated_ids,
                                              skip_special_tokens=True,
                                              )[0]
        prefix = "[gMask]sop<|user|> \n " + prompt_prefix + prompt + "<|assistant|>\n "
        entities = decoded_pres[len(prefix):].strip()

        print("\n实体有：\n")
        for entity in entities.split("\n"):
            print(" " + entity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default="train", type=str, required=True)
    parser.add_argument("--num_samples", default=5000, type=int)
    parser.add_argument("--max_length", default=600, type=int)
    parser.add_argument("--max_new_tokens", default=8192, type=int)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--accum_steps", default=4, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)

    args = parser.parse_args()

    if args.mode == "train":
        # train()
        trainnew()
    elif args.mode == "infer":
        # generator("models/chatglm4ner", args.text)
        generatornew("models/chatglm4ner")
        # inference_pipeline("models/chatglm4ner", args.text)
