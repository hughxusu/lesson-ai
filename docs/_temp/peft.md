## PEFT实践

[Hugging Face PEFT库](https://huggingface.co/docs/peft/index)是一个为大型预训练模型提供多种高效微调方法的Python库，其中包括了Adapter Tuning、Prefix Tuning、Prompt Tuning和LoRA等方法。安装PEFT库

```shell
pip install peft
```

### 微调实践入门

以[facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b)模型为例进行模型微调

```python
from transformers import GPT2Tokenizer, OPTForCausalLM

model_id = "facebook/opt-6.7b"

model = OPTForCausalLM.from_pretrained(model_id, load_in_8bit=True)
tokenizer = GPT2Tokenizer.from_pretrained(model_id)
```

使用原始模型生成内容

```python
text = "Python is the best programming language."

inputs = tokenizer(text, return_tensors="pt").to(0)  
outputs = model.generate(**inputs, max_length=50)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Text:", generated_text)
```

模型预处理

* 将所有模块转换为全精度FP32以保证稳定性
* 为输入嵌入层添加一个`forward_hook`，以启用输入隐藏状态的梯度计算
* 启用梯度检查点以实现更高效的内存训练

```python
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)
```

查看显存占用和模型结构

```python
memory_footprint_bytes = model.get_memory_footprint()
memory_footprint_mib = memory_footprint_bytes / (1024 ** 3)  # 转换为 GB

print(f"{memory_footprint_mib:.2f}GB")
print(model)
```

配置LoRA参数

```python
from peft import LoraConfig, get_peft_model

# 创建一个LoraConfig对象，用于设置LoRA的配置参数
config = LoraConfig(
    r=8,            # LoRA的秩，影响LoRA矩阵的大小
    lora_alpha=32,  # LoRA适应的比例因子
    # 指定将LoRA应用到的模型模块，通常是attention和全连接层的投影
    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out"],
    lora_dropout=0.05,     # 在LoRA模块中使用的dropout率
    bias="none",           # 设置bias的使用方式，这里没有使用bias
    task_type="CAUSAL_LM"  # 任务类型，这里设置为因果(自回归）语言模型
)

# 使用get_peft_model函数和给定的配置来获取一个PEFT模型
model = get_peft_model(model, config)
model.print_trainable_parameters()
print(model)
```

使用[Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes)数据集进行模型微调

```python
from datasets import load_dataset

dataset = load_dataset("Abirate/english_quotes")
print(dataset["train"])
```

打印数据格式

```python
from datasets import ClassLabel, Sequence
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(lambda x: [typ.feature.names[i] for i in x])
    display(HTML(df.to_html()))
    
show_random_elements(dataset["train"])
```

将分词转换为训练数据

```python
from transformers import DataCollatorForLanguageModeling

tokenized_dataset = dataset.map(lambda samples: tokenizer(samples["quote"]), batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
```

* `data_collator`数据整理器，训练过程中动态生成批数据，`mlm=False`模型的任务是预测下一个token。

设置训练参数

```python
from transformers import TrainingArguments, Trainer

save_path = "./data/hf/models/opt-6.7b-lora"

training_args = TrainingArguments(
    output_dir=save_path,           # 指定模型输出和保存的目录
    per_device_train_batch_size=4,  # 每个设备上的训练批量大小
    learning_rate=2e-4,             # 学习率
    fp16=True,                      # 启用混合精度训练，可以提高训练速度，同时减少内存使用
    logging_steps=20,               # 指定日志记录的步长，用于跟踪训练进度
    max_steps=100,                  # 最大训练步长
    num_train_epochs=1              # 训练的总轮数
)
```

配置训练器

```python
trainer = Trainer(
    model=model,                               # 指定训练时使用的模型
    train_dataset=tokenized_dataset["train"],  # 指定训练数据集
    args=training_args,
    data_collator=data_collator,
)

model.use_cache = False # 禁用模型的自回归生成缓存
```

训练模型

```python
trainer.train()
```

保存训练结果

```python
model.save_pretrained(save_path)
```

测试训练结果

```python
lora_model = trainer.model

inputs = tokenizer(text, return_tensors="pt").to(0)
out = lora_model.generate(**inputs, max_length=50)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## QLoRA微调

使用QLoRA方法微调[zai-org/chatglm3-6b](https://huggingface.co/zai-org/chatglm3-6b)模型，训练数据为[HasturOfficial/adgen](https://huggingface.co/datasets/HasturOfficial/adgen)。

定义训练参数

```python
model_id = 'zai-org/chatglm3-6b'          # 模型ID
data_id = 'HasturOfficial/adgen'          # 训练数据ID
eval_data_path = None                     # 验证数据路径，如果没有则设置为None
seed = 8                                  # 随机种子
max_input_length = 512                    # 输入的最大长度
max_output_length = 1536                  # 输出的最大长度
lora_rank = 4                             # LoRA秩
lora_alpha = 32                           # LoRA alpha值
lora_dropout = 0.05                       # LoRA Dropout率
resume_from_checkpoint = None             # 如果从checkpoint恢复训练，指定路径
prompt_text = ''                          # 所有数据前的指令文本
compute_dtype = 'fp32'                    # 计算数据类型（fp32, fp16, bf16）
```

下载训练数据

```python
from datasets import load_dataset

dataset = load_dataset(data_id)
print(dataset)
```

打印训练数据信息

```python
show_random_elements(dataset["train"], num_examples=3)
```

加载分词器

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
```

数据预处理

```python
def tokenize_func(example, tokenizer, ignore_label_id=-100):
    question = prompt_text + example['content']
    if example.get('input', None) and example['input'].strip():
        question += f'\n{example["input"]}'

    answer = example['summary']

    q_ids = tokenizer.encode(text=question, add_special_tokens=False)
    a_ids = tokenizer.encode(text=answer, add_special_tokens=False)

    if len(q_ids) > max_input_length - 2:  
        q_ids = q_ids[:max_input_length - 2]
    if len(a_ids) > max_output_length - 1:  
        a_ids = a_ids[:max_output_length - 1]

    input_ids = tokenizer.build_inputs_with_special_tokens(q_ids, a_ids)
    question_length = len(q_ids) + 2  

    labels = [ignore_label_id] * question_length + input_ids[question_length:]
    return {'input_ids': input_ids, 'labels': labels}
```

* 解析`content`和`summary`的内容。
* 对输入数据进行截断处理。

对原始数据进行分词处理

```python
column_names = dataset['train'].column_names
tokenized_dataset = dataset['train'].map(
    lambda example: tokenize_func(example, tokenizer),
    batched=False, 
    remove_columns=column_names
)
show_random_elements(tokenized_dataset, num_examples=1)
```

将训练数据打乱

```python
tokenized_dataset = tokenized_dataset.shuffle(seed=seed)
tokenized_dataset = tokenized_dataset.flatten_indices()
```

对数据进行批量处理

```python
import torch
from typing import List, Dict

class DataCollatorForChatGLM:
    def __init__(self, pad_token_id: int, max_length: int = 2048, ignore_label_id: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_label_id = ignore_label_id
        self.max_length = max_length

    def __call__(self, batch_data: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        len_list = [len(d['input_ids']) for d in batch_data]
        batch_max_len = max(len_list) 

        input_ids, labels = [], []
        for len_of_d, d in sorted(zip(len_list, batch_data), key=lambda x: -x[0]):
            pad_len = batch_max_len - len_of_d 
            ids = d['input_ids'] + [self.pad_token_id] * pad_len
            label = d['labels'] + [self.ignore_label_id] * pad_len
            if batch_max_len > self.max_length:
                ids = ids[:self.max_length]
                label = label[:self.max_length]
            input_ids.append(torch.LongTensor(ids))
            labels.append(torch.LongTensor(label))

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

        return {'input_ids': input_ids, 'labels': labels}

data_collator = DataCollatorForChatGLM(pad_token_id=tokenizer.pad_token_id)
```

* 将一个批次的数据重新组合

```
[
	{'input_ids': [1, 2, 3], 'labels': [10, 20, 30]},  
	{'input_ids': [4, 5], 'labels': [40, 50]}
]

==>

{
	'input_ids': tensor([[1, 2, 3], [4, 5, pad]]),  
	'labels': tensor([[10, 20, 30], [40, 50, -100]])
}
```

加载模型，配置量化参数

```python
from transformers import AutoModel, BitsAndBytesConfig

q_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type='nf4', 
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModel.from_pretrained(
    model_id, quantization_config=q_config, device_map='auto', trust_remote_code=True,
)
```

打印当前模型占用的显存

```python
memory_footprint_bytes = model.get_memory_footprint()
memory_footprint_mib = memory_footprint_bytes / (1024 ** 2)
print(f"{memory_footprint_mib:.2f}MiB")
```

模型预处理

```python
from peft import prepare_model_for_kbit_training

kbit_model = prepare_model_for_kbit_training(model)
```

设置低秩适配矩阵的位置，在PEFT库的[constants.py](https://github.com/huggingface/peft/blob/main/src/peft/utils/constants.py)文件中，定义了不同大模型上的微调适配模块。

```python
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['chatglm']
print(target_modules)
```

* `chatglm`对 `query`、`key`、`value` 的投影矩阵都添加了适配参数。

设置LoRA参数

```python
from peft import TaskType, LoraConfig

lora_config = LoraConfig(
    target_modules=target_modules,
    r=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias='none',
    inference_mode=False,
    task_type=TaskType.CAUSAL_LM
)
```

打印微调参数的比例

```python
from peft import get_peft_model

qlora_model = get_peft_model(kbit_model, lora_config)
qlora_model.print_trainable_parameters()
```

设置训练参数

```python
from transformers import TrainingArguments

save_path = './data/hf/models/chatglm3-6b-qlora'

training_args = TrainingArguments(
    output_dir=save_path,                              # 输出目录
    per_device_train_batch_size=6,                     # 每个设备的训练批量大小
    gradient_accumulation_steps=8,                     # 梯度累积步数
    learning_rate=1e-3,                                # 学习率
    num_train_epochs=1,                                # 训练轮数
    lr_scheduler_type="linear",                        # 学习率调度器类型
    warmup_ratio=0.1,                                  # 预热比例
    logging_steps=100,                                 # 日志记录步数
    optim="adamw_torch",                               # 优化器类型
    fp16=True,                                         # 是否使用混合精度训练
)
```

创建训练器

```python
from transformers import Trainer

trainer = Trainer(
    model=qlora_model, args=training_args, train_dataset=tokenized_dataset, data_collator=data_collator
)
```

训练模型

```python
trainer.train()
```

