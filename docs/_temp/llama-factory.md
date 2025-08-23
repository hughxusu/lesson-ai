# Llama-Factory安装

| 依赖库           | 一般要求 (参考)                      | 严格版本限制 (参考)                             | 说明                                                     |
| :--------------- | :----------------------------------- | :---------------------------------------------- | :------------------------------------------------------- |
| **Transformers** | ≥4.41.218                            | ≥4.41.2, ≤4.48.3, !=4.46.*, !=4.47.*, !=4.48.01 | 核心依赖，版本兼容性要求高，不建议使用不符合要求的版本。 |
| **Python**       | ≥3.8, 推荐 3.10 或 3.11248           | -                                               |                                                          |
| **PyTorch**      | ≥1.13.1, 推荐 2.2.0+28               | 需与CUDA版本匹配26                              | 需根据你的CUDA版本选择合适的PyTorch。                    |
| **CUDA**         | 11.6, 推荐 12.1+28                   | -                                               | 通常通过PyTorch的版本间接指定。                          |
| **其他依赖**     | accelerate, peft, datasets, trl 等28 | -                                               | LLaMA Factory 通常会处理这些依赖的安装。                 |

使用腾讯云重点cloud studio安装Llama-Factory

```mermaid
graph LR

a(登录腾讯云)-->b(选择控制台)-->c(工具-cloud studio)-->d(AI模版-pytorch)
```

创建开发环境，升级torch环境

```python
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

* `torch`版本为2.7.1，`torchvision`和`torchaudio`的版本应用与之配合。
* cuda版本为12.8.

> [!warning]
>
> 删除文件夹下的所有文件，包括隐藏文件

clone Llama-Factory项目

```shell
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git .
```

* `.`表示clone到当前的根目录下，不创建子文件夹。
* `--depth 1`深度为1，只克隆最近的一次提交。

安装Llama-Factory

```python
pip install -e ".[torch,metrics]"
```

* `".[torch,metrics]"`中的`.`表示安装当前目录的内容；`[torch,metrics]`可选依赖组。
* `-e`将包以"开发模式"安装到当前环境中

验证是否安装成功

```shell
llamafactory-cli version 
```

查看命令安装路径

```shell
which llamafactory-cli
```

启动webui

```shell
llamafactory-cli webui
```

访问webui

```shell
https://jwbded.ap-singapore.cloudstudio.work/         # 如果浏览器访问cloud studio地址为
https://jwbded--7860.ap-singapore.cloudstudio.work/   # 访问端口--7860
```





