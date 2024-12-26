# Installation

使用脚本命令一键下载（最为简单，不需要梯子）：

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

[Linux系统离线内网安装教程（CSDN)](https://blog.csdn.net/u010197332/article/details/137604798)
# Frequent Use Command

ollama的设计非常简洁高效，其中命令也是很简单的，下面记录一下常用的命令：

```bash
Usage:
  ollama [flags]
  ollama [command]

Available Commands:
  serve       Start ollama
  create      Create a model from a Modelfile
  show        Show information for a model
  run         Run a model
  pull        Pull a model from a registry
  push        Push a model to a registry
  list        List models
  ps          List running models
  cp          Copy a model
  rm          Remove a model
  help        Help about any command

Flags:
  -h, --help      help for ollama
  -v, --version   Show version information

Use "ollama [command] --help" for more information about a command.
```

## 修改默认的ollama模型路径

使用环境变量启动就可以了，下面就是一个例子。

```bash
nohup env OLLAMA_MODELS=/zhangshuhai/work/ollama-models/ ollama serve > ollama_serve.log 2>&1 &
```

## 下载模型

ollama下载模型也是非常的简单，首先可以到ollama的官网找到有没有对应的模型，然后再确认模型的型号就可以了。

[ollama supported models](https://ollama.com/library)

```bash
# 下载gemma2模型
nohup ollama pull gemma2:27b > ollama_download.log 2>&1 &
```

