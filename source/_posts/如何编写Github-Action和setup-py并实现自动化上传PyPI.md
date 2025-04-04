---
title: 如何编写Github Action和setup.py并实现自动化上传PyPI
date: 2025-04-04 19:42:03
hide: false
description: 前些日子想给自己的代码做一个巨大的更新——上传PyPI平台。然而！来自懒狗的认知里，在维护代码的时候不想做一些重复性非常高的方法。众所周知，Github是一个自动化流程很高的平台，那么编写一套setup.py代码和Github Action流程就迫在眉睫。
categories: 
- [运维, 应用配置]
- [运维项目应用]
tags: [Github, 自动化, pip]
---

# 0 前言

鄙人平时会写点开源代码，并且个人维护了一个叫[IDDM: Integrated-Design-Diffusion-Model](https://github.com/chairc/Integrated-Design-Diffusion-Model)的开源仓库。作为一个忙里偷闲的懒狗程序员，前些日子想给自己的代码做一个巨大的更新——上传PyPI平台（内心OS：这样就可以执行`pip install iddm`下载了）。

然而！来自懒狗的认知里，在维护代码的时候不想做一些重复性非常高的方法。众所周知，Github是一个自动化流程很高的平台，那么编写一套`setup.py`代码和`Github Action`流程就迫在眉睫。
**文章最新更新日期**：2025年3月11日15:37:42

# 1 准备工作

## 1.1 标准文件格式准备

想要打包成PyPI的格式，首先应该将你的项目文件重构到一个目标文件夹中。以IDDM为例，在`1.1.8`版本之前，所有的`py`文件均在主目录下，如图所示：

![1.1.8版本之前的目录结构](/37a977d1e7b94489952a210344955567.png)

鄙人在实践中发现这是一种打包上传`PyPI`很蠢的结构，为什么会这么说呢？在执行`python -m build`的时候，这玩意生成的`whl`文件不会给你创建文件夹，**而是直接打包到整体目录下**，这样你安装`pip`的时候，环境会直接将所有文件夹安装到`site-packages`下面（就很蠢，会让你导包的时候找不到你想导入的东西），如图所示：

![错误的打包版本](/3ef4ecc5d712464584419c30687ea666.png)

正常来说，我们应该重构一个文件夹作为打包的入口文件夹，例如IDDM的`1.1.8`版本，如图3所示：

![1.1.8版本创建打包主文件夹](/fcb860e4648e460598a6372713436fee.png)

在需要打包的文件夹中创建`__init__.py`文件（所有文件夹都需要有`__init__.py`文件，不然不识别），该文件的作用是告诉打包程序，这是需要打包的文件夹，文件夹结构如图所示：

![__init__.py文件夹结构](/f4d3046048ef47e3bae39e18c5f097c7.png)

## 1.2 环境准备

针对于打包，我么需要准备`build`这个PyPI包，安装方法如下：。

```bash
pip install build
```

检查一下自己项目的虚拟环境或物理环境。这里我们以IDDM项目环境为例，打包项目所需的前置环境如下：

```bash
coloredlogs==15.0.1
gradio==5.0.0
matplotlib==3.7.1
numpy==1.25.0
Pillow==10.3.0
Requests==2.32.0
scikit-image==0.22.0
torch_summary==1.4.5
tqdm==4.66.3
pytorch_fid==0.3.0
fastapi==0.115.6
tensorboardX==2.6.1

# 如果你想下载GPU版本请使用：pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
# 想了解更多信息请访问：https://pytorch.org/get-started/previous-versions/#linux-and-windows-25
# 更多版本请访问：https://pytorch.org/get-started/previous-versions
# 需要注意torch版本 >= 1.9.0
torch>=1.9.0 # 更多信息：https://pytorch.org/get-started/locally/ （推荐）
torchvision>=0.10.0 # 更多信息：https://pytorch.org/get-started/locally/ （推荐）
```

# 2 代码编写

## 2.1 setup.py文件编写

```python
from setuptools import setup, find_packages

def get_long_description():
    with open(file="README.md", mode="r", encoding="utf-8") as f:
        long_description = f.read()
    return long_description

if __name__ == "__main__":
	# 安装需要的包
    package_list = [
        "coloredlogs==15.0.1",
        "gradio==5.0.0",
        "matplotlib==3.7.1",
        "numpy==1.25.0",
        "Pillow==10.3.0",
        "Requests==2.32.0",
        "scikit-image==0.22.0",
        "torch_summary==1.4.5",
        "tqdm==4.66.3",
        "pytorch_fid==0.3.0",
        "fastapi==0.115.6",
        "tensorboardX==2.6.1",
        "torch>=1.9.0",
        "torchvision>=0.10.0"
    ]
    # 定义setup
    setup(
        name="iddm", # 发布包名称
        version="1.1.8-b3", # 版本号，不能重复
        packages=find_packages(), # 寻找你要打包的文件
        python_requires=">=3.8", # Python要求
        install_requires=package_list, #  安装需要的包
        license="Apache-2.0", # 证书
        description="IDDM: Integrated Design Diffusion Model", # 包描述
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        author="chairc", # 作者
        author_email="chenyu1998424@gmail.com", # 作者邮箱
        url="https://github.com/chairc/Integrated-Design-Diffusion-Model", # 项目地址
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
        ],
        project_urls={
            "Documentation": "https://github.com/chairc/Integrated-Design-Diffusion-Model/blob/main/README.md",
            "Source": "https://github.com/chairc/Integrated-Design-Diffusion-Model",
            "Tracker": "https://github.com/chairc/Integrated-Design-Diffusion-Model/issues",
        },
    )

```

## 2.2 测试setup.py文件

**别写了代码不进行测试**！不然你就会像鄙人一样在之后的工作里成为小丑！！

执行以下命令进行测试，命令如下：

```bash
cd Integrated-Design-Diffusion-Model
python -m build
```

等待指令执行，出现`Successfully built iddm-1.1.8b3.tar.gz and iddm-1.1.8b3-py3-none-any.whl`，如下：

```bash
* Creating isolated environment: venv+pip...
* Installing packages in isolated environment:
  - setuptools >= 40.8.0
* Getting build dependencies for sdist...
...
中间忽略
...
* Building wheel from sdist
* Creating isolated environment: venv+pip...
* Installing packages in isolated environment:
  - setuptools >= 40.8.0
* Getting build dependencies for wheel...
...
中间忽略
...
adding 'tools/__init__.py'
adding 'iddm-1.1.8b3.dist-info/LICENSE'
adding 'iddm-1.1.8b3.dist-info/METADATA'
adding 'iddm-1.1.8b3.dist-info/WHEEL'
adding 'iddm-1.1.8b3.dist-info/top_level.txt'
adding 'iddm-1.1.8b3.dist-info/RECORD'
removing build\bdist.win-amd64\wheel
Successfully built iddm-1.1.8b3.tar.gz and iddm-1.1.8b3-py3-none-any.whl # 出现这个就成功了
```

现在我们再次查看`whl`文件的格式，如下：

![正确的whl文件格式](/f7667c688259439fa64f932419828c92.png)

此时我们执行以下指令进行本地安装，如下：

```bash
cd Integrated-Design-Diffusion-Model
pip install ./dist/iddm-1.1.8b3-py3-none-any.whl
```

使用一个小demo进行测试，如下：

![测试IDDM的demo](/e4a5511406bc4fc29aa305b6e5f29a08.png)

很好，我们测试本地发布+安装`iddm`成功，说明之后发布到`PyPI`上的包也是基本没有问题的！

上传代码至Github仓库

```bash
git add .
git commit -m "提交setup.py"
git push origin main # 可以直接推仓库，但我建议先push到dev分支，再提交PR进行合并
```

## 2.3 Github Action编写
这一步相对于简单，打开Github仓库页面，点击`Actions`：

![打开Actions](/8de7968ea71b447a9ad31fdab601be6c.png)

点击`New workflow`，创建新的工作流，如下：

![创建新的工作流](/bdb1454343d749888df326b425b13819.png)

选择`Publish Python Package`中的`Configure`

![创建Pip](/9ad3843e182c4e74903063e11bd34dc7.png)

网上有很多版本都要去`PyPI`申请`token`，因为Github提交`PyPI`都已经更新好几版本了，**Github官方和PyPI官方都给我们写好了**，我们直接commit这个文件就好了，如下：

![编写pip工作流](/9dd78e8472c24977810317613d7f588f.png)

# 3 上传发布

## 3.1 绑定PyPI

工作流提交完毕后，我们需要去`PyPI`创建一个账号，**和现在这个项目进行绑定**，如下：

![publishing](/d346cc9b0cac42f58ee54b7995db77dd.png)
![添加项目](/3556759887614b00a2891ba9be2f8fe3.png)

## 3.2 发布版本

此时，我们的项目就已经添加到PyPI中了，在每次发布`Release`包的时候Github Action会自动`build`我们的程序并上传`PyPI`仓库，例如我们发布一个`iddm v1.1.8-beta.3`：

![自动打包](/50820d0919344ac28c0c225599999266.png)
![打包明细](/38703bab2c4f4f9c813e9c308055b50d.png)

鄙人的`iddm`包就发布到了`PyPI`中了

![PyPI官方](/eb6f66b34d3b4f0f8703aaa11b54448e.png)

# 4 下载运行

## 4.1 pip下载

试着下载一下pip包
```bash
pip install iddm
```

运行结果如下：

![pip安装](/1d85f8a81e6b491a96949eee3cacc96a.png)



