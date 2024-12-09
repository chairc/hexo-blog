---
title: Pytorch复现经典扩散模型DDPM&DDIM&PLMS及分布式训练应用
date: 2024-11-23 14:04:57
description: 当前，生成式人工智能（AIGC）已被越来越广泛应用在工业、动漫业、设计业等诸多场景。我们都知道现阶段主流的生成模型如生成对抗网络（GAN）、自分编码器（VAE）、流模型（Flow-based Models）和扩散模型（Diffusion Models）。而扩散模型中还分为概率扩散模型，噪声条件评分网络和去噪概率模型。去噪概率模型中较为经典的就是DDPM。
categories: 
- [人工智能, 计算机视觉, 扩散模型]
- [Python项目应用]
tags: [扩散模型, DDPM, DDIM, Python]
---

# 0 前言
当前，生成式人工智能（AIGC）已被越来越广泛应用在工业、动漫业、设计业等诸多场景。我们都知道现阶段主流的生成模型如生成对抗网络（GAN）、自分编码器（VAE）、流模型（Flow-based Models）和扩散模型（Diffusion Models）。而扩散模型中还分为概率扩散模型，噪声条件评分网络和去噪概率模型。去噪概率模型中较为经典的就是DDPM（[**Denoising Diffusion Probabilistic Models**](https://arxiv.org/abs/2006.11239)）。
**本文章和GitHub仓库，如有问题请在此仓库提交issue，如果你认为我的项目有意思请给我点一颗⭐⭐⭐Star⭐⭐⭐吧。本文持续更新**，以GitHub为准嗷~
代码最新更新仓库：[https://github.com/chairc/Integrated-Design-Diffusion-Model](https://github.com/chairc/Integrated-Design-Diffusion-Model)
代码最新问题总结：[https://github.com/chairc/Integrated-Design-Diffusion-Model/issues/9](https://github.com/chairc/Integrated-Design-Diffusion-Model/issues/9)
访问不了问题总结可以点击CSDN链接：[https://blog.csdn.net/qq_43226466/article/details/143199474](https://blog.csdn.net/qq_43226466/article/details/143199474)
**文章最新更新日期：2024年10月24日09:19:40**

![image-20241123140628143](/image-20241123140628143.png)

# 1 简单原理
原理应该挺多的，具体参考这个[博客](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#forward-diffusion-process)
# 2 项目结构设计
**整体结构**
```yaml
Integrated Design Diffusion Model
├── config
│   ├── choices.py
│   └── version.py
├── datasets
│   └── dataset_demo
│       ├── class_1
│       ├── class_2
│       └── class_3
├── model
│   ├── modules
│   │   ├── activation.py
│   │   ├── attention.py
│   │   ├── block.py
│   │   ├── conv.py
│   │   ├── ema.py
│   │   └── module.py
│   ├── networks
│   │   ├── sr
│   │   │   └── srv1.py
│   │   ├── base.py
│   │   ├── cspdarkunet.py
│   │   └── unet.py
│   └── samples
│       ├── base.py
│       ├── ddim.py
│       ├── ddpm.py
│       └── plms.py
├── results
├── sr
│   ├── dataset.py
│   ├── demo.py
│   ├── interface.py
│   └── train.py
├── test
│   ├── noising_test
│   │   ├── landscape
│   │   └── noise
│   └── test_module.py
├── tools
│   ├── deploy.py
│   ├── generate.py
│   └── train.py
├── utils
│   ├── checkpoint.py
│   ├── initializer.py
│   ├── logger.py
│   ├── lr_scheduler.py
│   └── utils.py
├── webui
│   └──web.py
└── weight
```
`datasets`用于存放数据集文件，自动划分标签torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)。若数据集为：
        dataset_path/class_1/image_1.jpg
        dataset_path/class_1/image_2.jpg
        ...
        dataset_path/class_2/image_1.jpg
        dataset_path/class_2/image_2.jpg
        ...
其中，dataset_path是数据集所在的根目录，class_1, class_2等是数据集中的不同类别，每个类别下包含若干张图像文件。使用ImageFolder类可以方便地加载这种文件夹结构的图像数据集，并自动为每个图像分配相应的标签。可以通过传递dataset_path参数指定数据集所在的根目录，并通过其他可选参数进行图像预处理、标签转换等操作。  
`model`是存放模型的文件夹，UNet模型和采样器模型均在其中。
`results`是存放输出结果的文件夹，包括tensorboard日志、绘图和pt模型文件。
`test`是进行单元测试的文件夹。
`tools`是训练、生成等运行文件。
`utils`是各种工具文件，例如学习率、数据加载、图像绘制与保存等。
`weight`是存放预训练模型或训练较好的权重文件。

# 3 代码实现
代码仅为部分核心代码，代码完整版在[**github**](https://github.com/chairc/Industrial-Defect-Diffusion-Model)。
注：所有的代码及注释均为英文，以下代码为源代码的中文版本。
## 3.1 训练设计
扩散模型是一种对于显卡要求极高的模型。对于如何解决训练速度问题，我们使用了多机多卡训练方式，换言之就是分布式训练（未进行模型分布式，仅为数据分布式）。

### 3.1.1 分布式训练
分布式训练中，一般分为主线程和其他线程。对于我们这个应用，主线程的作用是广播整个训练中的参数、指数、资源、写入保存等操作。

```bash
    +------------------------+                     +-----------+
    |DistributedSampler      |                     |DataLoader |
    |                        |     2 indices       |           |
    |    Some strategy       +-------------------> |           |
    |                        |                     |           |
    |-------------+----------|                     |           |
                  ^                                |           |  4 data  +-------+
                  |                                |       -------------->+ train |
                1 | length                         |           |          +-------+
                  |                                |           |
    +-------------+----------+                     |           |
    |DataSet                 |                     |           |
    |        +---------+     |      3 Load         |           |
    |        |  Data   +-------------------------> |           |
    |        +---------+     |                     |           |
    |                        |                     |           |
    +------------------------+                     +-----------+
```
如上图所示，在数据集初始化方面，首先DistributedSampler类将数据集读入并返回一个Sampler类，该采样器是将数据加载限制为数据集子集，换言之就是显卡平分数据集。例如有1000张图，2张GPU，那么每张GPU中的500张图片作为数据集。

```python
dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=True, sampler=sampler)
```
在`train.py`启动线程，训练器会自动根据当前情况判断存在的设备数量，从而进行其它线程的开启。每个进程都会有一个rank，也就是当前的设备编号。具体实现如下：

```python
def main(args):
    """
    主方法
    :param args: 输入参数
    :return: None
    """
    if args.distributed:
        gpus = torch.cuda.device_count()
        mp.spawn(train, args=(args,), nprocs=gpus)
    else:
        train(args=args)
```

```python
def train(rank=None, args=None):
    """
    训练
    :param rank: GPU编号
    :param args: 输入参数
    :return: None
    """
    # 训练其它代码...
    # ...
    # 是否开启分布式训练
    if args.distributed and torch.cuda.device_count() > 1 and torch.cuda.is_available():
        distributed = True
        world_size = args.world_size
        # 设置地址和端口
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        # 进程总数等于显卡数量
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", rank=rank,
                                world_size=world_size)
        # 设置设备ID
        device = torch.device("cuda", rank)
        # 可能出现随机性错误，使用可减少cudnn随机性错误
        # torch.backends.cudnn.deterministic = True
        # 同步
        dist.barrier()
        # 如果分布式训练是第一块显卡，则保存模型标识位为真
        if dist.get_rank() != args.main_gpu:
            save_models = False
        logger.info(msg=f"[{device}]: Successfully Use distributed training.")
    # 训练其它代码...
    # ...
    if distributed:
        model = nn.parallel.DistributedDataParallel(module=model, device_ids=[device], find_unused_parameters=True)
    # 训练其它代码...
    # ...
    for epoch in range(start_epoch, args.epochs):
    	# 训练其它代码...
    	# ...
    	# 分布式在训练过程中进行同步
        if distributed:
            dist.barrier()
    # 训练其它代码...
    # ...
    if distributed:
        dist.destroy_process_group()
```
### 3.1.2 普通训练
均为正常训练方法，不开启多线程，这里不做过多讲解。
## 3.2 模型基类
因为DDPM与DDIM在方法实现上相近，DDIM相较于DDPM仅在采样过程有所改变（只是增加了跳步方案，如果跳步设置为noise_steps的个数即是DDPM），所以我们在这里定义一个扩散模型基类BaseDiffusion，方便DDPM和DDIM继承相同方法与变量。
```python
class BaseDiffusion:
    """
    扩散模型基类
    """
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cpu"):
        """
        扩散模型基类
        :param noise_steps: 噪声步长
        :param beta_start: β开始值
        :param beta_end: β结束值
        :param img_size: 图像大小
        :param device: 设备类型
        """
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        # 噪声步长
        self.beta = self.prepare_noise_schedule().to(self.device)
        # 公式α = 1 - β
        self.alpha = 1. - self.beta
        # 这里做α累加和操作
        self.alpha_hat = torch.cumprod(input=self.alpha, dim=0)

    def prepare_noise_schedule(self, schedule_name="linear"):
        """
        准备噪声schedule，可以自定义，可使用openai的schedule
        :param schedule_name: 方法名称，linear线性方法；cosine余弦方法
        :return: schedule
        """
        if schedule_name == "linear":
            # torch.linspace为指定的区间内生成一维张量，其中的值均匀分布
            return torch.linspace(start=self.beta_start, end=self.beta_end, steps=self.noise_steps)
        elif schedule_name == "cosine":
            def alpha_hat(t):
                """
                其参数t从0到1，并生成(1 - β)到扩散过程的该部分的累积乘积
                原式â计算公式为：α_hat(t) = f(t) / f(0)
                原式f(t)计算公式为：f(t) = cos(((t / (T + s)) / (1 + s)) · (π / 2))²
                在此函数中s = 0.008且f(0) = 1
                所以仅返回f(t)即可
                :param t: 时间
                :return: t时alpha_hat的值
                """
                return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

            # 要产生的beta的数量
            noise_steps = self.noise_steps
            # 使用的最大β值；使用小于1的值来防止出现奇点
            max_beta = 0.999
            # 创建一个分散给定alpha_hat(t)函数的β时间表，从t = [0,1]定义了（1 - β）的累积产物
            betas = []
            # 循环遍历
            for i in range(noise_steps):
                t1 = i / noise_steps
                t2 = (i + 1) / noise_steps
                # 计算β在t时刻的值，公式为：β(t) = min(1 - (α_hat(t) - α_hat(t-1)), 0.999)
                beta_t = min(1 - alpha_hat(t2) / alpha_hat(t1), max_beta)
                betas.append(beta_t)
            return torch.tensor(betas)

    def noise_images(self, x, time):
        """
        给图片增加噪声
        :param x: 输入图像信息
        :param time: 时间
        :return: sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, t时刻形状与x张量相同的张量
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[time])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[time])[:, None, None, None]
        # 生成一个形状与x张量相同的张量，其中的元素是从标准正态分布（均值为0，方差为1）中随机抽样得到的
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_time_steps(self, n):
        """
        采样时间步长
        :param n: 图像尺寸
        :return: 形状为(n,)的整数张量
        """
        # 生成一个具有指定形状(n,)的整数张量，其中每个元素都在low和high之间（包含 low，不包含 high）随机选择
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
```

## 3.3 DDPM类

```python
class Diffusion(BaseDiffusion):
    """
    DDPM扩散模型
    """

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cpu"):
        """
        扩散模型ddpm复现
        论文：《Denoising Diffusion Probabilistic Models》
        链接：https://arxiv.org/abs/2006.11239
        :param noise_steps: 噪声步长
        :param beta_start: β开始值
        :param beta_end: β结束值
        :param img_size: 图像大小
        :param device: 设备类型
        """

        super().__init__(noise_steps, beta_start, beta_end, img_size, device)

    def sample(self, model, n, labels=None, cfg_scale=None):
        """
        采样
        :param model: 模型
        :param n: 采样图片个数
        :param labels: 标签
        :param cfg_scale: classifier-free guidance插值权重，用于提升生成质量，避免后验坍塌（posterior collapse）问题
                            参考论文：《Classifier-Free Diffusion Guidance》
        :return: 采样图片
        """
        logger.info(msg=f"DDPM Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            # 输入格式为[n, 3, img_size, img_size]
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            # reversed(range(1, self.noise_steps)为反向迭代整数序列
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                # 时间步长，创建大小为n的张量
                t = (torch.ones(n) * i).long().to(self.device)
                # 这里判断网络是否有条件输入，例如多个类别输入
                if labels is None and cfg_scale is None:
                    # 图像与时间步长输入进模型中
                    predicted_noise = model(x, t)
                else:
                    predicted_noise = model(x, t, labels)
                    # 用于提升生成，避免后验坍塌（posterior collapse）问题
                    if cfg_scale > 0:
                        # 无条件预测噪声
                        unconditional_predicted_noise = model(x, t, None)
                        # torch.lerp根据给定的权重，在起始值和结束值之间进行线性插值，公式：input + weight * (end - input)
                        predicted_noise = torch.lerp(unconditional_predicted_noise, predicted_noise, cfg_scale)
                # 拓展为4维张量，根据时间步长t获取值
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                # 只需要步长大于1的噪声，详细参考论文P4页Algorithm2的第3行
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                # 在每一轮迭代中用x计算x的t - 1，详细参考论文P4页Algorithm2的第4行
                x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        model.train()
        # 将值恢复到0和1的范围
        x = (x.clamp(-1, 1) + 1) / 2
        # 乘255进入有效像素范围
        x = (x * 255).type(torch.uint8)
        return x
```

## 3.4 DDIM类（改进DDPM采样器）
实验表明，DDPM的每次去噪加噪过程都是从初始到结束，完成N次。而这种方法尽管生成的效果非常出色，但是带来的问题是**采样时间过长，训练速度慢**。为了解决该问题，跳步方法sample_steps被设计出来。

```python
class Diffusion(BaseDiffusion):
    """
    DDIM扩散模型
    """

    def __init__(self, noise_steps=1000, sample_steps=20, beta_start=1e-4, beta_end=0.02, img_size=256, device="cpu"):
        """
        扩散模型ddim复现
        论文：《Denoising Diffusion Implicit Models》
        链接：https://arxiv.org/abs/2010.02502
        :param noise_steps: 噪声步长
        :param sample_steps: 采样步长
        :param beta_start: β开始值
        :param beta_end: β结束值
        :param img_size: 图像大小
        :param device: 设备类型
        """
        super().__init__(noise_steps, beta_start, beta_end, img_size, device)
        # 采样步长，用于跳步
        self.sample_steps = sample_steps

        self.eta = 0

        # 计算迭代步长，跳步操作
        self.time_step = torch.arange(0, self.noise_steps, (self.noise_steps // self.sample_steps)).long() + 1
        self.time_step = reversed(torch.cat((torch.tensor([0], dtype=torch.long), self.time_step)))
        self.time_step = list(zip(self.time_step[:-1], self.time_step[1:]))

    def sample(self, model, n, labels=None, cfg_scale=None):
        """
        采样
        :param model: 模型
        :param n: 采样图片个数
        :param labels: 标签
        :param cfg_scale: classifier-free guidance插值权重，用于提升生成质量，避免后验坍塌（posterior collapse）问题
                            参考论文：《Classifier-Free Diffusion Guidance》
        :return: 采样图片
        """
        logger.info(msg=f"DDIM Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            # 输入格式为[n, 3, img_size, img_size]
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            # i和i的前一个时刻
            for i, p_i in tqdm(self.time_step):
                # t时间步长，创建大小为n的张量
                t = (torch.ones(n) * i).long().to(self.device)
                # t的前一个时间步长
                p_t = (torch.ones(n) * p_i).long().to(self.device)
                # 拓展为4维张量，根据时间步长t获取值
                alpha_t = self.alpha_hat[t][:, None, None, None]
                alpha_prev = self.alpha_hat[p_t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                # 这里判断网络是否有条件输入，例如多个类别输入
                if labels is None and cfg_scale is None:
                    # 图像与时间步长输入进模型中
                    predicted_noise = model(x, t)
                else:
                    predicted_noise = model(x, t, labels)
                    # 用于提升生成，避免后验坍塌（posterior collapse）问题
                    if cfg_scale > 0:
                        # 无条件预测噪声
                        unconditional_predicted_noise = model(x, t, None)
                        # torch.lerp根据给定的权重，在起始值和结束值之间进行线性插值，公式：input + weight * (end - input)
                        predicted_noise = torch.lerp(unconditional_predicted_noise, predicted_noise, cfg_scale)
                # 核心计算公式
                x0_t = torch.clamp((x - (predicted_noise * torch.sqrt((1 - alpha_t)))) / torch.sqrt(alpha_t), -1, 1)
                c1 = self.eta * torch.sqrt((1 - alpha_t / alpha_prev) * (1 - alpha_prev) / (1 - alpha_t))
                c2 = torch.sqrt((1 - alpha_prev) - c1 ** 2)
                x = torch.sqrt(alpha_prev) * x0_t + c2 * predicted_noise + c1 * noise
        model.train()
        # 将值恢复到0和1的范围
        x = (x + 1) * 0.5
        # 乘255进入有效像素范围
        x = (x * 255).type(torch.uint8)
        return x
```

# 4 生成结果
我们在以下4个数据集做了训练，采样器为`DDPM`，图片尺寸均为`64*64`，分别是`cifar10`，`NEUDET`，`NRSD-MN`和`Animate Face`。结果如下图所示：
**cifar10**

![cifar_244_ema](/cifar_244_ema.jpg)![cifar_294_ema](/cifar_294_ema.jpg)

**NEUDET**

![neudet_270_ema](/neudet_270_ema.jpg)![neudet_276_ema](/neudet_276_ema.jpg)![neudet_290_ema](/neudet_290_ema.jpg)![neudet_298_ema](/neudet_298_ema.jpg)![neudet_240_ema](/neudet_240_ema.jpg)![neudet_244_ema](/neudet_244_ema.jpg)![neudet_265_ema](/neudet_265_ema.jpg)

**NRSD**

![nrsd_180_ema](/nrsd_180_ema.jpg)![nrsd_188_ema](/nrsd_188_ema.jpg)![nrsd_194_ema](/nrsd_194_ema.jpg)![nrsd_203_ema](/nrsd_203_ema.jpg)![nrsd_210_ema](/nrsd_210_ema.jpg)![nrsd_217_ema](/nrsd_217_ema.jpg)![nrsd_218_ema](/nrsd_218_ema.jpg)![nrsd_248_ema](/nrsd_248_ema.jpg)![nrsd_276_ema](/nrsd_276_ema.jpg)![nrsd_285_ema](/nrsd_285_ema.jpg)![nrsd_298_ema](/nrsd_298_ema.jpg)

**Animate Face（整活生成)**

![animate_face_488_ema](/animate_face_488_ema.jpg)![animate_face_497_ema](/animate_face_497_ema.jpg)![animate_face_499_ema](/animate_face_499_ema.jpg)![animate_face_428_ema](/animate_face_428_ema.jpg)![animate_face_459_ema](/animate_face_459_ema.jpg)



同时，我们利用生成的64× 64的模型生成了160×160的NEU-DET图像（显存占用21GB）

![neu160_0](/neu160_0.jpg)![neu160_1](/neu160_1.jpg)![neu160_2](/neu160_2.jpg)![neu160_3](/neu160_3.jpg)![neu160_4](/neu160_4.jpg)![neu160_5](/neu160_5.jpg)

# 5 项目使用流程与参数设计
该部分为GitHub项目的具体使用流程和使用参数设计，详情请移步[README.md](https://github.com/chairc/Industrial-Defect-Diffusion-Model/blob/main/README.md)
## 5.1 环境检查
首先，你需要检查当前Anaconda或Miniconda中的环境是否符合本项目运行。

## 5.2 训练
#### 注意

本自README的训练GPU环境如下：使用具有6GB显存的NVIDIA RTX 3060显卡、具有11GB显存的NVIDIA RTX 2080Ti显卡和具有24GB（总计48GB，分布式训练）显存的NVIDIA RTX 6000（×2）显卡对模型进行训练和测试。**上述GPU均可正常训练**。
#### 5.2.1 开始你的第一个训练（以cifar10为例，模式单卡，有条件训练）

1. **导入数据集** 

   首先，将数据集上传至目标文件夹`datasets`中。上传后文件夹格式（例如：cifar10文件夹下存放着所有类别；class0文件夹下存储着class0这个类别的所有图片）如下方列表所示：

   ```yaml
    datasets
    └── cifar10
        ├── class0
        ├── class1
        ├── class2
        ├── class3
        ├── class4
        ├── class5
        ├── class6
        ├── class7
        ├── class8
        └── class9
   ```

   此时你的训练前准备已经完毕。

2. **设置训练参数**

   打开`train.py`文件，修改`if __name__ == "__main__":`中的`parser`参数；

   设置`--conditional`参数为`True`，因为是多类别训练，所以需要开启，单类别可以不开启也可以开启；

   设置`--run_name`参数为你想创建的文件名称，例如`cifar_exp1`；

   设置`--dataset_path`参数为`/你的/本地/或/远程服务器/文件/地址/datasets/cifar10`；

   设置`--result_path`参数为`/你的/本地/或/远程服务器/文件/地址/results`；

   设置`--num_classes`参数为`10`，这是你的类别总数）；

   设置更多参数（自定义），如果报`CUDA out of memory`错误，将`--batch_size`、`--num_workers`调小；

   在自定义参数中，你可以设置不同的`--sample`例如`ddpm`或`ddim`，设置不同的训练网络`--network`例如`unet`或`cspdarkunet`。当然激活函数`--act`，优化器`--optim`，半精度训练`--fp16`，学习率方法`--lr_func`等参数也都是可以自定义设置的。

   详细命令可参考**训练参数**。

3. **等待训练过程**

   点击`run`运行后，项目会在`results`文件夹中生成`cifar_exp1`文件夹，该文件夹中会保存训练日志文件、模型训练文件、模型EMA文件、模型优化器文件、训练的所有最后一次保存的文件和评估后生成的图片。

4. **查看结果**

   找到`results/cifar_exp1`文件夹即可查看训练结果。

**↓↓↓↓↓↓↓↓↓↓下方为多种训练方式、训练详细参数讲解↓↓↓↓↓↓↓↓↓↓**
### 5.2.2 普通训练

1. 以`landscape`数据集为例，将数据集文件放入`datasets`文件夹中，该数据集的总路径如下`/your/path/datasets/landscape`，图片存放在`/your/path/datasets/landscape/images`（是的你需要把图片放到文件夹下面，不然`util`中的`ImageFolder`会报找不到错误），数据集图片路径如下`/your/path/datasets/landscape/images/*.jpg`

2. 打开`train.py`文件，找到`--dataset_path`参数，将参数中的路径修改为数据集的总路径，例如`/your/path/datasets/landscape`

3. 设置必要参数，例如`--sample`，`--conditional`，`--run_name`，`--epochs`，`--batch_size`，`--image_size`，`--result_path`等参数，若不设置参数则使用默认设置。我们有两种参数设置方法，其一是直接对`train.py`文件`if __name__ == "__main__":`中的`parser`进行设置（**我们推荐这种方式**）；其二是在控制台在`/your/path/Defect-Diffiusion-Model/tools`路径下输入以下命令：
   **有条件训练命令**

   ```bash
   python train.py --sample ddpm --conditional --run_name df --epochs 300 --batch_size 16 --image_size 64 --num_classes 10 --dataset_path /your/dataset/path --result_path /your/save/path
   ```

   **无条件训练命令**

   ```bash
   python train.py --sample ddpm --run_name df --epochs 300 --batch_size 16 --image_size 64 --dataset_path /your/dataset/path --result_path /your/save/path
   ```

4. 等待训练即可

5. 若因异常原因中断训练，我们可以在`train.py`文件，首先将`--resume`设置为`True`，其次设置异常中断的迭代编号，再写入该次训练的所在文件夹（run_name），最后运行文件即可。也可以使用如下命令进行恢复：
   **有条件恢复训练命令**

   ```bash
   # 此处为输入--start_epoch参数，使用当前编号权重
   python train.py --resume --start_epoch 10 --sample ddpm --conditional --run_name df --epochs 300 --batch_size 16 --image_size 64 --num_classes 10 --dataset_path /your/dataset/path --result_path /your/save/path
   ```
   
   ```bash
   # 此处为不输入--start_epoch参数，默认使用last权重
   python train.py --resume --sample ddpm --conditional --run_name df --epochs 300 --batch_size 16 --image_size 64 --num_classes 10 --dataset_path /your/dataset/path --result_path /your/save/path
   ```
   **无条件恢复训练命令**

   ```bash
   python train.py --resume --start_epoch 10 --sample ddpm --run_name df --epochs 300 --batch_size 16 --image_size 64 --dataset_path /your/dataset/path --result_path /your/save/path
   ```
   
   ```bash
   # 此处为不输入--start_epoch参数，默认使用last权重
   python train.py --resume --sample ddpm --run_name df --epochs 300 --batch_size 16 --image_size 64 --dataset_path /your/dataset/path --result_path /your/save/path
   ```

6. 预训练模型在每次大版本[Release](https://github.com/chairc/Integrated-Design-Diffusion-Model/releases)中发布，请留意。预训练模型使用方法如下，首先将对应`network`、`image_size`、`act`等相同参数的模型下到本地任意文件夹下。直接调整`train.py`中`--pretrain`和`--pretrain_path`即可。也可以使用如下命令进行预训练：  
   **使用有条件预训练模型训练命令**

   ```bash
   python train.py --pretrain --pretrain_path /your/pretrain/path/model.pt --sample ddpm --conditional --run_name df --epochs 300 --batch_size 16 --image_size 64 --dataset_path /your/dataset/path --result_path /your/save/path
   ```

   **使用无条件预训练模型训练命令**

   ```bash
   python train.py --pretrain --pretrain_path /your/pretrain/path/model.pt --sample ddpm --run_name df --epochs 300 --batch_size 16 --image_size 64 --dataset_path /your/dataset/path --result_path /your/save/path
   ```
### 5.2.3 分布式训练

1. 基本配置与普通训练相似，值得注意的是开启分布式训练需要将`--distributed`设置为`True`。为了防止随意设置分布式训练，我们为开启分布式训练设置了几个基本条件，例如`args.distributed`、`torch.cuda.device_count() > 1`和`torch.cuda.is_available()`。

2. 设置必要的参数，例如`--main_gpu`和`--world_size`。`--main_gpu`通常设置为主要GPU，例如做验证、做测试或保存权重，我们仅在单卡中运行即可。而`world_size`的值会与实际使用的GPU数量或分布式节点数量相对应。

3. 我们有两种参数设置方法，其一是直接对`train.py`文件`if __name__ == "__main__":`中的`parser`进行设置；其二是在控制台在`/your/path/Defect-Diffiusion-Model/tools`路径下输入以下命令：

**有条件训练命令**

   ```bash
python train.py --sample ddpm --conditional --run_name df --epochs 300 --batch_size 16 --image_size 64 --num_classes 10 --dataset_path /your/dataset/path --result_path /your/save/path --distributed --main_gpu 0 --world_size 2
   ```

   **无条件训练命令**

   ```bash
python train.py --sample ddpm --run_name df --epochs 300 --batch_size 16 --image_size 64 --dataset_path /your/dataset/path --result_path /your/save/path --distributed --main_gpu 0 --world_size 2
   ```

4. 等待训练即可，中断恢复同基本训练一致。

![IDDM_training](/IDDM_training.png)

**参数讲解**

| **参数名称**                 | 条件参数 | 参数使用方法                     | 参数类型 | 参数解释                                                     |
| ---------------------------- | :------: | -------------------------------- | :------: | ------------------------------------------------------------ |
| --seed                       |          | 初始化种子                       |   int    | 设置初始化种子，可复现网络生成的图片                         |
| --conditional                |          | 开启条件训练                     |   bool   | 若开启可修改自定义配置，例如修改类别、classifier-free guidance插值权重 |
| --sample                     |          | 采样方式                         |   str    | 设置采样器类别，当前支持ddpm，ddim                           |
| --network                    |          | 训练网络                         |   str    | 设置训练网络，当前支持UNet，CSPDarkUNet                      |
| --run_name                   |          | 文件名称                         |   str    | 初始化模型的文件名称，用于设置保存信息                       |
| --epochs                     |          | 总迭代次数                       |   int    | 训练总迭代次数                                               |
| --batch_size                 |          | 训练批次                         |   int    | 训练批次大小                                                 |
| --num_workers                |          | 加载进程数量                     |   int    | 用于数据加载的子进程数量，大量占用CPU和内存，但可以加快训练速度 |
| --image_size                 |          | 输入图像大小                     |   int    | 输入图像大小，自适应输入输出尺寸                             |
| --dataset_path               |          | 数据集路径                       |   str    | 有条件数据集，例如cifar10，每个类别一个文件夹，路径为主文件夹；无条件数据集，所有图放在一个文件夹，路径为图片文件夹 |
| --amp                        |          | 混合精度训练                     |   bool   | 开启混合精度训练，有效减少显存使用，但无法保证训练精度和训练结果 |
| --optim                      |          | 优化器                           |   str    | 优化器选择，目前支持adam和adamw                              |
| --act                        |          | 激活函数                         |   str    | 激活函数选择，目前支持gelu、silu、relu、relu6和lrelu         |
| --lr                         |          | 学习率                           |  float   | 初始化学习率                                                 |
| --lr_func                    |          | 学习率方法                       |   str    | 设置学习率方法，当前支持linear、cosine和warmup_cosine        |
| --result_path                |          | 保存路径                         |   str    | 保存路径                                                     |
| --save_model_interval        |          | 是否在训练中储存                 |   bool   | 是否在训练中储存，根据可视化生成样本信息筛选模型，如果为False，则只保存最后一个模型 |
| --save_model_interval_epochs |          | 保存模型周期                     |   int    | 保存模型间隔并每 X 周期保存一个模型                          |
| --start_model_interval       |          | 设置开始每次训练存储编号         |   int    | 设置开始每次训练存储的epoch编号，该设置可节约磁盘空间，若不设置默认-1，若设置则从第epoch时开始保存每次训练pt文件，需要与--save_model_interval同时开启 |
| --vis                        |          | 可视化数据集信息                 |   bool   | 打开可视化数据集信息，根据可视化生成样本信息筛选模型         |
| --num_vis                    |          | 生成的可视化图像数量             |   int    | 生成的可视化图像数量。如果不填写，则默认生成图片个数为数据集类别的个数 |
| --image_format               |          | 生成图片格式                     |   str    | 在训练中生成图片格式，默认为png                              |
| --noise_schedule             |          | 加噪方法                         |   str    | 该方法是模型噪声添加方法                                     |
| --resume                     |          | 中断恢复训练                     |   bool   | 恢复训练将设置为“True”。注意：设置异常中断的epoch编号若在--start_model_interval参数条件外，则不生效。例如开始保存模型时间为100，中断编号为50，由于我们没有保存模型，所以无法设置任意加载epoch点。每次训练我们都会保存xxx_last.pt文件，所以我们需要使用最后一次保存的模型进行中断训练 |
| --start_epoch                |          | 中断迭代编号                     |   int    | 设置异常中断的epoch编号，模型会自动加载当前编号的检查点      |
| --pretrain                   |          | 预训练模型训练                   |   bool   | 设置是否启用加载预训练模型训练                               |
| --pretrain_path              |          | 预训练模型路径                   |   str    | 预训练模型加载地址                                           |
| --use_gpu                    |          | 设置运行指定的GPU                |   int    | 一般训练中设置指定的运行GPU，输入为GPU的编号                 |
| --distributed                |          | 分布式训练                       |   bool   | 开启分布式训练                                               |
| --main_gpu                   |          | 分布式训练主显卡                 |   int    | 设置分布式中主显卡                                           |
| --world_size                 |          | 分布式训练的节点等级             |   int    | 分布式训练的节点等级， world_size的值会与实际使用的GPU数量或分布式节点数量相对应 |
| --num_classes                |    是    | 类别个数                         |   int    | 类别个数，用于区分类别                                       |
| --cfg_scale                  |    是    | classifier-free guidance插值权重 |   int    | classifier-free guidance插值权重，用户更好生成模型效果       |




## 5.3 生成

1. 打开`generate.py`文件，找到`--weight_path`参数，将参数中的路径修改为模型权重路径，例如`/your/path/weight/model.pt`

2. 设置必要参数，例如`--conditional`，`--generate_name`，`--num_images`，`--num_classes`，`--class_name`，`--image_size`，`--result_path`等参数，若不设置参数则使用默认设置。我们有两种参数设置方法，其一是直接对`generate.py`文件`if __name__ == "__main__":`中的`parser`进行设置；其二是在控制台在`/your/path/Defect-Diffiusion-Model/tools`路径下输入以下命令：
    **有条件生成命令（1.1.1版本以上）**

   ```bash
   python generate.py --generate_name df --num_images 8 --class_name 0 --image_size 64 --weight_path /your/path/weight/model.pt --sample ddpm
   ```

   **无条件生成命令（1.1.1版本以上）**

   ```bash
   python generate.py --generate_name df --num_images 8 --image_size 64 --weight_path /your/path/weight/model.pt --sample ddpm
   ```
   **有条件生成命令（1.1.1版本及以下）**
   
   ```bash
   python generate.py --conditional --generate_name df --num_images 8 --num_classes 10 --class_name 0 --image_size 64 --weight_path /your/path/weight/model.pt --sample ddpm --network unet --act gelu 
   ```
   
   **无条件生成命令（1.1.1版本及以下）**
   
   ```bash
   python generate.py --generate_name df --num_images 8 --image_size 64 --weight_path /your/path/weight/model.pt --sample ddpm --network unet --act gelu 
   ```

3. 等待生成即可



**参数讲解**

**参数讲解**

| **参数名称**    | 条件参数 | 参数使用方法                     | 参数类型 | 参数解释                                                     |
| --------------- | :------: | -------------------------------- | :------: | ------------------------------------------------------------ |
| --conditional   |          | 开启条件生成                     |   bool   | 若开启可修改自定义配置，例如修改类别、classifier-free guidance插值权重 |
| --generate_name |          | 文件名称                         |   str    | 初始化模型的文件名称，用于设置保存信息                       |
| --image_size    |          | 输入图像大小                     |   int    | 输入图像大小，自适应输入输出尺寸。如果输入为-1并且开启条件生成为真，则模型为每类输出一张图片 |
| --image_format  |          | 生成图片格式                     |   str    | 生成图片格式，jpg/png/jpeg等。推荐使用png获取更好的生产质量  |
| --num_images    |          | 生成图片个数                     |   int    | 单次生成图片个数                                             |
| --weight_path   |          | 权重路径                         |   str    | 模型权重路径，网络生成需要加载文件                           |
| --result_path   |          | 保存路径                         |   str    | 保存路径                                                     |
| --sample        |          | 采样方式                         |   str    | 设置采样器类别，当前支持ddpm，ddim**（1.1.1版本后的模型可不用设置）** |
| --network       |          | 训练网络                         |   str    | 设置训练网络，当前支持UNet，CSPDarkUNet**（1.1.1版本后的模型可不用设置）** |
| --act           |          | 激活函数                         |   str    | 激活函数选择，目前支持gelu、silu、relu、relu6和lrelu。如果不选择，会产生马赛克现象**（1.1.1版本后的模型可不用设置）** |
| --num_classes   |    是    | 类别个数                         |   int    | 类别个数，用于区分类别**（1.1.1版本后的模型可不用设置）**    |
| --class_name    |    是    | 类别名称                         |   int    | 类别序号，用于对指定类别生成。如果输入为-1，则模型为每类输出一张图片 |
| --cfg_scale     |    是    | classifier-free guidance插值权重 |   int    | classifier-free guidance插值权重，用户更好生成模型效果       |


# 6 当前已完成工作
- [x] 新增cosine学习率优化（2023-07-31）
- [x] 使用效果更优的U-Net网络模型（2023-11-09）
- [x] 更大尺寸的生成图像（2023-11-09）
- [x] 多卡分布式训练（2023-07-15）
- [x] 云服务器快速部署和接口（2023-08-28）
- [x] 增加DDIM采样方法（2023-08-03）
- [x] 支持其它图像生成（2023-09-16）
- [x] 低分辨率生成图像进行超分辨率增强[~~超分模型效果待定~~]（2024-02-18）
- [x] 重构model整体结构（2023-12-06）
- [x] 编写可视化webui界面（2024-01-23）
- [x] 增加PLMS采样方法（2024-03-12）
- [x] 增加FID方法验证图像质量（2024-05-06）
- [x] 增加生成图像Socket和网站服务部署（2024-11-13）
# 7 下一步计划

- [ ] 使用Latent方式降低显存消耗
- [ ] 增加Docker部署
- [ ] 增加PSNR和SSIM方法验证超分图像质


如有任何问题，请到Github提交issue或联系我email：chenyu1998424@gmail.com
