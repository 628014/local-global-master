
## Install pre-requested libraries
- PyTorch 1.0
- Sklearn
## Steps to reproduce the results
- download the pre-processd [dataset](https://drive.google.com/file/d/19JMK_IeBFlEQAEt_nrWsJcHrdyHcZMhm/view?usp=sharing) 
- extract the dataset to a folder
- run `python experiments.py <experiment name> <path to the dataset>`


# 1. 前期说明 ：
我执行的环境如下 ：
- python 3.8
- pytorch 1.10.0 
- cuda 11.0
- conda 4.8.4
- Sklearn 
# 2. 前期准备 ：
## 2.1 使用conda安装GPU版pytorch
因为我之前是装过的`cuda 11.0 `的，之前我以为，一定要的对应的`cuda`和对应的`pytorch`要对应上，所有就有很多是装了又卸，但是事实上却不是的，只是装对应的
比较好一些，但是没有强调，而在`pytorch`的官网上是打开就默认好了你电脑最适合的版本的，我们直接下就OK了，我使用的是 ：`conda install pytorch torchvision torchaudio cudatoolkit=10.2`
具体参照

 [需要windows10使用conda安装GPU版pytorch并使用GPU进行运算](https://blog.csdn.net/weixin_44405644/article/details/102992782) 
 [如何查看windows的CUDA版本](https://blog.csdn.net/qq_38295511/article/details/89223169)
注意这个版本是我们电脑能够用的最高的版本，当然也是自己装上去的，但是不是我们需要的`pytorch`对应的版本
 
## 2.2 项目运行

解压项目到某一个文件夹下面，图片[dataset](https://drive.google.com/file/d/19JMK_IeBFlEQAEt_nrWsJcHrdyHcZMhm/view?usp=sharing),
不放在和项目的里面，放在外面就可以了


# 3. 常用的命令 ：
## 3.1 conda环境使用基本命令

[conda常用命令](https://blog.csdn.net/zhayushui/article/details/80433768)

- conda update -n base conda        #update最新版本的conda
- conda create -n xxxx python=3.8   #创建python3.8的xxxx虚拟环境
- activate xxxx                     #开启xxxx环境
- conda deactivate                  #关闭环境
- conda env list                    #显示所有的虚拟环境
- conda info --envs                 #显示所有的虚拟环境

### 3.2 pip 相关命令 
- pip list                          #列出当前缓存的包
- pip purge                         #清除缓存
- pip remove                        #删除对应的缓存
- pip help                          #帮助
- pip install xxx                   #安装xxx包
- pip uninstall xxx                 #删除xxx包
- pip show xxx                      #展示指定的已安装的xxx包
- pip check xxx                     #检查xxx包的依赖是否合适

### 3.3 国内源 
- 阿里云                    http://mirrors.aliyun.com/pypi/simple/
- 中国科技大学         https://pypi.mirrors.ustc.edu.cn/simple/ 
- 豆瓣(douban)         http://pypi.douban.com/simple/ 
- 清华大学                https://pypi.tuna.tsinghua.edu.cn/simple/
- 中国科学技术大学  http://pypi.mirrors.ustc.edu.cn/simple/

### 3.4 打印版本的一些问题 ：

- 查看pytorch版本的方法   `print(torch.__version__)`
- 查看显卡的使用情况 windows下可以这样做：打开`cmd`窗口，输入`nvidai-smi`查看显卡使用情况如图，如果`nvidai-smi`无效，需要进行path环境的导入
`C:\Program Files\NVIDIA Corporation\NVSMI` 之后才能出现结果
- 检查是否有对应的 `cuda`版本的`pytorch` : `print(torch.cuda.is_available())`

### 3.5 遇到的一些问题 ：
1. [往pycharm里面添加我anconda里面的虚拟环境](往pycharm里面添加我anconda里面的虚拟环境 )
2. [IOError: broken data stream when reading image file](https://blog.csdn.net/fengzhongluoleidehua/article/details/85949173) 图片访问不对 使用代码 ：`ImageFile.LOAD_TRUNCATED_IMAGES = True` 
3. [windows释放GPU内存方法](http://www.noobyard.com/article/p-rlafxvmt-rx.html)
4. [桌面窗口管理器（dwm.exe）占用内存高怎么办？](https://www.zhihu.com/question/429569646)
5. [Pytorch GPU内存占用很高,但是利用率很低如何解决](https://www.jb51.net/article/213809.htm)
or [Pytorch GPU内存占用很高，但是利用率很低](https://blog.csdn.net/weixin_43402775/article/details/108725040)
输入nvidia-smi来观察显卡的GPU内存占用率（Memory-Usage），显卡的GPU利用率（GPU-util）
PU内存占用率（Memory-Usage） 往往是由于模型的大小以及batch size的大小，来影响这个指标 显卡的GPU利用率（GPU-util） 往往跟代码有关，有更多的io运算，cpu运算就会导致利用率变低。
针对这个代码，我的机器是跑不了的，所以只能修改对应的batch size来降低利用率

6. 检查是否有对应的 `cuda`版本的`pytorch` [AssertionError: Torch not compiled with CUDA enabled](https://blog.csdn.net/dujuancao11/article/details/114006234)



