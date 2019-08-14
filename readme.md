# 智子杯口音分类比赛<br>
2019.7
==============================<br>
##数据处理部分：<br>
1.做了数据增强：将图片进行了水平镜像翻转，之后将之前的整体垂直翻转（这个对于之后的实验来说好像并不理想）<br>
2.一开始还做了OSTU的处理，但是没有继续尝试往下做，一个是不知道这个对后面的有没有作用，还一个就是通道问题得进行转换来转换去【如果再次碰到可以在做一下】<br>
<br>
<br>
##解题思路：<br>
1.数据给出的不同种类人的语音（被处理过后的）图像，噪声点很大，想用滤波去噪，但是会发现，有些语音的图像，居然有模糊【这部分是我能力有限的地方，对于一些图像做出正确的处理】，所以我是直接把图片放入网络里进行的操作<br>
2.选择网络阶段:<br>
1)首先想到的是LSTM，因为处理语音，还是LSTM厉害呀，不是嘛？[lstm_minist分类](https://github.com/EillotY/DL_game/blob/master/lstm_mnist.py)<br>
2)采用卷积神经网络，因为是分类问题，卷积来做的话，一个是因为cnn已经都可以做语音了，你一个小图像算什么？，采用了densenet和resnet<br>

##实验阶段<br>
1）LSTM做的比较差，只是比random高了10个百分点，所以在中期就开始放弃了【可能没有深入这个网络所导致的吧，以后有机会可以好好看一下】<br>
2）直接用了[densenet](https://github.com/EillotY/DL_game/blob/master/demo_densenet_adddropout.py)，它其实就是将之前层得到的特征层继续给后面的网络层一同输入（也为特征层复用），数据在9000量下，能跑到72%<br>
![](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1565778177731&di=925bcceb374a21c1a77d7e4e317864b7&imgtype=0&src=http%3A%2F%2Fpic2.zhimg.com%2Fv2-8999bcd09274bc92a89cea939fcb44f9_b.jpg)<br>
3）resnet，有53，101，152；对于resnet152来说，图片的输入格式有要求，224*224，所以得将之前处理的数据resize下，网络采用的是残差trick，以便网络深度越大的情况下还能够收敛，这种tirck非常有用，只不过后期我硬件只是1060,跑的很慢，就resnet152没有得到一个好的结果<br>

##提升阶段<br>
1)在这个阶段就是调参的过程，和朋友聊天得知一个叫遗传算法的东西（还有其他）[其实就是用来调整你当前网络的最佳参数设置]，这个对比赛的话会很有用（笔者还没学习，这个后期补）<br>
2)但是期间我找到了一个trick，叫[Senet]，这个trick，大致的意思就是从特征层出发将一些对后续网络层没用的就不传递了，有用的就往后传递<br>
![](https://img-blog.csdn.net/20180423230918755)<br>


