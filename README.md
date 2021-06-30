# 2021微信大数据挑战赛

> 对于给定的一定数量到访过微信视频号“热门推荐”的用户， 根据这些用户在视频号内的历史n天的行为数据，通过算法在测试集上预测出这些用户对于不同视频内容的互动行为（包括点赞、点击头像、收藏、转发等）的发生概率。 本次比赛以多个行为预测结果的加权uAUC值进行评分

第一次接触这类比赛，成绩不大行，记录一下自己实验到的东西，并不一定对

因为第一次对于什么特征在这类任务中比较重要也不清楚，造的特征主要参考了大佬的分享[微信视频号推荐算法解题思路](https://mp.weixin.qq.com/s/yE5yThqZ8R9v4EIxlr3bsA)

nn模型用的是deepctr的deepfm模型，树模型用的lightgbm

- 对nn尝试过各类采样方法都没有全量训练效果好
- deepfm的dnn层数[128,128,128]好像就够用了
- 使用adam比adagrad好
- deepfm、xdeepfm、autoint效果差不多
- batchsize尝试中1024结果比较好
- deepfm中统计特征的作用不大，加上feed embedding后有很大的提升
- 对tags和keyword使用word2vec好像没有multilable降维后好用
- user相关的特征比较好用，feed的特征作用不是很明显
- nn多折融合有明显的提升，融合树模型和nn也有大的提升

- 没有利用到任务间的相关性，没来的急试试mmoe

想看看大佬的参数是怎么样的，等着看分享

**数据集下载地址**

链接: https://pan.baidu.com/s/1SPmyv7zoHhDVm57N0OgUzA

提取码: 8g95 