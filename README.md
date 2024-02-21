# DPO方法训练大语言模型,简易实现代码

环境信息:

python=3.10

torch==2.1.0(cuda)

transformers==4.34.0

datasets==2.14.5

trl==0.7.2

视频课程:https://www.bilibili.com/video/BV1Fa4y1X7xh

在model_dpo(训练)上预测产生choice回答的概率和reject回答概率之差 和 model_dpo_ref(不训练)产生choice回答的概率-reject回答概率之差的KL散度 作为loss
model_dpo(训练)上预测产生choice回答的概率:
问题：S427.01=

choice 回答：8.85*55.21+-61.60E,计算预测产生choice回答的概率如下

$$
\sum_{i}^{|N|} log P(label_{i}|S427.01)
$$

其中|N|表示字符串`label=8.85*55.21+-61.60E`的token个数,$label_{i}$表示`label`字符串中的第i个token

reject 回答：None
