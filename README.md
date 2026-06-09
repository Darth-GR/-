## 参考配置
- 系统：Ubuntu 20.04
- CUDA：12.4
- gcc：9.5
- cmake：4.0.1

## 作业要求
    1. 实现DataSet、DataLoader类，能够从表格数据文件、图像文件读取Iris、MNIST、Fashion-MNIST等数据集；
    2. 实现ppt中列举的Module、Optimizer、Scheduler子类，实现He和Xavier初始化算法；
    3. 使用简易框架进行ResidualMLP实验，实现模型存储和读取，测试阶段不计算梯度。
    4. 补齐Value、Op、Tensor等类中缺少的成员函数 ，实现ppt中列举的TensorOp子类 (够用即可)，实现拓扑排序、自动求导并测试（选做自动微分部分，只要能实现模块化的程序设计即可）。

     关于神经网络的梯度计算，可以补齐示例代码里的needle框架，也可以手写Module子类的backward()。
     DDL: 2026-7-10，提交代码和实验报告。