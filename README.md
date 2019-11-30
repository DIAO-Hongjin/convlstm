# convlstm

A graduation project

运行说明：

在运行代码前，需要安装python3.6和pytorch1.0.0，并执行project_code/make.sh安装可变形卷积模块。

在移动手写数字数据集上进行实验时，数据集路径为data/moving-mnist-example/*.npz。
测试convlstm方法时，运行文件convlstm_moving_mnist.py。
测试deformable convlstm方法时，运行文件dconvlstm_moving_mnist.py。
测试modulated deformable convlstm方法时，运行文件mdconvlstm_moving_mnist.py。

在香港天文台雷达回波图像数据集上进行实验时，按照 https://github.com/sxjscience/HKO-7 中的方法获取和使用数据集。
测试最后一帧方法时，运行文件last_frame.py。
测试ROVER方法时，运行文件rover.py。
测试引入平衡损失函数的convlstm方法时，运行文件convlstm_HKO_7.py。
测试引入平衡损失函数的deformable convlstm方法时，运行文件dcomvlstm_HKO_7.py。
测试引入平衡损失函数的modulated deformable convlstm方法时，运行文件mdconvlstm_HKO-7.py。
测试使用普通损失函数的convlstm方法时，运行文件convlstm_HKO_7_non_balance.py。

参考论文：

[1]. Shi X, Chen Z, Wang H, et al. Convolutional LSTM Network: A Machine Learning Approach for Precipitation 
Nowcasting[A]. Advances In Neural Information Processing Systems[C], 2015: 802-810.
[2]. Dai J, Qi H, Xiong Y, et al. Deformable convolutional networks[A]. Proceedings of the IEEE international 
conference on computer vision[C], 2017: 764-773.
[3]. Zhu X, Hu H, Lin S, et al. Deformable convnets v2: More deformable, better results[J]. arXiv preprint 
arXiv:1811.11168, 2018.
