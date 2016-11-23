操作平台

ubuntu 14.04 + CPU + python

mnist例子为手写体识别例子，caffe的HelloWorld

1.下载数据： ./get_mnist.sh

2.转换数据格式：./create_mnist.sh
  生成  mnist_test_lmdb mnist_train_lmdb

3.确定模型：lenet_train_test.prototxt

4.确定超参数:lenet_solver.prototxt

5.确定模型 开始训练
  ./Bulid/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt

6.迭代10000次 



ps：
   lenet_iter_10000.caffemodel 为 最终模型

   draw_loss.py 为 绘制loss图像工具

   mnist.png 为 绘制的net模型

   mnistloss.png 为绘制的loss图像
   
   成功率为 98.1%

