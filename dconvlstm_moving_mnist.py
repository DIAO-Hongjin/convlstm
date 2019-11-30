__author__ = 'DIAO Hongjin'


import torch
import numpy
import time
import logging
import random
from numpy import sqrt
from modules import *
from torch.utils import data
from torch import nn
from torch import sigmoid
from torch import tanh
from torch import optim
from PIL import Image
from matplotlib import pyplot
from torchvision import transforms


# logger的输出格式
format_dict = {
    1: logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
    2: logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
    3: logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
    4: logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
    5: logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
}


class Logger(object):
    """
    指定保存日志的文件路径，日志级别，以及调用文件，将日志存入到指定的文件中。
    """

    def __init__(self, log_name, log_level, log_object):
        """
        初始化日志对象。

        :param log_name: 日志文件的路径
        :param log_level: 日志级别
        :param log_object: 日志对象
        """

        # 创建logger
        self.logger = logging.getLogger(log_object)
        self.logger.setLevel(logging.DEBUG)

        # 创建用于写入日志文件的handler
        file_handler = logging.FileHandler(log_name)
        file_handler.setLevel(logging.DEBUG)

        # 创建用于输出到控制台的handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 定义handler的输出格式
        formatter = format_dict[int(log_level)]
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_log(self):
        """
        获取logger对象。

        :return: 返回定义的logger对象
        """

        return self.logger


class MovingMnistDataSet(data.Dataset):
    """
    定义Moving Mnist数据集。
    ----------
    Moving Mnist 数据集说明

    该数据集包含3个如下属性：
    dims: 形状为(1, 3)的numpy数组，（特征维度，行，列）
    clips: 形状为(2, n, 2)的3维int32类型numpy数组，（输入/输出，下标，起点/序列长度），其中起点从0开始
    input_raw_data: 4维numpy数组，（时间戳，特征维度，行，列）
    """

    def __init__(self, path, patch_size):
        """
        初始化Moving Mnist数据集。

        :param path: Moving Mnist数据集的路径
        :param patch_size: 切割尺寸
        """
        self.data_set = numpy.load(path)
        self.data = torch.from_numpy(self.data_set["input_raw_data"])
        self.clips = self.data_set["clips"]
        self.raw_shape = self.data_set["dims"][0][1:]
        self.patch_size = patch_size

    def __getitem__(self, item):
        """
        为数据集的样本提供索引。

        :param item: 样本的索引
        :return: （样本的输入序列，样本的输出序列）
        """

        # 读取序列
        in_seq_item = self.data[self.clips[0, item, 0]:self.clips[0, item, 0] + self.clips[0, item, 1], :, :, :]
        out_seq_item = self.data[self.clips[1, item, 0]:self.clips[1, item, 0] + self.clips[1, item, 1], :, :, :]

        # 切割图片
        in_seq_item = reshape_patch(in_seq_item, self.clips[0, item, 1], self.raw_shape, self.patch_size)
        out_seq_item = reshape_patch(out_seq_item, self.clips[1, item, 1], self.raw_shape, self.patch_size)

        return in_seq_item, out_seq_item

    def __len__(self):
        """
        获取数据集的大小。

        :return: 数据集的样本数量
        """
        return self.clips.shape[1]


def moving_mnist_data_loader(path, batch_size=16, patch_size=4, shuffle=False, num_workers=0):
    """
    按照批尺寸将数据集分成多个批次。

    :param path: 数据集所在路径
    :param batch_size: 批尺寸
    :param patch_size: 切割尺寸
    :param shuffle: 是否打乱数据集样本的顺序
    :param num_workers: 加载数据的子进程数
    :return: 按批次打包后的数据集
    """

    data_loader = torch.utils.data.DataLoader(
        MovingMnistDataSet(path, patch_size),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return data_loader


def reshape_patch(in_tensor, seq_len, raw_shape, patch_size):
    """
    按切割尺寸切割图片。
    切割前：将单通道的图片(1, raw height, raw width)
    切割后：多通道图片(patch size * patch size, raw height / patch size, raw width / patch size)

    :param in_tensor: 待切割的单通道图片
    :param seq_len: 图片的序列长度
    :param raw_shape: 图片的原始形状
    :param patch_size: 切割尺寸
    :return: 切割后的多通道图片
    """

    ret = in_tensor.reshape(seq_len, int(raw_shape[0]/patch_size), patch_size, int(raw_shape[1]/patch_size), patch_size)
    ret = ret.permute(0, 2, 4, 1, 3)
    ret = ret.reshape(seq_len, patch_size*patch_size, int(raw_shape[0]/patch_size), int(raw_shape[1]/patch_size))

    return ret


def reshape_back(patch_tensor, seq_len, patch_shape, patch_size):
    """
    恢复被切割的图片。

    :param patch_tensor: 切割后的多通道图片
    :param seq_len: 图片的序列长度
    :param patch_shape: 切割后的图片形状
    :param patch_size: 切割尺寸
    :return: 单通道原始图片
    """

    ret = patch_tensor.reshape(seq_len, patch_size, patch_size, patch_shape[0], patch_shape[1])
    ret = ret.permute(0, 3, 1, 4, 2)
    ret = ret.reshape(seq_len, 1, patch_shape[0]*patch_size, patch_shape[1]*patch_size)

    return ret


class DeformConvLSTMCell(nn.Module):
    """
        单个dConvLSTM模块（输入--隐藏层--输出）
    """

    def __init__(self, shape, in_channels, out_channels, kernel_size):
        """
        初始化单个dconvlstm模块。

        :param shape: 输入张量的形状，即：（高度，宽度）
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核的尺寸
        """

        super(DeformConvLSTMCell, self).__init__()

        self.shape = shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size[0]//2, kernel_size[1]//2

        # ==================================================
        # 执行（可变形）卷积操作
        # conv1: W_xi * X_t    conv2: W_hi * H_(t-1)
        # conv3: W_xf * X_t    conv4: W_hf * H_(t-1)
        # conv5: W_xc * X_t    conv6: W_hc * H_(t-1)
        # conv7: W_xo * X_t    conv8: W_ho * H_(t-1)
        # ==================================================

        self.conv1 = DeformConvPack(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=False
        )
        self.conv2 = DeformConvPack(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=False
        )
        self.conv3 = DeformConvPack(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=False
        )
        self.conv4 = DeformConvPack(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=False
        )
        self.conv5 = DeformConvPack(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=False
        )
        self.conv6 = DeformConvPack(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=False
        )
        self.conv7 = DeformConvPack(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=False
        )
        self.conv8 = DeformConvPack(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=False
        )

        # 权重初始化
        nn.init.uniform_(self.conv1.weight, -1.0/sqrt(self.in_channels), 1.0/sqrt(self.in_channels))
        nn.init.uniform_(self.conv2.weight, -1.0/sqrt(self.out_channels), 1.0/sqrt(self.out_channels))
        nn.init.uniform_(self.conv3.weight, -1.0/sqrt(self.in_channels), 1.0/sqrt(self.in_channels))
        nn.init.uniform_(self.conv4.weight, -1.0/sqrt(self.out_channels), 1.0/sqrt(self.out_channels))
        nn.init.uniform_(self.conv5.weight, -1.0/sqrt(self.in_channels), 1.0/sqrt(self.in_channels))
        nn.init.uniform_(self.conv6.weight, -1.0/sqrt(self.out_channels), 1.0/sqrt(self.out_channels))
        nn.init.uniform_(self.conv7.weight, -1.0/sqrt(self.in_channels), 1.0/sqrt(self.in_channels))
        nn.init.uniform_(self.conv8.weight, -1.0/sqrt(self.out_channels), 1.0/sqrt(self.out_channels))

        # 添加偏置
        self.bi = nn.Parameter(torch.Tensor(1, self.out_channels, 1, 1), requires_grad=True)
        self.bf = nn.Parameter(torch.Tensor(1, self.out_channels, 1, 1), requires_grad=True)
        self.bc = nn.Parameter(torch.Tensor(1, self.out_channels, 1, 1), requires_grad=True)
        self.bo = nn.Parameter(torch.Tensor(1, self.out_channels, 1, 1), requires_grad=True)

        # 偏置初始化
        nn.init.zeros_(self.bi)
        nn.init.zeros_(self.bf)
        nn.init.zeros_(self.bc)
        nn.init.zeros_(self.bo)

        # 权重
        self.wci = nn.Parameter(torch.Tensor(1, self.out_channels, 1, 1), requires_grad=True)
        self.wcf = nn.Parameter(torch.Tensor(1, self.out_channels, 1, 1), requires_grad=True)
        self.wco = nn.Parameter(torch.Tensor(1, self.out_channels, 1, 1), requires_grad=True)

        # 权重初始化
        nn.init.zeros_(self.wci)
        nn.init.zeros_(self.wcf)
        nn.init.zeros_(self.wco)

    def init_hidden(self, batch_size):
        """
        初始化隐藏层参数。

        :param batch_size: 批尺寸
        :return: （初始隐藏状态，初始模块输出）
        """

        return (torch.zeros(batch_size, self.out_channels, self.shape[0], self.shape[1]).to(device),
                torch.zeros(batch_size, self.out_channels, self.shape[0], self.shape[1]).to(device))

    def forward(self, in_tensor, last_hidden_state, last_cell_out):
        """
        前向传导过程。

        :param in_tensor: 输入张量，即（批尺寸，时间戳，通道数，高度，宽度）
        :param last_hidden_state: 前一个时刻的隐藏状态
        :param last_cell_out: 前一个时刻的模块输出
        :return: （当前时刻的隐藏状态，当前时刻的模块输出）
        """

        # ================================================================================
        # i_t = sigmoid(W_xi * X_t + W_hi * H_(t-1) + W_ci·C_(t-1) + b_i)
        # f_t = sigmoid(W_xf * X_t + W_hf * H_(t-1) + W_cf·C_(t-1) + b_f)
        # C_t = f_t·C_(t-1) + i_t ·tanh(W_xc * X_t + W_hc * H_(t-1) + b_c)
        # o_t = sigmoid(W_xo * X_t + W_ho * H_(t-1) + W_co·C_(t-1) + b_o)
        # H_t = o_t · C_t
        # ================================================================================

        cur_in_gate = sigmoid(
            self.conv1(in_tensor)
            + self.conv2(last_hidden_state)
            + last_cell_out*self.wci
            + self.bi
        )
        cur_forget_gate = sigmoid(
            self.conv3(in_tensor)
            + self.conv4(last_hidden_state)
            + last_cell_out*self.wcf
            + self.bf
        )
        cur_cell_out = (cur_forget_gate*last_cell_out
                        + cur_in_gate*tanh(self.conv5(in_tensor)+self.conv6(last_hidden_state)+self.bc))
        cur_out_gate = sigmoid(
            self.conv7(in_tensor)
            + self.conv8(last_hidden_state)
            + cur_cell_out*self.wco
            + self.bo
        )
        cur_hidden_state = cur_out_gate*tanh(cur_cell_out)

        return cur_hidden_state, cur_cell_out


class DeformConvLSTMNet(nn.Module):
    """
    dConvLSTM网络（输入--dConvLSTM_1--dConvLSTM_2--...--dConvLSTM_n--输出）
    """

    def __init__(self, shape, in_channels, hidden_channels, kernel_size):
        """
        初始化dConvLSTM网络。

        :param shape: 输入张量的形状，即：（高度，宽度）
        :param in_channels: 输入张量的通道数
        :param hidden_channels: 每一dConvLSTM隐藏层的输出通道数
        :param kernel_size: 卷积核的尺寸
        """

        super(DeformConvLSTMNet, self).__init__()

        self.shape = shape
        self.input_channels = [in_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.layers_num = len(hidden_channels)

        layers_list = []

        # 将每一个dConvLSTM层加入到网络中

        for i in range(self.layers_num):
            layers_list.append(
                DeformConvLSTMCell(self.shape, self.input_channels[i], self.hidden_channels[i], self.kernel_size).to(device)
            )

        self.layers_list = nn.ModuleList(layers_list)

    def init_hidden(self, batch_size):
        """
        初始化每一隐藏层的参数。

        :param batch_size: 批尺寸
        :return: 每一隐藏层的初始参数所组成的列表
        """

        init_hidden_list = []

        for i in range(self.layers_num):
            init_hidden_list.append(self.layers_list[i].init_hidden(batch_size))

        return init_hidden_list

    def forward(self, in_tensor, last_hidden_list=None):
        """
        前向传导过程。

        :param in_tensor: 输入张量，（批尺寸，时间戳，通道数，高度，宽度）
        :param last_hidden_list: 每一层中（前一个时刻的隐藏状态，模块的前一个时刻的输出）所组成的列表
        :return: 每一层的输出张量所组成的列表，每一层中（当前时刻的隐藏状态，模块的当前时刻的输出）所组成的列表
        """

        batch_size, seq_len, _, _, _ = in_tensor.size()
        cur_input = in_tensor

        if last_hidden_list is None:
            last_hidden_list = self.init_hidden(batch_size)

        layer_out_list = []
        cur_hidden_list = []

        for i in range(self.layers_num):
            hidden_state, cell_out = last_hidden_list[i]
            out_inner = []

            for j in range(seq_len):
                hidden_state, cell_out = self.layers_list[i](cur_input[:, j, :, :, :].contiguous(), hidden_state, cell_out)
                out_inner.append(hidden_state)

            layer_out = torch.stack(out_inner, 1)
            cur_input = layer_out
            layer_out_list.append(layer_out)
            cur_hidden_list.append([hidden_state, cell_out])

        return layer_out_list, cur_hidden_list


class EncoderForecasterModel(nn.Module):
    """
    编码-预测模型：输入过去的时空序列，预测未来的时空序列。
    """

    def __init__(self, out_seq_len, shape, in_channels, hidden_channels, kernel_size):
        """
        初始化编码-预测模型。

        :param out_seq_len: 输出序列的长度
        :param shape: 输入张量的形状，即：（高度，宽度）
        :param in_channels: 输入张量的通道数
        :param hidden_channels: 每一dConvLSTM隐藏层的输出通道数
        :param kernel_size: 卷积核的尺寸
        """

        super(EncoderForecasterModel, self).__init__()

        self.out_seq_len = out_seq_len
        self.shape = shape
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        hidden_channel_sum = 0
        for i in self.hidden_channels:
            hidden_channel_sum += i
        self.hidden_channel_sum = hidden_channel_sum

        # ==========
        # encoder: 编码器，提取输入序列的特征（hidden_state）
        # forecaster: 预测器，复制编码器的hidden_state，得到输出序列的特征
        # final_conv: 组合输出序列的特征，得到输出序列
        # ==========

        self.encoder = DeformConvLSTMNet(self.shape, self.in_channels, self.hidden_channels, self.kernel_size)
        self.forecaster = DeformConvLSTMNet(self.shape, self.in_channels, self.hidden_channels, self.kernel_size)
        self.final_conv = nn.Conv2d(self.hidden_channel_sum, self.in_channels, (1, 1), 1, (0, 0), True)

        # 初始化权重和偏置
        nn.init.uniform_(self.final_conv.weight, -1.0/sqrt(self.hidden_channel_sum), 1.0/sqrt(self.hidden_channel_sum))
        nn.init.zeros_(self.final_conv.bias)

        self.encoder.to(device)
        self.forecaster.to(device)
        self.final_conv.to(device)

    def forward(self, in_tensor):
        """
        前向传导过程。

        :param in_tensor: 输入张量，（批尺寸，时间戳，通道数，高度，宽度）
        :return: 预测序列
        """

        batch_size, _, in_channels, height, width = in_tensor.size()
        temp_tensor = torch.zeros(batch_size, self.out_seq_len-1, in_channels, height, width).to(device)

        # 提取输入序列的特征
        encoder_layer_out_list, encoder_hidden_list = self.encoder(in_tensor)

        # 得到输出序列的特征
        forecaster_layer_out_list, _ = self.forecaster(temp_tensor, encoder_hidden_list)

        # 拼接预测序列的特征
        forecaster_layer_out_stack = torch.cat(
            [
                torch.cat([i[:, -1:, :, :, :] for i in encoder_layer_out_list], 2),
                torch.cat(forecaster_layer_out_list, 2)
            ],
            1
        )  # ##

        # 得到预测序列

        forecaster_item_tensor_list = []

        for i in range(self.out_seq_len):
            forecaster_item_tensor = self.final_conv(forecaster_layer_out_stack[:, i, :, :, :])
            forecaster_item_tensor = sigmoid(forecaster_item_tensor)
            forecaster_item_tensor_list.append(forecaster_item_tensor)

        forecaster_seq_tensor = torch.stack(forecaster_item_tensor_list, 1)

        return forecaster_seq_tensor


def plot_image(index, in_seq_len, out_seq_len,
               item_num, patch_size, shape, in_seq_batch, target_seq_batch, pred_seq_batch):
    """
    显示输入序列、目标序列和预测序列的图片。

    :param index: 第index个批次
    :param in_seq_len: 输入序列的长度（图片数量）
    :param out_seq_len: 目标序列和预测序列的长度（图片数量）
    :param item_num: 每一批次的样本数量
    :param patch_size: 图片切割的尺寸
    :param shape: 切割后的图片的尺寸，（高度，宽度）
    :param in_seq_batch: 输入图片序列
    :param target_seq_batch: 目标图片序列
    :param pred_seq_batch: 预测图片序列
    :return: 无
    """

    in_seq_batch = in_seq_batch.to(torch.device("cpu"))
    target_seq_batch = target_seq_batch.to(torch.device("cpu"))
    pred_seq_batch = pred_seq_batch.to(torch.device("cpu"))

    # ==========
    # 将一个序列的图片横向拼接成一张图片
    #
    # unit_size: 每张方形图片的尺寸
    # input_image_seq_width: 拼接后输入序列的图片长度
    # target_image_seq_width: 拼接后目标序列的图片长度
    # forecast_image_seq_width: 拼接后预测序列的图片长度
    # ==========

    unit_size = shape[0]*patch_size
    input_image_seq_width = unit_size*in_seq_len
    target_image_seq_width = unit_size*out_seq_len
    forecast_image_seq_width = unit_size*out_seq_len

    to_pil_image = transforms.ToPILImage()

    for i in range(item_num):
        input_image_seq_list = []
        target_image_seq_list = []
        forecast_image_seq_list = []

        # 拼接被切割的图片
        reshape_in_seq_batch = reshape_back(in_seq_batch[i, :, :, :, :], in_seq_len, shape, patch_size)
        reshape_target_seq_batch = reshape_back(target_seq_batch[i, :, :, :, :], out_seq_len, shape, patch_size)
        reshape_pred_seq_batch = reshape_back(pred_seq_batch[i, :, :, :, :], out_seq_len, shape, patch_size)

        # 将张量转化为PIL图片，并加入到相应的列表中
        for j in range(in_seq_len):
            input_image_seq_list.append(to_pil_image(reshape_in_seq_batch[j, :, :, :]*255))
        for j in range(out_seq_len):
            target_image_seq_list.append(to_pil_image(reshape_target_seq_batch[j, :, :, :]*255))
        for j in range(out_seq_len):
            forecast_image_seq_list.append(to_pil_image(reshape_pred_seq_batch[j, :, :, :]*255))

        # 横向拼接输入序列的图片
        input_image_seq = Image.new('L', (input_image_seq_width, unit_size))
        left = 0
        right = unit_size
        for image in input_image_seq_list:
            input_image_seq.paste(image, (left, 0, right, unit_size))
            left += unit_size
            right += unit_size

        # 横向拼接目标序列的图片
        target_image_seq = Image.new('L', (target_image_seq_width, unit_size))
        left = 0
        right = unit_size
        for image in target_image_seq_list:
            target_image_seq.paste(image, (left, 0, right, unit_size))
            left += unit_size
            right += unit_size

        # 横向拼接预测序列的图片
        forecast_image_seq = Image.new('L', (forecast_image_seq_width, unit_size))
        left = 0
        right = unit_size
        for image in forecast_image_seq_list:
            forecast_image_seq.paste(image, (left, 0, right, unit_size))
            left += unit_size
            right += unit_size

        # 显示并保存图片

        pyplot.figure("The %0.4dth test sample" % (index*item_num+i+1), dpi=128)

        pyplot.subplot(3, 1, 1)
        pyplot.title("input image sequence")
        pyplot.imshow(input_image_seq)
        pyplot.axis('off')

        pyplot.subplot(3, 1, 2)
        pyplot.title("target image sequence")
        pyplot.imshow(target_image_seq)
        pyplot.axis('off')

        pyplot.subplot(3, 1, 3)
        pyplot.title("output image sequence")
        pyplot.imshow(forecast_image_seq)
        pyplot.axis('off')

        pyplot.savefig(
            "dconvlstm_k5_moving_mnist_test_result/%0.4dth_test_sample.png" % (index*item_num+i+1),
            dpi=128
        )
        pyplot.show()


seed = 10000  # 随机种子
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 判断是否使用GPU
logger = Logger(log_name='dconvlstm_k5_moving_mnist_log.txt', log_level=1, log_object="moving-mnist").get_log()  # 定义日志

# 设定随机种子
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
numpy.random.seed(seed)
random.seed(seed)


if __name__ == '__main__':

    """
    数据加载及预处理

    ----------
    BATCH_SIZE: 批尺寸
    PARCH_SIZE: 切割图片的大小
    NUM_WORKER: 加载数据的子进程数

    train_path: 训练集路径
    valid_path: 验证集路径
    test_path:  测试集路径

    train_loader: 分批次后的训练集
    valid_loader: 分批次后的验证集
    test_loader:  分批次后的测试集
    """

    BATCH_SIZE = 16
    PARCH_SIZE = 4
    NUM_WORKER = 0

    train_path = "data/moving-mnist-example/moving-mnist-train.npz"
    valid_path = "data/moving-mnist-example/moving-mnist-valid.npz"
    test_path = "data/moving-mnist-example/moving-mnist-test.npz"

    train_loader = moving_mnist_data_loader(train_path, BATCH_SIZE, PARCH_SIZE, True, NUM_WORKER)
    valid_loader = moving_mnist_data_loader(valid_path, BATCH_SIZE, PARCH_SIZE, False, NUM_WORKER)
    test_loader = moving_mnist_data_loader(test_path, BATCH_SIZE, PARCH_SIZE, False, NUM_WORKER)

    logger.info("数据加载完成！")

    """
    定义编码-预测网络

    ----------
    IN_SEQ_LEN: 输入的图片序列的长度
    OUT_SEQ_LEN: 输出的预测序列的长度
    SHAPE: 图片的形状（高度，宽度）
    IN_CHANNELS: 输入张量的通道数
    HIDDEN_CHANNELS: 隐藏层的输出通道数
    KERNEL_SIZE: 卷积核的尺寸

    net: 根据上述参数所定义的编码-预测网络
    """

    IN_SEQ_LEN = 10
    OUT_SEQ_LEN = 10
    SHAPE = (int(64/PARCH_SIZE), int(64/PARCH_SIZE))
    IN_CHANNELS = PARCH_SIZE*PARCH_SIZE
    HIDDEN_CHANNELS = [128, 64, 64]
    KERNEL_SIZE = (5, 5)

    net = EncoderForecasterModel(OUT_SEQ_LEN, SHAPE, IN_CHANNELS, HIDDEN_CHANNELS, KERNEL_SIZE).to(device)

    logger.info("编码-预测网络定义完成！")
    logger.debug("参数：")

    # 打印参数名称和尺寸
    for name, param in net.named_parameters():
        if param.requires_grad:
            logger.debug(
                "{para_name}: {para_size}, {para_num}".format(
                    para_name=name,
                    para_size=str(param.shape),
                    para_num=str(param.numel())
                )
            )

    """
    定义损失函数和优化器

    ----------
    LEARNING_RATE: 学习率
    MOMENTUM: 动量因子
    DECAY_RATE: 衰减速率

    criterion: 使用交叉熵作为损失函数
    optimizer: 使用RMSprop方法进行优化
    """

    LEARNING_RATE = 1e-3
    DECAY_RATE = 0.9

    criterion = nn.BCELoss(reduction='sum')
    optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE, alpha=DECAY_RATE, eps=1e-6)

    logger.info("损失函数和优化器定义完成！")
    logger.info("LEARNING_RATE: %f" % LEARNING_RATE)
    logger.info("DECAY_RATE: %f" % DECAY_RATE)

    """
    训练网络

    ----------
    MAX_EPOCH: 最大遍历次数
    NUM_THREAD: 线程数
    MAX_NO_BETTER_EPOCH: 所允许的验证集误差不降反升的最大遍历次数

    best_valid_running_loss: 目前最小的验证集误差
    no_better_validation_step: 验证集误差不降反升的次数
    """

    MAX_EPOCH = 50
    NUM_THREAD = 16
    MAX_NO_BETTER_EPOCH = 10

    best_valid_running_loss = 1e9
    no_better_validation_step = 0

    logger.info("开始训练模型！")

    torch.set_num_threads(NUM_THREAD)
    for epoch in range(MAX_EPOCH):
        start = time.time()

        # 利用训练集训练模型

        train_running_loss = 0.0

        for batch_index, train_data_batch in enumerate(train_loader):
            train_in_tensor, train_target_tensor = train_data_batch  # 输入训练数据
            train_in_tensor, train_target_tensor = train_in_tensor.to(device), train_target_tensor.to(device)

            sample_num = train_in_tensor.__len__()

            optimizer.zero_grad()  # 梯度清零
            train_out_tensor = net(train_in_tensor)  # 前向传递
            train_loss = criterion(train_out_tensor, train_target_tensor)  # 计算误差
            train_loss.backward()  # 反向传导
            optimizer.step()  # 更新参数

            train_running_loss += float(train_loss)
            logger.info('[%d, %5d] train loss: %.3f' % (epoch+1, batch_index+1, float(train_loss)))

        train_running_loss /= train_loader.dataset.__len__()
        logger.info('[%d] average train loss: %.3f' % (epoch+1, train_running_loss))

        # 检查验证集中误差变化情况

        cur_valid_running_loss = 0.0

        for _, valid_data_batch in enumerate(valid_loader):
            with torch.no_grad():
                valid_in_tensor, valid_target_tensor = valid_data_batch
                valid_in_tensor, valid_target_tensor = valid_in_tensor.to(device), valid_target_tensor.to(device)

                valid_out_tensor = net(valid_in_tensor)

                valid_loss = criterion(valid_out_tensor, valid_target_tensor).sum()
                cur_valid_running_loss += float(valid_loss)

        cur_valid_running_loss /= valid_loader.dataset.__len__()
        logger.info('[%d] valid loss: %.3f' % (epoch+1, cur_valid_running_loss))

        end = time.time()
        time_elapsed = end-start
        logger.info('每epoch运行时间：{:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed % 60))

        # early-stopping
        if cur_valid_running_loss > best_valid_running_loss:
            no_better_validation_step += 1
            if no_better_validation_step > MAX_NO_BETTER_EPOCH:
                break
        else:
            best_valid_running_loss = cur_valid_running_loss
            no_better_validation_step = 0
            torch.save(net, 'dconvlstm_k5_moving_mnist_best_model.pkl')

    logger.info("训练完成！")

    # 加载验证集上效果最好的模型
    if torch.cuda.is_available():
        final_net = torch.load('dconvlstm_k5_moving_mnist_best_model.pkl')
    else:
        final_net = torch.load('dconvlstm_k5_moving_mnist_best_model.pkl', map_location=lambda storage, loc: storage)

    # 检查模型在测试集中的表现

    final_test_loss = 0.0
    test_out = []

    for _, test_data_batch in enumerate(test_loader):
        with torch.no_grad():
            test_in_tensor, test_target_tensor = test_data_batch
            test_in_tensor, test_target_tensor = test_in_tensor.to(device), test_target_tensor.to(device)

            test_out_tensor = final_net(test_in_tensor)
            test_out.append(test_out_tensor)

            test_loss = criterion(test_out_tensor, test_target_tensor).sum()
            final_test_loss += float(test_loss)

    final_test_loss /= test_loader.dataset.__len__()
    logger.info("test loss: %.3f" % final_test_loss)

    # 绘制测试集的预测结果

    # for batch_index, test_data_batch in enumerate(test_loader):
    #     test_in_tensor, test_target_tensor = test_data_batch
    #     test_out_tensor = test_out[batch_index]

    #     plot_image(
    #         batch_index,
    #         IN_SEQ_LEN,
    #         OUT_SEQ_LEN,
    #         BATCH_SIZE,
    #         PARCH_SIZE,
    #         SHAPE,
    #         test_in_tensor,
    #         test_target_tensor,
    #         test_out_tensor
    #     )
