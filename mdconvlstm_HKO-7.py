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
from nowcasting.hko_iterator import HKOIterator
from nowcasting.config import cfg
from nowcasting.hko_benchmark import HKOBenchmarkEnv


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


def reshape_patch(in_tensor, batch_size, seq_len, raw_shape, patch_size):
    """
    按切割尺寸切割图片。
    切割前：将单通道的图片(1, raw height, raw width)
    切割后：多通道图片(patch size * patch size, raw height / patch size, raw width / patch size)

    :param in_tensor: 待切割的单通道图片
    :param batch_size: 批尺寸
    :param seq_len: 图片的序列长度
    :param raw_shape: 图片的原始形状
    :param patch_size: 切割尺寸
    :return: 切割后的多通道图片
    """

    ret = in_tensor.reshape(batch_size, seq_len, int(raw_shape[0]/patch_size), patch_size, int(raw_shape[1]/patch_size), patch_size)
    ret = ret.permute(0, 1, 3, 5, 2, 4)
    ret = ret.reshape(batch_size, seq_len, patch_size*patch_size, int(raw_shape[0]/patch_size), int(raw_shape[1]/patch_size))

    return ret


def reshape_back(patch_tensor, batch_size, seq_len, patch_shape, patch_size):
    """
    恢复被切割的图片。

    :param patch_tensor: 切割后的多通道图片
    :param batch_size: 批尺寸
    :param seq_len: 图片的序列长度
    :param patch_shape: 切割后的图片形状
    :param patch_size: 切割尺寸
    :return: 单通道原始图片
    """

    ret = patch_tensor.reshape(batch_size, seq_len, patch_size, patch_size, patch_shape[0], patch_shape[1])
    ret = ret.permute(0, 1, 4, 2, 5, 3)
    ret = ret.reshape(batch_size, seq_len, 1, patch_shape[0]*patch_size, patch_shape[1]*patch_size)

    return ret


class ModulatedDeformConvLSTMCell(nn.Module):
    """
        单个mdConvLSTM模块（输入--隐藏层--输出）
    """

    def __init__(self, shape, in_channels, out_channels, kernel_size):
        """
        初始化单个mdconvlstm模块。

        :param shape: 输入张量的形状，即：（高度，宽度）
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核的尺寸
        """

        super(ModulatedDeformConvLSTMCell, self).__init__()

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

        self.conv1 = ModulatedDeformConvPack(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=False
        )
        self.conv2 = ModulatedDeformConvPack(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=False
        )
        self.conv3 = ModulatedDeformConvPack(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=False
        )
        self.conv4 = ModulatedDeformConvPack(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=False
        )
        self.conv5 = ModulatedDeformConvPack(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=False
        )
        self.conv6 = ModulatedDeformConvPack(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=False
        )
        self.conv7 = ModulatedDeformConvPack(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=False
        )
        self.conv8 = ModulatedDeformConvPack(
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


class ModulatedDeformConvLSTMNet(nn.Module):
    """
    mdConvLSTM网络（输入--mdConvLSTM_1--mdConvLSTM_2--...--mdConvLSTM_n--输出）
    """

    def __init__(self, shape, in_channels, hidden_channels, kernel_size):
        """
        初始化mdConvLSTM网络。

        :param shape: 输入张量的形状，即：（高度，宽度）
        :param in_channels: 输入张量的通道数
        :param hidden_channels: 每一mdConvLSTM隐藏层的输出通道数
        :param kernel_size: 卷积核的尺寸
        """

        super(ModulatedDeformConvLSTMNet, self).__init__()

        self.shape = shape
        self.input_channels = [in_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.layers_num = len(hidden_channels)

        layers_list = []

        # 将每一个mdConvLSTM层加入到网络中

        for i in range(self.layers_num):
            layers_list.append(
                ModulatedDeformConvLSTMCell(self.shape, self.input_channels[i], self.hidden_channels[i], self.kernel_size).to(device)
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
        :param hidden_channels: 每一mdConvLSTM隐藏层的输出通道数
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

        self.encoder = ModulatedDeformConvLSTMNet(self.shape, self.in_channels, self.hidden_channels, self.kernel_size)
        self.forecaster = ModulatedDeformConvLSTMNet(self.shape, self.in_channels, self.hidden_channels, self.kernel_size)
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


class BalanceLoss(nn.Module):
    def __init__(self):
        super(BalanceLoss, self).__init__()

        self.balancing_weights = cfg.HKO.EVALUATION.BALANCING_WEIGHTS
        self.thresholds = [rainfall_to_pixel(ele) for ele in cfg.HKO.EVALUATION.THRESHOLDS]

    def forward(self, pred, target, mask):
        weights = torch.ones(pred.size())*self.balancing_weights[0]
        weights = weights.to(device)
        for i, threshold in enumerate(self.thresholds):
            weights = weights+(self.balancing_weights[i+1]-self.balancing_weights[i])*((target >= threshold).type(torch.FloatTensor).to(device))
        weights = weights*mask

        b_mse = torch.mean(torch.sum(weights*torch.pow(pred-target, 2), dim=(2, 3, 4)))
        b_mae = torch.mean(torch.sum(weights*torch.abs(pred-target), dim=(2, 3, 4)))

        loss = b_mse*0.5+b_mae*0.5

        return b_mse, b_mae, loss


def rainfall_to_pixel(rainfall_intensity):
    """Convert the rainfall intensity to pixel values

    Parameters
    ----------
    rainfall_intensity : np.ndarray
    Returns
    -------
    pixel_vals : np.ndarray
    """

    a = cfg.HKO.EVALUATION.ZR.a
    b = cfg.HKO.EVALUATION.ZR.b
    dBR = numpy.log10(rainfall_intensity)*10.0
    dBZ = dBR*b+10.0*numpy.log10(a)
    pixel_vals = (dBZ+10.0)/70.0
    return pixel_vals


seed = 10000  # 随机种子
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 判断是否使用GPU
logger = Logger(log_name='mdconvlstm_hko_7_log.txt', log_level=1, log_object="hko-7").get_log()  # 定义日志

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
    SEQ_LEN: 图片序列的总长度（包括输入序列和输出序列）

    train_hko_iter: 训练集的迭代器
    """

    BATCH_SIZE = 2
    SEQ_LEN = cfg.MODEL.IN_LEN + cfg.MODEL.OUT_LEN

    train_hko_iter = HKOIterator(pd_path=cfg.HKO_PD.RAINY_TRAIN, sample_mode="random", seq_len=SEQ_LEN)

    logger.info("数据加载完成！")

    """
    定义编码-预测网络

    ----------
    PATCH_SIZE: 切割尺寸
    IN_SEQ_LEN: 输入的图片序列的长度
    OUT_SEQ_LEN: 输出的预测序列的长度
    RAW_SHAPE: 图片的原始形状（高度，宽度）
    PATCH_SHAPE: 图片切割后的形状（高度/PATCH_SIZE，宽度/PATCH_SIZE）
    IN_CHANNELS: 输入张量的通道数
    HIDDEN_CHANNELS: 隐藏层的输出通道数
    KERNEL_SIZE: 卷积核的尺寸

    net: 根据上述参数所定义的编码-预测网络
    """

    PATCH_SIZE = 4
    IN_SEQ_LEN = cfg.MODEL.IN_LEN
    OUT_SEQ_LEN = cfg.MODEL.OUT_LEN
    RAW_SHAPE = (cfg.HKO.ITERATOR.HEIGHT, cfg.HKO.ITERATOR.WIDTH)
    PATCH_SHAPE = (int(cfg.HKO.ITERATOR.HEIGHT/PATCH_SIZE), int(cfg.HKO.ITERATOR.WIDTH/PATCH_SIZE))
    IN_CHANNELS = PATCH_SIZE*PATCH_SIZE
    HIDDEN_CHANNELS = [64, 32, 32]
    KERNEL_SIZE = (3, 3)

    net = EncoderForecasterModel(OUT_SEQ_LEN, PATCH_SHAPE, IN_CHANNELS, HIDDEN_CHANNELS, KERNEL_SIZE).to(device)

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

    criterion: 损失函数
    optimizer: 使用RMSprop方法进行优化
    """

    LEARNING_RATE = 1e-3
    DECAY_RATE = 0.9

    criterion = BalanceLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE, alpha=DECAY_RATE, eps=1e-6)

    logger.info("损失函数和优化器定义完成！")
    logger.info("LEARNING_RATE: %f" % LEARNING_RATE)
    logger.info("DECAY_RATE: %f" % DECAY_RATE)

    """
    训练网络

    ----------
    MAX_ITERATION: 最大迭代次数
    VALID_ITERATION: 进行验证的迭代次数间隔
    NUM_THREAD: 线程数
    MAX_NO_BETTER_EPOCH: 所允许的验证集误差不降反升的最大遍历次数

    best_valid_running_loss: 目前最小的验证集误差
    no_better_validation_step: 验证集误差不降反升的次数
    """

    MAX_ITERATION = 80000
    VALID_ITERATION = 5000
    NUM_THREAD = 16
    MAX_NO_BETTER_EPOCH = 10

    best_valid_running_loss = 1e9
    no_better_validation_step = 0

    logger.info("开始训练模型！")

    count = 0
    torch.set_num_threads(NUM_THREAD)
    for iteration in range(MAX_ITERATION):

        # 利用训练集训练模型

        train_batch, train_mask, train_datetime_clips, train_new_start = train_hko_iter.sample(batch_size=BATCH_SIZE)
        train_batch = torch.from_numpy(train_batch/255.0).permute(1, 0, 2, 3, 4).type(torch.FloatTensor).to(device)
        train_mask = torch.from_numpy(train_mask.astype('float32')).permute(1, 0, 2, 3, 4).to(device)
        train_in_tensor = train_batch[:, 0:IN_SEQ_LEN, :, :, :]
        train_target_tensor = train_batch[:, IN_SEQ_LEN:(IN_SEQ_LEN+OUT_SEQ_LEN), :, :, :]
        train_mask_tensor = train_mask[:, IN_SEQ_LEN:(IN_SEQ_LEN+OUT_SEQ_LEN), :, :, :]

        optimizer.zero_grad()
        train_in_tensor = reshape_patch(train_in_tensor, BATCH_SIZE, IN_SEQ_LEN, RAW_SHAPE, PATCH_SIZE)
        train_out_tensor = net(train_in_tensor)
        train_out_tensor = reshape_back(train_out_tensor, BATCH_SIZE, OUT_SEQ_LEN, PATCH_SHAPE, PATCH_SIZE)
        train_b_mse, train_b_mae, train_loss = criterion(train_out_tensor, train_target_tensor, train_mask_tensor)
        train_loss.backward()
        optimizer.step()

        logger.info(
            '[%d] iterations, b-mse: %.3f, b-mae: %.3f, train loss: %.3f'
            % (iteration+1, float(train_b_mse), float(train_b_mae), float(train_loss))
        )

        # 检查验证集中误差变化情况
        if(iteration+1)%VALID_ITERATION == 0:
            count += 1
            with torch.no_grad():
                valid_env = HKOBenchmarkEnv(pd_path=cfg.HKO_PD.RAINY_VALID, save_dir="mdconvlstm_hko7_valid", mode="fixed")
                while not valid_env.done:
                    valid_in_frame_dat, _, _, _, valid_need_upload_prediction = valid_env.get_observation(1)
                    valid_in_tensor = torch.from_numpy(valid_in_frame_dat).permute(1, 0, 2, 3, 4).type(torch.FloatTensor).to(device)

                    if valid_need_upload_prediction:
                        valid_in_tensor = reshape_patch(valid_in_tensor, 1, IN_SEQ_LEN, RAW_SHAPE, PATCH_SIZE)
                        valid_out_tensor = net(valid_in_tensor)
                        valid_out_tensor = reshape_back(valid_out_tensor, 1, OUT_SEQ_LEN, PATCH_SHAPE, PATCH_SIZE)
                        valid_env.upload_prediction(valid_out_tensor.permute(1, 0, 2, 3, 4).to(torch.device("cpu")).numpy())

                valid_env.save_eval()
                _, _, valid_csi, valid_hss, _, _, _, valid_b_mse, valid_b_mae, _ = valid_env._all_eval.calculate_stat()
                valid_csi = valid_csi.mean(axis =0)
                valid_hss = valid_hss.mean(axis=0)
                valid_b_mse = valid_b_mse.mean()
                valid_b_mae = valid_b_mae.mean()
                valid_loss = valid_b_mse*0.5+valid_b_mae*0.5
                logger.info(
                    '[%d]th valid:\n\tcsi: %s\n\thss: %s\n\tb-mse: %.3f\n\tb-mae: %.3f\n\tvalid loss: %.3f\n'
                    % (
                        count,
                        ' '.join(map(lambda x:('%.3f'%x), valid_csi)),
                        ' '.join(map(lambda x:('%.3f'%x), valid_hss)),
                        valid_b_mse,
                        valid_b_mae,
                        valid_loss
                    )
                )

                # early-stopping
                if valid_loss > best_valid_running_loss:
                    no_better_validation_step += 1
                    if no_better_validation_step >= MAX_NO_BETTER_EPOCH:
                        break
                else:
                    best_valid_running_loss = valid_loss
                    no_better_validation_step = 0
                    torch.save(net, 'mdconvlstm_hko_7_best_model.pkl')

    logger.info("训练完成！")

    # 加载验证集上效果最好的模型
    if torch.cuda.is_available():
        final_net = torch.load('mdconvlstm_hko_7_best_model.pkl')
    else:
        final_net = torch.load('mdconvlstm_hko_7_best_model.pkl', map_location=lambda storage, loc: storage)

    # 检查模型在测试集中的表现

    with torch.no_grad():
        test_env = HKOBenchmarkEnv(pd_path=cfg.HKO_PD.RAINY_TEST, save_dir="mdconvlstm_hko7_test", mode="fixed")
        while not test_env.done:
            test_in_frame_dat, _, _, _, test_need_upload_prediction = test_env.get_observation(1)
            test_in_tensor = torch.from_numpy(test_in_frame_dat).permute(1, 0, 2, 3, 4).type(torch.FloatTensor).to(device)

            if test_need_upload_prediction:
                test_in_tensor = reshape_patch(test_in_tensor, 1, IN_SEQ_LEN, RAW_SHAPE, PATCH_SIZE)
                test_out_tensor = final_net(test_in_tensor)
                test_out_tensor = reshape_back(test_out_tensor, 1, OUT_SEQ_LEN, PATCH_SHAPE, PATCH_SIZE)
                test_env.upload_prediction(test_out_tensor.permute(1, 0, 2, 3, 4).to(torch.device("cpu")).numpy())

        test_env.save_eval()
        _, _, test_csi, test_hss, _, _, _, test_b_mse, test_b_mae, _ = test_env._all_eval.calculate_stat()
        test_csi = test_csi.mean(axis=0)
        test_hss = test_hss.mean(axis=0)
        test_b_mse = test_b_mse.mean()
        test_b_mae = test_b_mae.mean()
        test_loss = test_b_mse*0.5+test_b_mae*0.5
        logger.info(
            'test:\n\tcsi: %s\n\thss: %s\n\tb-mse: %.3f\n\tb-mae: %.3f\n\ttest loss: %.3f\n'
            % (
                ' '.join(map(lambda x:('%.3f'%x), test_csi)),
                ' '.join(map(lambda x:('%.3f'%x), test_hss)),
                test_b_mse,
                test_b_mae,
                test_loss
            )
        )
