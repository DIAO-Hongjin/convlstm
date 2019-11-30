__author__ = 'DIAO Hongjin'


import torch
import numpy
import time
import logging
import random
from numpy import sqrt
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


seed = 10000  # 随机种子
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # 判断是否使用GPU
logger = Logger(log_name='last_frame_log.txt', log_level=1, log_object="hko-7").get_log()  # 定义日志

# 设定随机种子
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
numpy.random.seed(seed)
random.seed(seed)


if __name__ == '__main__':
    
    logger.info('Last Frame Start!')

    with torch.no_grad():
        test_env = HKOBenchmarkEnv(pd_path=cfg.HKO_PD.RAINY_TEST, save_dir="last_frame_test", mode="fixed")
        while not test_env.done:
            test_in_frame_dat, _, _, _, test_need_upload_prediction = test_env.get_observation(1)
            test_in_tensor = torch.from_numpy(test_in_frame_dat).permute(1, 0, 2, 3, 4).type(torch.FloatTensor).to(device)

            if test_need_upload_prediction:
                logger.info('The Test Sample.')
                test_out_tensor = test_in_tensor[:, cfg.MODEL.IN_LEN-1, :, :, :]
                test_out_tensor = test_out_tensor.expand(1, cfg.MODEL.OUT_LEN, 1, cfg.HKO.ITERATOR.HEIGHT, cfg.HKO.ITERATOR.WIDTH)
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
