import numpy
import torch
import random
import logging
import mxnet as mx
import mxnet.ndarray as nd
import os
from varflow import VarFlowFactory
from nowcasting.config import cfg
from nowcasting.hko_evaluation import HKOEvaluation, pixel_to_dBZ, dBZ_to_pixel
from nowcasting.hko_benchmark import HKOBenchmarkEnv
from nowcasting.hko_iterator import precompute_mask
from nowcasting.helpers.visualization import save_hko_gif
from nowcasting.utils import logging_config


class NonLinearRoverTransform(object):
    def __init__(self, Zc=33, sharpness=4):
        self.Zc = float(Zc)
        self.sharpness = float(sharpness)

    def transform(self, img):
        dbz_img = pixel_to_dBZ(img)
        dbz_lower = pixel_to_dBZ(0.0)
        dbz_upper = pixel_to_dBZ(1.0)
        transformed_lower = numpy.arctan((dbz_lower - self.Zc) / self.sharpness)
        transformed_upper = numpy.arctan((dbz_upper - self.Zc) / self.sharpness)
        transformed_img = numpy.arctan((dbz_img - self.Zc) / self.sharpness)
        transformed_img = (transformed_img - transformed_lower) / (transformed_upper - transformed_lower)
        return transformed_img

    def rev_transform(self, transformed_img):
        dbz_lower = pixel_to_dBZ(0.0)
        dbz_upper = pixel_to_dBZ(1.0)
        transformed_lower = numpy.arctan((dbz_lower - self.Zc) / self.sharpness)
        transformed_upper = numpy.arctan((dbz_upper - self.Zc) / self.sharpness)
        img = transformed_img * (transformed_upper - transformed_lower) + transformed_lower
        img = numpy.tan(img) * self.sharpness + self.Zc
        img = dBZ_to_pixel(dBZ_img=img)
        return img


def advection(im, flow):
    """

    Parameters
    ----------
    im : nd.NDArray
        Shape: (batch_size, C, H, W)
    flow : nd.NDArray
        Shape: (batch_size, 2, H, W)
    Returns
    -------
    new_im : nd.NDArray
    """
    grid = nd.GridGenerator(-flow, transform_type="warp")
    new_im = nd.BilinearSampler(im, grid)

    return new_im


def run(pd_path=cfg.HKO_PD.RAINY_TEST, mode="fixed", nonlinear_transform=True):
    transformer = NonLinearRoverTransform()
    flow_factory = VarFlowFactory(max_level=6, start_level=0, n1=2, n2=2, rho=1.5, alpha=2000, sigma=4.5)

    if nonlinear_transform:
        base_dir = os.path.join('hko7_benchmark', 'rover-nonlinear')
    else:
        base_dir = os.path.join('hko7_benchmark', 'rover-linear')

    logging_config(base_dir)

    counter = 0
    env = HKOBenchmarkEnv(pd_path=pd_path, save_dir=base_dir, mode=mode)
    while not env.done:
        in_frame_dat, in_datetime_clips, out_datetime_clips, begin_new_episode, need_upload_prediction = env.get_observation(batch_size=1)
        if need_upload_prediction:
            counter += 1
            prediction = numpy.zeros(shape=(cfg.HKO.BENCHMARK.OUT_LEN,) + in_frame_dat.shape[1:], dtype=numpy.float32)
            I1 = in_frame_dat[-2, :, 0, :, :]
            I2 = in_frame_dat[-1, :, 0, :, :]
            mask_I1 = precompute_mask(I1)
            mask_I2 = precompute_mask(I2)
            I1 = I1 * mask_I1
            I2 = I2 * mask_I2
            if nonlinear_transform:
                I1 = transformer.transform(I1)
                I2 = transformer.transform(I2)
            flow = flow_factory.batch_calc_flow(I1=I1, I2=I2)

            init_im = nd.array(I2.reshape((I2.shape[0], 1, I2.shape[1], I2.shape[2])), ctx=mx.gpu())
            # init_im_tensor = torch.Tensor(I2.reshape((I2.shape[0], 1, I2.shape[1], I2.shape[2]))).to(device)
            nd_flow = nd.array(numpy.concatenate((flow[:, :1, :, :], -flow[:, 1:, :, :]), axis=1), ctx=mx.gpu())
            # flow_tensor = torch.Tensor(numpy.concatenate((flow[:, :1, :, :], -flow[:, 1:, :, :]), axis=1)).to(device)
            nd_pred_im = nd.zeros(shape=prediction.shape)
            # pred_im_tensor = torch.zeros(prediction.shape)
            for i in range(cfg.HKO.BENCHMARK.OUT_LEN):
                new_im = advection(init_im, nd_flow)
                nd_pred_im[i][:] = new_im
                init_im[:] = new_im
                # new_im = advection(init_im_tensor, flow_tensor)
                # pred_im_tensor[i][:] = new_im
                # init_im_tensor[:] = new_im
            prediction = nd_pred_im.asnumpy()
            # prediction = pred_im_tensor.numpy()

            if nonlinear_transform:
                prediction = transformer.rev_transform(prediction)

            env.upload_prediction(prediction=prediction)

            if counter % 10 == 0:
                # save_hko_gif(in_frame_dat[:, 0, 0, :, :], save_path=os.path.join(base_dir, 'in.gif'))
                # save_hko_gif(prediction[:, 0, 0, :, :], save_path=os.path.join(base_dir, 'pred.gif'))
                env.print_stat_readable()
                # import matplotlib.pyplot as plt
                # Q = plt.quiver(flow[1, 0, ::10, ::10], flow[1, 1, ::10, ::10])
                # plt.gca().invert_yaxis()
                # plt.show()
                # ch = raw_input()
    env.save_eval()
    _, _, csi, hss, _, _, _, b_mse, b_mae, _ = env._all_eval.calculate_stat()
    csi = csi.mean(axis=0)
    hss = hss.mean(axis=0)
    b_mse = b_mse.mean()
    b_mae = b_mae.mean()
    loss = b_mse * 0.5 + b_mae * 0.5
    logger.info(
        'Final:\n\tcsi: %s\n\thss: %s\n\tb-mse: %.3f\n\tb-mae: %.3f\n\ttest loss: %.3f'
        % (
            ' '.join(map(lambda x: ('%.3f' % x), csi)),
            ' '.join(map(lambda x: ('%.3f' % x), hss)),
            b_mse,
            b_mae,
            loss
        )
    )


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
logger = Logger(log_name='rover_log.txt', log_level=1, log_object="hko-7").get_log()  # 定义日志

# 设定随机种子
numpy.random.seed(seed)
random.seed(seed)

if __name__ == '__main__':
    with torch.no_grad():
        logger.info('Linear ROVER Validation Start!')
        run(cfg.HKO_PD.RAINY_VALID, mode="fixed", nonlinear_transform=False)
        logger.info('Linear ROVER Validation Finish!')

        logger.info('Linear ROVER Test Start!')
        run(cfg.HKO_PD.RAINY_TEST, mode="fixed", nonlinear_transform=False)
        logger.info('Linear ROVER Test Finish!')

        logger.info('Nonlinear ROVER Validation Start!')
        run(cfg.HKO_PD.RAINY_VALID, mode="fixed", nonlinear_transform=True)
        logger.info('Nonlinear ROVER Validation Finish!')

        logger.info('Nonlinear ROVER Test Start!')
        run(cfg.HKO_PD.RAINY_TEST, mode="fixed", nonlinear_transform=True)
        logger.info('Nonlinear ROVER Test Finish!')
