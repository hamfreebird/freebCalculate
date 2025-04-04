import cv2
import numpy as np
from tqdm import tqdm


class VideoInterpolator:
    def __init__(self, input_path, output_path, interp_factor=1, method='linear',
                 flow_params=None, show_progress=True, use_gpu=False):
        """
        视频插帧类
        :param input_path: 输入视频路径
        :param output_path: 输出视频路径
        :param interp_factor: 插帧倍数（每两个原始帧之间插入的帧数）
        :param method: 插值方法（当前支持'linear'）
        """
        self.input_path = input_path
        self.output_path = output_path
        self.interp_factor = interp_factor
        self.method = method
        self.use_gpu = use_gpu

        # 初始化硬件加速环境
        self._init_hardware_acceleration()

        # 视频参数
        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            raise ValueError("无法打开视频文件")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.new_fps = self.original_fps * (interp_factor + 1)

        # 初始化光流法需要的变量
        self.prev_gray = None
        self.next_gray = None

        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.new_fps,
            (self.width, self.height)
        )

        # 光流法参数设置
        self.flow_params = flow_params or {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }

        self.show_progress = show_progress  # 进度显示开关
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数

    def _init_hardware_acceleration(self):
        """初始化Intel GPU加速环境"""
        if self.use_gpu:
            try:
                # 检查OpenVINO环境可用性
                from openvino.runtime import Core
                ie = Core()
                self.gpu_devices = ie.available_devices

                if 'GPU' not in self.gpu_devices:
                    raise RuntimeError("未检测到兼容的Intel GPU设备")

                # 设置OpenCL环境（适用于Arc系列）
                cv2.ocl.setUseOpenCL(True)
                if not cv2.ocl.haveOpenCL():
                    raise RuntimeError("OpenCL不可用")

                print(f"已启用Intel GPU加速（设备：{ie.get_property('GPU', 'FULL_DEVICE_NAME')}）")

                # 初始化GPU光流计算器
                self.gpu_flow = cv2.optflow.createOptFlow_Farneback_GPU(
                    pyrScale=self.flow_params['pyr_scale'],
                    numLevels=self.flow_params['levels'],
                    winSize=self.flow_params['winsize'],
                    numIters=self.flow_params['iterations'],
                    polyN=self.flow_params['poly_n'],
                    polySigma=self.flow_params['poly_sigma'],
                    flags=self.flow_params['flags']
                )

            except ImportError:
                raise RuntimeError("需要安装OpenVINO工具套件")
            except Exception as e:
                try:
                    # 使用DIS光流算法代替Farneback
                    self.gpu_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
                    self.gpu_flow.setUseSpatialPropagation(True)

                    print("已启用DIS光流GPU加速")
                except Exception as e:
                    print(f"GPU初始化失败: {str(e)}，回退CPU模式")
                    self.use_gpu = False

    def _calculate_optical_flow(self, frame1, frame2):
        """计算双向光流"""
        if self.use_gpu:
            # 转换图像到GPU内存
            gpu_frame1 = cv2.UMat(frame1)
            gpu_frame2 = cv2.UMat(frame2)
            gpu_gray1 = cv2.cvtColor(gpu_frame1, cv2.COLOR_BGR2GRAY)
            gpu_gray2 = cv2.cvtColor(gpu_frame2, cv2.COLOR_BGR2GRAY)

            # GPU加速光流计算
            forward_flow = self.gpu_flow.calc(gpu_gray1, gpu_gray2, None)
            backward_flow = self.gpu_flow.calc(gpu_gray2, gpu_gray1, None)

            return forward_flow.get(), backward_flow.get()
        else:
            # 原有CPU计算代码
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            forward_flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, **self.flow_params)
            backward_flow = cv2.calcOpticalFlowFarneback(
                gray2, gray1, None, **self.flow_params)

            return forward_flow, backward_flow

    def _warp_frame(self, frame, flow):
        """根据光流场变形图像"""
        if self.use_gpu:
            # 使用GPU加速的remap
            gpu_frame = cv2.UMat(frame)
            h, w = flow.shape[:2]
            map_x = np.tile(np.arange(w), (h, 1)).astype(np.float32)
            map_y = np.tile(np.arange(h), (w, 1)).T.astype(np.float32)
            map_x += flow[..., 0]
            map_y += flow[..., 1]

            gpu_warped = cv2.remap(gpu_frame, map_x, map_y, cv2.INTER_LINEAR)
            return gpu_warped.get()
        else:
            h, w = flow.shape[:2]
            flow_map = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1)
            flow_map = flow_map.astype(np.float32) + flow
            return cv2.remap(
                frame,
                flow_map[..., 0],
                flow_map[..., 1],
                interpolation=cv2.INTER_LINEAR
            )

    def _optical_flow_interpolation(self, prev_frame, next_frame, alpha):
        """基于光流的插值方法"""
        # 计算双向光流
        forward_flow, backward_flow = self._calculate_optical_flow(prev_frame, next_frame)

        # 计算中间时刻的光流
        forward_flow_t = forward_flow * alpha
        backward_flow_t = backward_flow * (1 - alpha)

        # 变形前后帧
        warped_forward = self._warp_frame(prev_frame, forward_flow_t)
        warped_backward = self._warp_frame(next_frame, backward_flow_t)

        # 融合两帧结果
        return cv2.addWeighted(warped_forward, 0.5, warped_backward, 0.5, 0)

    def _generate_inter_frames(self, prev_frame, next_frame):
        """生成中间帧序列（更新版）"""
        inter_frames = []
        for i in range(1, self.interp_factor + 1):
            alpha = i / (self.interp_factor + 1)
            if self.method == 'linear':
                frame = self._linear_interpolation(prev_frame, next_frame, alpha)
            elif self.method == 'optical_flow':
                frame = self._optical_flow_interpolation(prev_frame, next_frame, alpha)
            inter_frames.append(frame)
        return inter_frames

    def _linear_interpolation(self, frame1, frame2, alpha):
        """线性插值生成中间帧"""
        return cv2.addWeighted(frame1, 1-alpha, frame2, alpha, 0)

    def process(self):
        """处理视频并生成插帧结果"""
        ret, prev_frame = self.cap.read()
        if not ret:
            raise ValueError("无法读取视频帧")

        # 初始化进度条
        if self.show_progress:
            pbar = tqdm(
                total=self.total_frames,
                desc="处理进度",
                unit="frame",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )

        # 写入第一帧
        self.writer.write(prev_frame)
        processed_frames = 1  # 已处理帧数计数器

        while True:
            ret, next_frame = self.cap.read()
            if not ret:
                break

            # 生成并写入中间帧
            inter_frames = self._generate_inter_frames(prev_frame, next_frame)
            for frame in inter_frames:
                self.writer.write(frame)

            # 写入原始帧
            self.writer.write(next_frame)
            prev_frame = next_frame
            processed_frames += 1

            # 更新进度条
            if self.show_progress:
                pbar.update(1)
                pbar.set_postfix({
                    "当前方法": self.method,
                    "插帧倍数": self.interp_factor,
                    "输出FPS": f"{self.new_fps:.1f}"
                })

        # 释放资源
        self.cap.release()
        self.writer.release()
        if self.show_progress:
            pbar.close()


# 使用示例
if __name__ == "__main__":
    print(cv2.__version__)  # 应输出4.5.5

    interpolator = VideoInterpolator(
        input_path='input.mp4',
        output_path='output.mp4',
        interp_factor=2,  # 每两帧之间插入2帧
        method='optical_flow',
        use_gpu=True
    )
    interpolator.process()
    print("视频插帧处理完成")

