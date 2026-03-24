import warnings
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


class VideoInterpolator:
    """
    视频插帧类

    参数:
    :param input_path: 输入视频路径
    :param output_path: 输出视频路径
    :param interp_factor: 插帧倍数（每两个原始帧之间插入的帧数，必须 >= 1）
    :param method: 插值方法，支持 'linear' 或 'optical_flow'
    :param flow_params: 光流算法参数字典
    :param show_progress: 是否显示进度条
    :param use_gpu: 是否尝试使用GPU加速（支持NVIDIA CUDA和Intel OpenCL）
    """

    # 支持的插值方法
    SUPPORTED_METHODS = ["linear", "optical_flow"]

    # 默认光流参数
    DEFAULT_FLOW_PARAMS = {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0,
    }

    def __init__(
        self,
        input_path: str,
        output_path: str,
        interp_factor: int = 1,
        method: str = "linear",
        flow_params: Optional[Dict] = None,
        show_progress: bool = True,
        use_gpu: bool = False,
    ):
        """
        初始化视频插帧器
        """
        # 参数验证
        self._validate_params(interp_factor, method)

        # 基础参数
        self.input_path = input_path
        self.output_path = output_path
        self.interp_factor = interp_factor
        self.method = method.lower()
        self.use_gpu = use_gpu
        self.show_progress = show_progress

        # 光流参数（在硬件初始化之前设置，修复BUG）
        self.flow_params = flow_params or self.DEFAULT_FLOW_PARAMS.copy()

        # 视频参数（将在_open_video中初始化）
        self.cap = None
        self.writer = None
        self.width = 0
        self.height = 0
        self.original_fps = 0
        self.new_fps = 0
        self.total_frames = 0

        # 硬件加速相关
        self.gpu_flow = None
        self.gpu_available = False
        self.gpu_backend = None  # 'cuda', 'opencl', 或 None

        # 内部状态
        self._is_processing = False
        self._resources_to_cleanup = []

        # 初始化视频和硬件
        try:
            self._open_video()
            self._init_hardware_acceleration()
        except Exception as e:
            self._cleanup_resources()
            raise

    def _validate_params(self, interp_factor: int, method: str) -> None:
        """
        验证输入参数
        """
        if interp_factor < 1:
            raise ValueError(f"interp_factor 必须 >= 1，当前值: {interp_factor}")

        if method.lower() not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"不支持的方法: {method}，支持的方法: {self.SUPPORTED_METHODS}"
            )

    def _open_video(self) -> None:
        """
        打开输入视频并初始化参数
        """
        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.input_path}")

        # 获取视频属性
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)

        if self.original_fps <= 0:
            self.original_fps = 30.0  # 默认值
            warnings.warn(f"无法获取视频FPS，使用默认值: {self.original_fps}")

        self.new_fps = self.original_fps * (self.interp_factor + 1)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 初始化视频写入器
        # 智能选择编解码器
        fourcc = self._get_fourcc_for_path(self.output_path)
        self.writer = cv2.VideoWriter(
            self.output_path, fourcc, self.new_fps, (self.width, self.height)
        )

        if not self.writer.isOpened():
            # 尝试备用编解码器
            backup_fourccs = self._get_backup_fourccs(self.output_path)
            for backup_fourcc in backup_fourccs:
                if backup_fourcc == fourcc:
                    continue  # 跳过已经尝试过的
                self.writer.release()
                self.writer = cv2.VideoWriter(
                    self.output_path,
                    backup_fourcc,
                    self.new_fps,
                    (self.width, self.height),
                )
                if self.writer.isOpened():
                    print(f"警告: 使用备用编解码器: {backup_fourcc}")
                    break

            if not self.writer.isOpened():
                raise ValueError(
                    f"无法创建输出视频文件: {self.output_path}，尝试了多种编解码器"
                )

    def _init_hardware_acceleration(self) -> None:
        """
        初始化硬件加速环境，支持多种GPU后端
        """
        if not self.use_gpu:
            self.gpu_available = False
            self.gpu_backend = None
            return

        # 尝试不同GPU后端，按优先级排序
        backends = [
            self._try_cuda_backend,
            self._try_opencl_backend,
            self._try_intel_backend,
        ]

        for backend_func in backends:
            try:
                backend_func()
                if self.gpu_available:
                    return
            except Exception as e:
                continue

        # 所有GPU后端都失败，回退到CPU
        self.gpu_available = False
        self.gpu_backend = None
        warnings.warn("所有GPU加速尝试失败，回退到CPU模式")

    def _try_cuda_backend(self) -> None:
        """
        尝试使用NVIDIA CUDA后端
        """
        if not hasattr(cv2, "cuda") or not hasattr(
            cv2.cuda, "FarnebackOpticalFlow_create"
        ):
            raise RuntimeError("OpenCV未编译CUDA支持")

        # 检查是否有可用的CUDA设备
        device_count = cv2.cuda.getCudaEnabledDeviceCount()
        if device_count <= 0:
            raise RuntimeError("未检测到CUDA设备")

        # 创建CUDA光流计算器
        self.gpu_flow = cv2.cuda.FarnebackOpticalFlow_create(
            pyrScale=self.flow_params["pyr_scale"],
            numLevels=self.flow_params["levels"],
            winSize=self.flow_params["winsize"],
            numIters=self.flow_params["iterations"],
            polyN=self.flow_params["poly_n"],
            polySigma=self.flow_params["poly_sigma"],
            flags=self.flow_params["flags"],
        )

        self.gpu_available = True
        self.gpu_backend = "cuda"
        print(f"已启用CUDA GPU加速（设备: {cv2.cuda.getDevice()})")

    def _try_opencl_backend(self) -> None:
        """
        尝试使用OpenCL后端
        """
        cv2.ocl.setUseOpenCL(True)
        if not cv2.ocl.haveOpenCL():
            raise RuntimeError("OpenCL不可用")

        # 尝试创建OpenCL光流计算器
        try:
            # 检查是否有GPU加速的Farneback光流
            if hasattr(cv2, "optflow") and hasattr(
                cv2.optflow, "createOptFlow_Farneback_GPU"
            ):
                self.gpu_flow = cv2.optflow.createOptFlow_Farneback_GPU(
                    pyrScale=self.flow_params["pyr_scale"],
                    numLevels=self.flow_params["levels"],
                    winSize=self.flow_params["winsize"],
                    numIters=self.flow_params["iterations"],
                    polyN=self.flow_params["poly_n"],
                    polySigma=self.flow_params["poly_sigma"],
                    flags=self.flow_params["flags"],
                )
            else:
                # 尝试DIS光流作为备选
                self.gpu_flow = cv2.DISOpticalFlow_create(
                    cv2.DISOPTICAL_FLOW_PRESET_FAST
                )
                self.gpu_flow.setUseSpatialPropagation(True)

            self.gpu_available = True
            self.gpu_backend = "opencl"
            print("已启用OpenCL GPU加速")
        except Exception as e:
            raise RuntimeError(f"OpenCL初始化失败: {str(e)}")

    def _try_intel_backend(self) -> None:
        """
        尝试使用Intel专用后端（兼容原版）
        """
        try:
            # 检查OpenVINO环境可用性
            from openvino.runtime import Core

            ie = Core()
            gpu_devices = ie.available_devices

            if "GPU" not in gpu_devices:
                raise RuntimeError("未检测到兼容的Intel GPU设备")

            print(f"检测到Intel GPU设备: {ie.get_property('GPU', 'FULL_DEVICE_NAME')}")

            # 使用OpenCL后端处理Intel GPU
            self._try_opencl_backend()
        except ImportError:
            # OpenVINO未安装，跳过
            raise RuntimeError("OpenVINO未安装")
        except Exception as e:
            raise RuntimeError(f"Intel后端初始化失败: {str(e)}")

    def _calculate_optical_flow(
        self, frame1: np.ndarray, frame2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算双向光流，支持多种硬件后端
        """
        if not self.gpu_available:
            # CPU计算
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            forward_flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, **self.flow_params
            )
            backward_flow = cv2.calcOpticalFlowFarneback(
                gray2, gray1, None, **self.flow_params
            )

            return forward_flow, backward_flow

        # GPU计算 - 根据后端类型处理
        if self.gpu_backend == "cuda":
            # CUDA后端
            gpu_frame1 = cv2.cuda_GpuMat(frame1)
            gpu_frame2 = cv2.cuda_GpuMat(frame2)
            gpu_gray1 = cv2.cuda.cvtColor(gpu_frame1, cv2.COLOR_BGR2GRAY)
            gpu_gray2 = cv2.cuda.cvtColor(gpu_frame2, cv2.COLOR_BGR2GRAY)

            forward_flow_gpu = self.gpu_flow.calc(gpu_gray1, gpu_gray2, None)
            backward_flow_gpu = self.gpu_flow.calc(gpu_gray2, gpu_gray1, None)

            forward_flow = forward_flow_gpu.download()
            backward_flow = backward_flow_gpu.download()

        else:
            # OpenCL后端
            gpu_frame1 = cv2.UMat(frame1)
            gpu_frame2 = cv2.UMat(frame2)
            gpu_gray1 = cv2.cvtColor(gpu_frame1, cv2.COLOR_BGR2GRAY)
            gpu_gray2 = cv2.cvtColor(gpu_frame2, cv2.COLOR_BGR2GRAY)

            forward_flow = self.gpu_flow.calc(gpu_gray1, gpu_gray2, None)
            backward_flow = self.gpu_flow.calc(gpu_gray2, gpu_gray1, None)

            # 如果是UMat，转换为numpy数组
            if isinstance(forward_flow, cv2.UMat):
                forward_flow = forward_flow.get()
            if isinstance(backward_flow, cv2.UMat):
                backward_flow = backward_flow.get()

        return forward_flow, backward_flow

    def _warp_frame(self, frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """
        根据光流场变形图像，优化性能版本
        """
        h, w = flow.shape[:2]

        # 预计算映射网格（优化性能）
        if not hasattr(self, "_flow_map_base"):
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            self._flow_map_base = np.stack([x, y], axis=-1).astype(np.float32)

        flow_map = self._flow_map_base + flow

        return cv2.remap(
            frame,
            flow_map[..., 0],
            flow_map[..., 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    def _optical_flow_interpolation(
        self, prev_frame: np.ndarray, next_frame: np.ndarray, alpha: float
    ) -> np.ndarray:
        """
        基于光流的插值方法，改进版
        """
        # 计算双向光流
        forward_flow, backward_flow = self._calculate_optical_flow(
            prev_frame, next_frame
        )

        # 改进的时间插值公式
        # 使用加权平均而不是简单缩放，减少伪影
        forward_flow_t = forward_flow * alpha
        backward_flow_t = backward_flow * (1.0 - alpha)

        # 变形前后帧
        warped_forward = self._warp_frame(prev_frame, forward_flow_t)
        warped_backward = self._warp_frame(next_frame, backward_flow_t)

        # 自适应权重融合
        # 根据alpha值调整前后帧权重，使结果更平滑
        forward_weight = 1.0 - alpha
        backward_weight = alpha

        # 归一化权重
        total_weight = forward_weight + backward_weight
        forward_weight /= total_weight
        backward_weight /= total_weight

        return cv2.addWeighted(
            warped_forward, forward_weight, warped_backward, backward_weight, 0
        )

    def _linear_interpolation(
        self, frame1: np.ndarray, frame2: np.ndarray, alpha: float
    ) -> np.ndarray:
        """
        线性插值生成中间帧
        """
        return cv2.addWeighted(frame1, 1.0 - alpha, frame2, alpha, 0)

    def _generate_inter_frames(
        self, prev_frame: np.ndarray, next_frame: np.ndarray
    ) -> list:
        """
        生成中间帧序列
        """
        inter_frames = []

        for i in range(1, self.interp_factor + 1):
            alpha = i / (self.interp_factor + 1)

            if self.method == "linear":
                frame = self._linear_interpolation(prev_frame, next_frame, alpha)
            elif self.method == "optical_flow":
                frame = self._optical_flow_interpolation(prev_frame, next_frame, alpha)
            else:
                # 不应该发生，已在初始化时验证
                raise ValueError(f"未知的方法: {self.method}")

            inter_frames.append(frame)

        return inter_frames

    def process(self) -> None:
        """
        处理视频并生成插帧结果，异常安全版本
        """
        if self._is_processing:
            raise RuntimeError("已经在处理视频中，不能重复调用process()")

        self._is_processing = True
        pbar = None

        try:
            # 读取第一帧
            ret, prev_frame = self.cap.read()
            if not ret:
                raise ValueError("无法读取视频帧，视频文件可能为空或损坏")

            # 初始化进度条
            if self.show_progress:
                pbar = tqdm(
                    total=self.total_frames,
                    desc="处理进度",
                    unit="frame",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                )

            # 写入第一帧
            self.writer.write(prev_frame)
            processed_frames = 1

            # 主处理循环
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
                    pbar.set_postfix(
                        {
                            "方法": self.method,
                            "插帧倍数": self.interp_factor,
                            "输出FPS": f"{self.new_fps:.1f}",
                            "GPU加速": f"{self.gpu_backend or 'CPU'}",
                        }
                    )

            if self.show_progress:
                pbar.close()
                print(
                    f"处理完成！总共处理了 {processed_frames} 帧，"
                    f"输出帧率: {self.new_fps:.1f} FPS"
                )

        except KeyboardInterrupt:
            print("\n处理被用户中断")
            if self.show_progress and pbar:
                pbar.close()
            raise

        except Exception as e:
            if self.show_progress and pbar:
                pbar.close()
            print(f"处理过程中发生错误: {str(e)}")
            raise

        finally:
            self._cleanup_resources()
            self._is_processing = False

    def _cleanup_resources(self) -> None:
        """
        清理所有资源，确保异常安全
        """
        if self.cap is not None:
            self.cap.release()

        if self.writer is not None:
            self.writer.release()

        # 清理GPU资源
        if self.gpu_backend == "cuda" and hasattr(cv2.cuda, "resetDevice"):
            cv2.cuda.resetDevice()

        # 清理OpenCL上下文
        cv2.ocl.setUseOpenCL(False)

    def _get_fourcc_for_path(self, path: str) -> int:
        """
        根据文件路径智能选择编解码器

        参数:
            path: 输出文件路径

        返回:
            FourCC编解码器代码
        """
        path_lower = path.lower()

        if path_lower.endswith(".avi"):
            # AVI格式优先使用XVID，备选MJPG, DIVX
            codec_candidates = ["XVID", "MJPG", "DIVX", "FMP4"]
        elif path_lower.endswith(".mp4") or path_lower.endswith(".m4v"):
            # MP4格式优先使用mp4v，备选avc1, X264
            codec_candidates = ["mp4v", "avc1", "X264", "H264"]
        elif path_lower.endswith(".mov"):
            # MOV格式使用MP4V或MJPG
            codec_candidates = ["mp4v", "MJPG", "avc1"]
        elif path_lower.endswith(".mkv"):
            # MKV格式使用X264或MP4V
            codec_candidates = ["X264", "mp4v", "avc1"]
        else:
            # 其他格式使用通用编解码器
            codec_candidates = ["mp4v", "XVID", "MJPG", "avc1"]

        # 尝试候选编解码器
        for codec in codec_candidates:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                # 测试编解码器是否可用（通过创建临时VideoWriter）
                test_writer = cv2.VideoWriter(
                    path,
                    fourcc,
                    25.0,
                    (640, 480),  # 使用测试参数
                )
                if test_writer.isOpened():
                    test_writer.release()
                    return fourcc
                test_writer.release()
            except Exception:
                continue

        # 如果没有可用的编解码器，返回默认的mp4v
        return cv2.VideoWriter_fourcc(*"mp4v")

    def _get_backup_fourccs(self, path: str) -> List[int]:
        """
        获取备用编解码器列表

        参数:
            path: 输出文件路径

        返回:
            备用FourCC编解码器代码列表
        """
        path_lower = path.lower()
        backup_candidates = []

        # 基础备用编解码器
        base_codecs = ["XVID", "MJPG", "mp4v", "avc1", "FMP4", "DIVX"]

        # 根据文件类型调整优先级
        if path_lower.endswith(".avi"):
            backup_candidates = ["XVID", "MJPG", "DIVX", "FMP4", "mp4v", "avc1"]
        elif path_lower.endswith(".mp4") or path_lower.endswith(".m4v"):
            backup_candidates = ["mp4v", "avc1", "X264", "H264", "XVID", "MJPG"]
        else:
            backup_candidates = base_codecs

        # 转换为FourCC代码
        fourcc_list = []
        for codec in backup_candidates:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                fourcc_list.append(fourcc)
            except Exception:
                continue

        return fourcc_list

    def get_video_info(self) -> Dict[str, Any]:
        """
        获取视频信息
        """
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "resolution": (self.width, self.height),
            "original_fps": self.original_fps,
            "new_fps": self.new_fps,
            "total_frames": self.total_frames,
            "interp_factor": self.interp_factor,
            "method": self.method,
            "gpu_acceleration": self.gpu_backend or "CPU",
        }


# 使用示例
if __name__ == "__main__":
    # 打印OpenCV版本和GPU支持信息
    print(f"OpenCV版本: {cv2.__version__}")

    # 检查CUDA支持
    if hasattr(cv2, "cuda"):
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"CUDA设备数量: {cuda_devices}")

    # 检查OpenCL支持
    cv2.ocl.setUseOpenCL(True)
    print(f"OpenCL可用: {cv2.ocl.haveOpenCL()}")
    cv2.ocl.setUseOpenCL(False)

    # 示例使用
    print("\n=== VideoInterpolator 示例 ===")

    try:
        interpolator = VideoInterpolator(
            input_path="input.mp4",
            output_path="output.mp4",
            interp_factor=2,  # 每两帧之间插入2帧
            method="optical_flow",
            use_gpu=True,
            show_progress=True,
        )

        # 打印视频信息
        info = interpolator.get_video_info()
        print(f"视频信息: {info}")

        # 开始处理
        interpolator.process()
        print("视频插帧处理完成！")

    except FileNotFoundError as e:
        print(f"文件错误: {str(e)}")
        print("请确保 input.mp4 文件存在")
    except Exception as e:
        print(f"处理失败: {str(e)}")
