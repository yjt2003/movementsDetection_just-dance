import numpy as np
from typing import List, Optional


class Smoother:
    """
    平滑滤波基类
    """

    def reset(self):
        pass

    def smooth(self, keypoints: List[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError


class EMASmoother(Smoother):
    """
    指数加权平均滤波器
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.last = None

    def reset(self):
        self.last = None

    def smooth(self, keypoints: List[np.ndarray]) -> List[np.ndarray]:
        smoothed = []
        for kp in keypoints:
            kp = np.array(kp, dtype=float)  # 确保是float类型的numpy数组
            if self.last is None:
                self.last = kp.copy()
                smoothed.append(self.last.copy())
            else:
                # 确保形状匹配
                if self.last.shape != kp.shape:
                    self.last = kp.copy()
                self.last = self.alpha * kp + (1 - self.alpha) * self.last
                smoothed.append(self.last.copy())
        return smoothed


class KalmanSmoother(Smoother):
    """
    简单卡尔曼滤波器（对每个关键点独立处理）- 修复numpy错误
    """

    def __init__(self, process_noise: float = 1e-2, measurement_noise: float = 1e-1):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.x = None  # 状态
        self.P = None  # 协方差

    def reset(self):
        self.x = None
        self.P = None

    def smooth(self, keypoints: List[np.ndarray]) -> List[np.ndarray]:
        smoothed = []
        for z in keypoints:
            z = np.array(z, dtype=float).flatten()  # 确保是1D float数组

            if self.x is None:
                self.x = z.copy()
                self.P = np.eye(len(z)) * 0.1  # 初始协方差
                smoothed.append(self.x.copy())
            else:
                # 确保状态向量维度匹配
                if len(self.x) != len(z):
                    self.x = z.copy()
                    self.P = np.eye(len(z)) * 0.1
                    smoothed.append(self.x.copy())
                    continue

                try:
                    # 预测步骤
                    # x_pred = x_prev (简单的匀速模型)
                    P_pred = self.P + self.process_noise * np.eye(len(z))

                    # 更新步骤
                    S = P_pred + self.measurement_noise * np.eye(len(z))  # 创新协方差
                    K = P_pred @ np.linalg.inv(S)  # 卡尔曼增益

                    # 状态更新
                    innovation = z - self.x
                    self.x = self.x + K @ innovation
                    self.P = (np.eye(len(z)) - K) @ P_pred

                    smoothed.append(self.x.copy())

                except np.linalg.LinAlgError:
                    # 如果矩阵求逆失败，直接使用测量值
                    self.x = z.copy()
                    self.P = np.eye(len(z)) * 0.1
                    smoothed.append(self.x.copy())
                except Exception as e:
                    # 其他错误，回退到简单平均
                    self.x = 0.7 * self.x + 0.3 * z
                    smoothed.append(self.x.copy())

        return smoothed


def smooth_keypoints(
        keypoints_seq: List[List[np.ndarray]],
        method: str = 'ema',
        **kwargs
) -> List[List[np.ndarray]]:
    """
    对多帧多人的关键点序列进行平滑处理 - 修复版本
    :param keypoints_seq: 形如[帧][人][关键点坐标]的序列
    :param method: 'ema' 或 'kalman'
    :param kwargs: 传递给滤波器的参数
    :return: 平滑后的关键点序列
    """
    if method == 'ema':
        smoother_cls = EMASmoother
    elif method == 'kalman':
        smoother_cls = KalmanSmoother
    else:
        raise ValueError(f'未知平滑方法: {method}')

    # 安全检查
    if not keypoints_seq or not keypoints_seq[0]:
        return keypoints_seq

    try:
        num_persons = len(keypoints_seq[0])
        num_frames = len(keypoints_seq)

        # 为每个人创建独立的平滑器
        results = [[] for _ in range(num_persons)]

        for person_idx in range(num_persons):
            # 收集该人的所有帧的关键点
            person_kps = []
            for frame_idx in range(num_frames):
                if person_idx < len(keypoints_seq[frame_idx]):
                    kp = keypoints_seq[frame_idx][person_idx]
                    # 确保是正确格式的numpy数组
                    if isinstance(kp, (list, tuple)):
                        kp = np.array(kp, dtype=float)
                    elif isinstance(kp, np.ndarray):
                        kp = kp.astype(float)
                    person_kps.append(kp)
                else:
                    # 如果该帧没有这个人，用零填充
                    if person_kps:
                        person_kps.append(person_kps[-1].copy())
                    else:
                        person_kps.append(np.array([0.0, 0.0]))

            # 创建平滑器并处理
            smoother = smoother_cls(**kwargs)
            try:
                smoothed = smoother.smooth(person_kps)
                results[person_idx] = smoothed
            except Exception as e:
                print(f"Smoother error for person {person_idx}: {e}")
                # 回退：使用原始数据
                results[person_idx] = person_kps

        # 转置回[帧][人][关键点]格式
        results_by_frame = []
        for frame_idx in range(num_frames):
            frame_result = []
            for person_idx in range(num_persons):
                if frame_idx < len(results[person_idx]):
                    frame_result.append(results[person_idx][frame_idx])
                else:
                    # 安全回退
                    if person_idx < len(keypoints_seq[frame_idx]):
                        frame_result.append(keypoints_seq[frame_idx][person_idx])
                    else:
                        frame_result.append(np.array([0.0, 0.0]))
            results_by_frame.append(frame_result)

        return results_by_frame

    except Exception as e:
        print(f"Smoothing failed: {e}, returning original data")
        return keypoints_seq