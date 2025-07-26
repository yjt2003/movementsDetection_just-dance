import numpy as np
from typing import List, Optional


def fix_keypoints(
        keypoints_seq: List[List[np.ndarray]],
        conf_seq: List[List[np.ndarray]],
        method: str = 'linear',
        conf_threshold: float = 0.3
) -> List[List[np.ndarray]]:
    """
    对多帧多人的关键点序列进行遮挡点补全 - 修复版本
    :param keypoints_seq: [帧][人][关键点坐标]
    :param conf_seq: [帧][人][关键点置信度]
    :param method: 'linear' 或 'symmetric'
    :param conf_threshold: 置信度阈值
    :return: 补全后的关键点序列
    """
    try:
        if method == 'linear':
            return linear_interpolate_missing(keypoints_seq, conf_seq, conf_threshold)
        elif method == 'symmetric':
            return symmetric_fix_missing(keypoints_seq, conf_seq, conf_threshold)
        else:
            print(f'未知补全方法: {method}, 返回原始数据')
            return keypoints_seq
    except Exception as e:
        print(f"Fix keypoints error: {e}, returning original data")
        return keypoints_seq


def linear_interpolate_missing(
        keypoints_seq: List[List[np.ndarray]],
        conf_seq: List[List[np.ndarray]],
        conf_threshold: float
) -> List[List[np.ndarray]]:
    """
    对每个关键点序列做线性插值补全 - 修复版本
    """
    try:
        if not keypoints_seq or not keypoints_seq[0]:
            return keypoints_seq

        num_frames = len(keypoints_seq)
        num_persons = len(keypoints_seq[0])

        # 深度复制以避免修改原始数据
        fixed_seq = []
        for frame in keypoints_seq:
            frame_copy = []
            for person_kps in frame:
                if isinstance(person_kps, (list, tuple)):
                    person_copy = [np.array(kp, dtype=float) for kp in person_kps]
                else:
                    person_copy = np.array(person_kps, dtype=float).copy()
                frame_copy.append(person_copy)
            fixed_seq.append(frame_copy)

        for person_idx in range(num_persons):
            # 获取该人的关键点数量
            num_keypoints = len(fixed_seq[0][person_idx]) if fixed_seq[0][person_idx] is not None else 0

            for kp_idx in range(num_keypoints):
                try:
                    # 收集该人该关键点的所有帧的置信度和坐标
                    confs = []
                    coords = []

                    for frame_idx in range(num_frames):
                        try:
                            if (person_idx < len(conf_seq[frame_idx]) and
                                    kp_idx < len(conf_seq[frame_idx][person_idx])):
                                conf_val = float(conf_seq[frame_idx][person_idx][kp_idx])
                                confs.append(conf_val)
                            else:
                                confs.append(0.0)

                            if (person_idx < len(keypoints_seq[frame_idx]) and
                                    kp_idx < len(keypoints_seq[frame_idx][person_idx])):
                                coord_val = np.array(keypoints_seq[frame_idx][person_idx][kp_idx], dtype=float)
                                coords.append(coord_val)
                            else:
                                coords.append(np.array([0.0, 0.0]))
                        except (IndexError, ValueError) as e:
                            confs.append(0.0)
                            coords.append(np.array([0.0, 0.0]))

                    # 找到有效帧
                    valid = [i for i, c in enumerate(confs) if c >= conf_threshold]

                    # 对无效帧进行插值
                    for i in range(num_frames):
                        if confs[i] < conf_threshold and valid:
                            # 前后找最近的有效帧做插值
                            prev = None
                            next_ = None

                            for j in range(i - 1, -1, -1):
                                if j in valid:
                                    prev = j
                                    break

                            for j in range(i + 1, num_frames):
                                if j in valid:
                                    next_ = j
                                    break

                            try:
                                if prev is not None and next_ is not None:
                                    # 双边插值
                                    ratio = (i - prev) / (next_ - prev)
                                    interpolated = coords[prev] * (1 - ratio) + coords[next_] * ratio
                                    fixed_seq[i][person_idx][kp_idx] = interpolated
                                elif prev is not None:
                                    # 前向填充
                                    fixed_seq[i][person_idx][kp_idx] = coords[prev].copy()
                                elif next_ is not None:
                                    # 后向填充
                                    fixed_seq[i][person_idx][kp_idx] = coords[next_].copy()
                                # 否则保持原值
                            except Exception as inner_e:
                                # 插值失败，保持原值
                                continue

                except Exception as kp_e:
                    # 单个关键点处理失败，跳过
                    continue

        return fixed_seq

    except Exception as e:
        print(f"Linear interpolation error: {e}")
        return keypoints_seq


# MediaPipe 33点对称关系（左:右）
SYMMETRIC_PAIRS = [
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)
]


def symmetric_fix_missing(
        keypoints_seq: List[List[np.ndarray]],
        conf_seq: List[List[np.ndarray]],
        conf_threshold: float
) -> List[List[np.ndarray]]:
    """
    用对称点补全丢失关键点 - 修复版本
    """
    try:
        if not keypoints_seq or not keypoints_seq[0]:
            return keypoints_seq

        num_frames = len(keypoints_seq)
        num_persons = len(keypoints_seq[0])

        # 深度复制
        fixed_seq = []
        for frame in keypoints_seq:
            frame_copy = []
            for person_kps in frame:
                if isinstance(person_kps, (list, tuple)):
                    person_copy = [np.array(kp, dtype=float) for kp in person_kps]
                else:
                    person_copy = np.array(person_kps, dtype=float).copy()
                frame_copy.append(person_copy)
            fixed_seq.append(frame_copy)

        for frame_idx in range(num_frames):
            for person_idx in range(num_persons):
                try:
                    for left, right in SYMMETRIC_PAIRS:
                        # 安全检查索引
                        if (left >= len(fixed_seq[frame_idx][person_idx]) or
                                right >= len(fixed_seq[frame_idx][person_idx])):
                            continue

                        # 获取置信度
                        try:
                            left_conf = float(conf_seq[frame_idx][person_idx][left]) if (
                                    frame_idx < len(conf_seq) and
                                    person_idx < len(conf_seq[frame_idx]) and
                                    left < len(conf_seq[frame_idx][person_idx])
                            ) else 0.0

                            right_conf = float(conf_seq[frame_idx][person_idx][right]) if (
                                    frame_idx < len(conf_seq) and
                                    person_idx < len(conf_seq[frame_idx]) and
                                    right < len(conf_seq[frame_idx][person_idx])
                            ) else 0.0
                        except (IndexError, ValueError):
                            continue

                        # 对称补全
                        try:
                            if left_conf < conf_threshold and right_conf >= conf_threshold:
                                # 用右点对称补左点
                                right_point = np.array(fixed_seq[frame_idx][person_idx][right], dtype=float)
                                # 简单的x轴对称（可能需要根据实际坐标系调整）
                                left_point = right_point.copy()
                                if len(left_point) >= 2:
                                    # 假设图像宽度为640，可以根据实际情况调整
                                    left_point[0] = 640 - left_point[0]
                                fixed_seq[frame_idx][person_idx][left] = left_point

                            elif right_conf < conf_threshold and left_conf >= conf_threshold:
                                # 用左点对称补右点
                                left_point = np.array(fixed_seq[frame_idx][person_idx][left], dtype=float)
                                right_point = left_point.copy()
                                if len(right_point) >= 2:
                                    right_point[0] = 640 - right_point[0]
                                fixed_seq[frame_idx][person_idx][right] = right_point
                        except Exception as sym_e:
                            # 对称操作失败，跳过
                            continue

                except Exception as person_e:
                    # 单个人处理失败，跳过
                    continue

        return fixed_seq

    except Exception as e:
        print(f"Symmetric fixing error: {e}")
        return keypoints_seq
