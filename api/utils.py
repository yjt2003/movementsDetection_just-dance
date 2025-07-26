import base64
import cv2
import numpy as np
import io
from typing import List, Optional, Tuple, Union
from PIL import Image
import logging

from .model import Keypoint, Landmarks

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def decode_base64_image(image_data: str) -> Optional[np.ndarray]:
    """
    解码base64图片数据

    Args:
        image_data: base64编码的图片数据

    Returns:
        解码后的图片数组，失败返回None
    """
    try:
        # 处理data URL格式
        if ',' in image_data:
            header, data = image_data.split(',', 1)
            img_bytes = base64.b64decode(data)
        else:
            img_bytes = base64.b64decode(image_data)

        # 使用PIL进行解码，更稳定
        try:
            pil_image = Image.open(io.BytesIO(img_bytes))
            # 转换为RGB格式
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # 转换为numpy数组
            frame = np.array(pil_image)
            # 转换为BGR格式（OpenCV格式）
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        except Exception as pil_error:
            logger.warning(f"PIL解码失败，尝试OpenCV解码: {pil_error}")
            # fallback到OpenCV解码
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return frame

    except Exception as e:
        logger.error(f"图片解码失败: {e}")
        return None



def encode_image_to_base64(image: np.ndarray, format: str = 'jpeg', quality: int = 80) -> str:
    """
    将图片编码为base64格式

    Args:
        image: 图片数组
        format: 图片格式 ('jpeg', 'png')
        quality: 图片质量 (1-100)

    Returns:
        base64编码的图片字符串
    """
    try:
        if format.lower() == 'jpeg':
            _, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            mime_type = 'image/jpeg'
        elif format.lower() == 'png':
            _, encoded = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            mime_type = 'image/png'
        else:
            raise ValueError(f"不支持的图片格式: {format}")

        base64_str = base64.b64encode(encoded.tobytes()).decode('utf-8')
        return f'data:{mime_type};base64,{base64_str}'

    except Exception as e:
        logger.error(f"图片编码失败: {e}")
        return ''


def pad_landmarks(landmarks: List[Keypoint], target_num: int = 33) -> List[Keypoint]:
    """
    补齐关键点到指定数量

    Args:
        landmarks: 关键点列表
        target_num: 目标关键点数量

    Returns:
        补齐后的关键点列表
    """
    try:
        # 如果输入为空，创建默认关键点
        if not landmarks:
            landmarks = []

        # 转换为Keypoint对象
        result = []
        for lm in landmarks:
            if isinstance(lm, Keypoint):
                result.append(lm)
            else:
                # 处理其他格式的关键点
                result.append(Keypoint(
                    x=getattr(lm, 'x', 0.0),
                    y=getattr(lm, 'y', 0.0),
                    z=getattr(lm, 'z', 0.0),
                    confidence=getattr(lm, 'confidence', 0.0),
                    visible=getattr(lm, 'visible', False)
                ))

        # 补齐到目标数量
        while len(result) < target_num:
            result.append(Keypoint(
                x=0.0, y=0.0, z=0.0,
                confidence=0.0, visible=False
            ))

        # 截断到目标数量
        return result[:target_num]

    except Exception as e:
        logger.error(f"关键点补齐失败: {e}")
        # 返回默认关键点
        return [Keypoint(x=0.0, y=0.0, z=0.0, confidence=0.0, visible=False) for _ in range(target_num)]


def draw_landmarks(image: np.ndarray, landmarks: List[Keypoint],
                   color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 2, radius: int = 4) -> np.ndarray:
    """
    在图片上绘制关键点

    Args:
        image: 输入图片
        landmarks: 关键点列表
        color: 绘制颜色 (B, G, R)
        thickness: 线条粗细
        radius: 关键点半径

    Returns:
        绘制后的图片
    """
    try:
        if image is None or len(landmarks) == 0:
            return image

        # 复制图片避免修改原图
        result = image.copy()
        h, w = result.shape[:2]

        # MediaPipe 33点连接关系
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (27, 29),
            (29, 31), (26, 28), (28, 30), (30, 32)
        ]

        # 绘制骨架连接
        for pt1_idx, pt2_idx in connections:
            if pt1_idx < len(landmarks) and pt2_idx < len(landmarks):
                pt1 = landmarks[pt1_idx]
                pt2 = landmarks[pt2_idx]

                if (pt1.visible and pt2.visible and
                        pt1.confidence > 0.5 and pt2.confidence > 0.5):
                    # 确保坐标在图片范围内
                    x1 = int(max(0, min(w - 1, pt1.x)))
                    y1 = int(max(0, min(h - 1, pt1.y)))
                    x2 = int(max(0, min(w - 1, pt2.x)))
                    y2 = int(max(0, min(h - 1, pt2.y)))

                    cv2.line(result, (x1, y1), (x2, y2), color, thickness)

        # 绘制关键点
        for kp in landmarks:
            if kp.visible and kp.confidence > 0.5:
                x = int(max(0, min(w - 1, kp.x)))
                y = int(max(0, min(h - 1, kp.y)))
                cv2.circle(result, (x, y), radius, color, -1)

        return result

    except Exception as e:
        logger.error(f"关键点绘制失败: {e}")
        return image


def resize_image(image: np.ndarray, target_size: Tuple[int, int],
                 keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    调整图片大小

    Args:
        image: 输入图片
        target_size: 目标尺寸 (width, height)
        keep_aspect_ratio: 是否保持宽高比

    Returns:
        调整后的图片
    """
    try:
        if image is None:
            return None

        h, w = image.shape[:2]
        target_w, target_h = target_size

        if keep_aspect_ratio:
            # 计算缩放比例
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            # 调整大小
            resized = cv2.resize(image, (new_w, new_h))

            # 创建目标大小的画布
            result = np.zeros((target_h, target_w, 3), dtype=np.uint8)

            # 计算居中位置
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2

            # 将调整后的图片放置在画布中央
            result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

            return result
        else:
            return cv2.resize(image, target_size)

    except Exception as e:
        logger.error(f"图片调整失败: {e}")
        return image


def normalize_landmarks(landmarks: List[Keypoint],
                        image_size: Tuple[int, int]) -> List[Keypoint]:
    """
    归一化关键点坐标到[0,1]范围

    Args:
        landmarks: 关键点列表
        image_size: 图片尺寸 (width, height)

    Returns:
        归一化后的关键点列表
    """
    try:
        if not landmarks:
            return landmarks

        w, h = image_size
        normalized = []

        for kp in landmarks:
            normalized.append(Keypoint(
                x=kp.x / w if w > 0 else 0.0,
                y=kp.y / h if h > 0 else 0.0,
                z=kp.z,
                confidence=kp.confidence,
                visible=kp.visible
            ))

        return normalized

    except Exception as e:
        logger.error(f"关键点归一化失败: {e}")
        return landmarks


def denormalize_landmarks(landmarks: List[Keypoint],
                          image_size: Tuple[int, int]) -> List[Keypoint]:
    """
    反归一化关键点坐标

    Args:
        landmarks: 归一化的关键点列表
        image_size: 图片尺寸 (width, height)

    Returns:
        反归一化后的关键点列表
    """
    try:
        if not landmarks:
            return landmarks

        w, h = image_size
        denormalized = []

        for kp in landmarks:
            denormalized.append(Keypoint(
                x=kp.x * w,
                y=kp.y * h,
                z=kp.z,
                confidence=kp.confidence,
                visible=kp.visible
            ))

        return denormalized

    except Exception as e:
        logger.error(f"关键点反归一化失败: {e}")
        return landmarks


def filter_valid_landmarks(landmarks: List[Keypoint],
                           min_confidence: float = 0.5) -> List[Keypoint]:
    """
    过滤有效的关键点

    Args:
        landmarks: 关键点列表
        min_confidence: 最小置信度阈值

    Returns:
        过滤后的关键点列表
    """
    try:
        return [kp for kp in landmarks
                if kp.visible and kp.confidence >= min_confidence]
    except Exception as e:
        logger.error(f"关键点过滤失败: {e}")
        return landmarks


def calculate_bounding_box(landmarks: List[Keypoint]) -> Tuple[int, int, int, int]:
    """
    计算关键点的边界框

    Args:
        landmarks: 关键点列表

    Returns:
        边界框坐标 (x, y, width, height)
    """
    try:
        valid_landmarks = filter_valid_landmarks(landmarks)

        if not valid_landmarks:
            return (0, 0, 0, 0)

        x_coords = [kp.x for kp in valid_landmarks]
        y_coords = [kp.y for kp in valid_landmarks]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        return (int(x_min), int(y_min),
                int(x_max - x_min), int(y_max - y_min))

    except Exception as e:
        logger.error(f"边界框计算失败: {e}")
        return (0, 0, 0, 0)


def smooth_landmarks_temporal(landmarks_sequence: List[List[Keypoint]],
                              window_size: int = 5) -> List[List[Keypoint]]:
    """
    对关键点序列进行时间平滑

    Args:
        landmarks_sequence: 关键点序列
        window_size: 平滑窗口大小

    Returns:
        平滑后的关键点序列
    """
    try:
        if not landmarks_sequence or len(landmarks_sequence) < 2:
            return landmarks_sequence

        smoothed = []
        half_window = window_size // 2

        for i in range(len(landmarks_sequence)):
            # 确定窗口范围
            start_idx = max(0, i - half_window)
            end_idx = min(len(landmarks_sequence), i + half_window + 1)

            # 获取窗口内的关键点
            window_landmarks = landmarks_sequence[start_idx:end_idx]

            # 对每个关键点位置求平均
            smoothed_frame = []
            for kp_idx in range(len(landmarks_sequence[i])):
                x_sum = sum(frame[kp_idx].x for frame in window_landmarks)
                y_sum = sum(frame[kp_idx].y for frame in window_landmarks)
                z_sum = sum(frame[kp_idx].z for frame in window_landmarks)
                conf_sum = sum(frame[kp_idx].confidence for frame in window_landmarks)

                window_len = len(window_landmarks)
                smoothed_frame.append(Keypoint(
                    x=x_sum / window_len,
                    y=y_sum / window_len,
                    z=z_sum / window_len,
                    confidence=conf_sum / window_len,
                    visible=landmarks_sequence[i][kp_idx].visible
                ))

            smoothed.append(smoothed_frame)

        return smoothed

    except Exception as e:
        logger.error(f"时间平滑失败: {e}")
        return landmarks_sequence


def validate_landmarks(landmarks: List[Keypoint],
                       image_size: Optional[Tuple[int, int]] = None) -> bool:
    """
    验证关键点的有效性

    Args:
        landmarks: 关键点列表
        image_size: 图片尺寸 (width, height)，用于坐标范围检查

    Returns:
        是否有效
    """
    try:
        if not landmarks:
            return False

        for kp in landmarks:
            # 检查坐标是否为有效数值
            if not all(isinstance(val, (int, float)) and not np.isnan(val)
                       for val in [kp.x, kp.y, kp.z, kp.confidence]):
                return False

            # 检查置信度范围
            if not (0.0 <= kp.confidence <= 1.0):
                return False

            # 检查坐标范围（如果提供了图片尺寸）
            if image_size:
                w, h = image_size
                if not (0 <= kp.x <= w and 0 <= kp.y <= h):
                    return False

        return True

    except Exception as e:
        logger.error(f"关键点验证失败: {e}")
        return False


def convert_landmarks_format(landmarks: List[Keypoint],
                             target_format: str = 'mediapipe') -> List[Keypoint]:
    """
    转换关键点格式

    Args:
        landmarks: 关键点列表
        target_format: 目标格式 ('mediapipe', 'coco', 'openpose')

    Returns:
        转换后的关键点列表
    """
    try:
        if target_format.lower() == 'mediapipe':
            # 确保33个关键点
            return pad_landmarks(landmarks, 33)
        elif target_format.lower() == 'coco':
            # 转换为COCO 17点格式
            coco_mapping = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
            result = []
            for i in coco_mapping:
                if i < len(landmarks):
                    result.append(landmarks[i])
                else:
                    result.append(Keypoint(x=0.0, y=0.0, z=0.0, confidence=0.0, visible=False))
            return result
        else:
            logger.warning(f"不支持的格式: {target_format}")
            return landmarks

    except Exception as e:
        logger.error(f"关键点格式转换失败: {e}")
        return landmarks

# 在你现有的 api/utils.py 文件末尾添加这个兼容性函数：

def ensure_landmarks_format(landmarks: List[Keypoint]) -> List[Keypoint]:
    """
    确保关键点格式正确，用于同步优化
    
    Args:
        landmarks: 关键点列表
        
    Returns:
        格式化后的关键点列表
    """
    try:
        if not landmarks:
            return []
        
        # 确保每个关键点都有必要的属性
        formatted = []
        for lm in landmarks:
            if hasattr(lm, 'x') and hasattr(lm, 'y'):
                formatted.append(Keypoint(
                    x=float(lm.x),
                    y=float(lm.y),
                    z=float(getattr(lm, 'z', 0.0)),
                    confidence=float(getattr(lm, 'confidence', 1.0)),
                    visible=bool(getattr(lm, 'visible', True))
                ))
            else:
                # 如果格式不对，添加默认值
                formatted.append(Keypoint(
                    x=0.0, y=0.0, z=0.0,
                    confidence=0.0, visible=False
                ))
        
        return formatted
        
    except Exception as e:
        logger.error(f"关键点格式化失败: {e}")
        return landmarks


def optimize_image_for_sync(image: np.ndarray, target_quality: float = 0.7) -> np.ndarray:
    """
    为同步优化图像质量
    
    Args:
        image: 输入图像
        target_quality: 目标质量 (0.1-1.0)
        
    Returns:
        优化后的图像
    """
    try:
        if image is None:
            return image
            
        # 根据质量调整图像大小
        if target_quality < 0.8:
            h, w = image.shape[:2]
            new_h = int(h * target_quality)
            new_w = int(w * target_quality)
            image = cv2.resize(image, (new_w, new_h))
            
        return image
        
    except Exception as e:
        logger.error(f"图像优化失败: {e}")
        return image
