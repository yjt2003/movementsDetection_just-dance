# detector/pose_detector.py
# 核心姿态检测模块

import cv2
import numpy as np
import mediapipe as mp
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. 数据结构定义
# =============================================================================

class DetectorType(Enum):
    """检测器类型枚举"""
    MEDIAPIPE = "mediapipe"
    YOLOV8 = "yolov8"
    HYBRID = "hybrid"


@dataclass
class Keypoint:
    """关键点数据结构"""
    x: float
    y: float
    confidence: float
    visible: bool = True
    predicted: bool = False  # 是否为预测点

    def to_dict(self) -> Dict[str, Any]:
        return {
            'x': self.x,
            'y': self.y,
            'confidence': self.confidence,
            'visible': self.visible,
            'predicted': self.predicted
        }


@dataclass
class Hand:
    """
    手势关键点数据结构
    """
    handedness: str  # 'Left' or 'Right'
    keypoints: List[Keypoint]
    confidence: float

# 修改Person，增加hands字段
@dataclass
class Person:
    id: int
    keypoints: List[Keypoint]
    bbox: Tuple[float, float, float, float]
    confidence: float
    hands: List[Hand]  # 新增
    timestamp: float
    detector_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'keypoints': [kp.to_dict() for kp in self.keypoints],
            'bbox': list(self.bbox),
            'confidence': self.confidence,
            'hands': [
                {
                    'handedness': hand.handedness,
                    'confidence': hand.confidence,
                    'keypoints': [kp.to_dict() for kp in hand.keypoints]
                } for hand in self.hands
            ],
            'timestamp': self.timestamp,
            'detector_type': self.detector_type
        }


# =============================================================================
# 2. 基础检测器接口
# =============================================================================

class BasePoseDetector:
    """姿态检测器基类"""

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.detection_count = 0

    def detect(self, frame: np.ndarray) -> List[Person]:
        """检测图像中的人员姿态"""
        raise NotImplementedError

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        raise NotImplementedError

    def cleanup(self):
        """清理资源"""
        pass


# =============================================================================
# 3. MediaPipe检测器实现
# =============================================================================

class MediaPipeDetector(BasePoseDetector):
    """MediaPipe全身+手势检测器"""

    def __init__(self, confidence_threshold: float = 0.5, model_complexity: int = 1):
        super().__init__(confidence_threshold)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold
        )
        self.mp_drawing = mp.solutions.drawing_utils
        # 新增手势检测器
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold
        )
        logger.info(f"✅ MediaPipe全身+手势检测器初始化完成 (confidence={confidence_threshold})")

    def detect(self, frame: np.ndarray) -> List[Person]:
        try:
            self.detection_count += 1
            height, width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            detected_persons = []
            if results.pose_landmarks:
                keypoints = []
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    kp = Keypoint(
                        x=landmark.x * width,
                        y=landmark.y * height,
                        confidence=landmark.visibility,
                        visible=landmark.visibility > self.confidence_threshold
                    )
                    keypoints.append(kp)
                bbox = self._calculate_bbox(keypoints, width, height)
                # 手势检测（对bbox区域裁剪）
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                person_img = rgb_frame[y1:y2, x1:x2]
                hands = self._detect_hands(person_img, x1, y1)
                person = Person(
                    id=0,
                    keypoints=keypoints,
                    bbox=bbox,
                    confidence=np.mean([kp.confidence for kp in keypoints if kp.visible]),
                    hands=hands,
                    timestamp=time.time(),
                    detector_type="mediapipe"
                )
                detected_persons.append(person)
                logger.debug(f"MediaPipe检测到1个人，{len([kp for kp in keypoints if kp.visible])}个有效关键点，{len(hands)}只手")
            return detected_persons
        except Exception as e:
            logger.error(f"MediaPipe检测错误: {e}")
            return []

    def _calculate_bbox(self, keypoints: List[Keypoint], width: int, height: int) -> Tuple[float, float, float, float]:
        """计算边界框"""
        valid_points = [(kp.x, kp.y) for kp in keypoints if kp.visible]

        if not valid_points:
            return (0, 0, width, height)

        xs, ys = zip(*valid_points)
        x1, y1 = max(0, min(xs) - 20), max(0, min(ys) - 20)
        x2, y2 = min(width, max(xs) + 20), min(height, max(ys) + 20)

        return (x1, y1, x2, y2)

    def _detect_hands(self, img: np.ndarray, x_offset: int, y_offset: int) -> List[Hand]:
        """
        对输入图像检测手势，并将关键点坐标映射回原图
        """
        hands_result = self.hands.process(img)
        hands = []
        if hands_result.multi_hand_landmarks and hands_result.multi_handedness:
            for hand_landmarks, handedness in zip(hands_result.multi_hand_landmarks, hands_result.multi_handedness):
                kp_list = []
                for lm in hand_landmarks.landmark:
                    kp = Keypoint(
                        x=lm.x * img.shape[1] + x_offset,
                        y=lm.y * img.shape[0] + y_offset,
                        confidence=lm.z,  # hands没有visibility，z可用作置信度参考
                        visible=True
                    )
                    kp_list.append(kp)
                hand = Hand(
                    handedness=handedness.classification[0].label,
                    keypoints=kp_list,
                    confidence=handedness.classification[0].score
                )
                hands.append(hand)
        return hands

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "MediaPipe Pose",
            "type": DetectorType.MEDIAPIPE.value,
            "keypoints": 33,
            "real_time": True,
            "detection_count": self.detection_count
        }

    def cleanup(self):
        if hasattr(self, 'pose'):
            self.pose.close()
        if hasattr(self, 'hands'):
            self.hands.close()
        logger.info("MediaPipe资源已清理")


# =============================================================================
# 4. YOLOv8检测器实现 (可选)
# =============================================================================

class YOLOv8Detector(BasePoseDetector):
    """YOLOv8姿态检测器 (可选组件)"""

    def __init__(self, model_path: str = "yolov8n-pose.pt", confidence_threshold: float = 0.3):
        super().__init__(confidence_threshold)

        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.available = True
            logger.info(f"✅ YOLOv8检测器初始化完成 (model={model_path})")
        except ImportError:
            logger.warning("⚠️  YOLOv8不可用，请安装: pip install ultralytics")
            self.available = False
        except Exception as e:
            logger.error(f"❌ YOLOv8初始化失败: {e}")
            self.available = False

    def detect(self, frame: np.ndarray) -> List[Person]:
        """YOLOv8姿态检测"""
        if not self.available:
            return []

        try:
            self.detection_count += 1
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)

            detected_persons = []

            for result in results:
                if result.keypoints is not None:
                    keypoints_data = result.keypoints.xyn.cpu().numpy()
                    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else None

                    for i, person_kpts in enumerate(keypoints_data):
                        keypoints = []

                        # 处理17个COCO关键点
                        for j, (x, y) in enumerate(person_kpts):
                            kp = Keypoint(
                                x=x * frame.shape[1],
                                y=y * frame.shape[0],
                                confidence=0.8,  # YOLOv8没有直接的关键点置信度
                                visible=x > 0 and y > 0
                            )
                            keypoints.append(kp)

                        # 边界框
                        if boxes is not None and i < len(boxes):
                            bbox = tuple(boxes[i])
                        else:
                            bbox = self._calculate_bbox_yolo(keypoints, frame.shape[1], frame.shape[0])

                        person = Person(
                            id=i,
                            keypoints=keypoints,
                            bbox=bbox,
                            confidence=0.8,
                            timestamp=time.time(),
                            detector_type="yolov8"
                        )

                        detected_persons.append(person)

            logger.debug(f"YOLOv8检测到{len(detected_persons)}个人")
            return detected_persons

        except Exception as e:
            logger.error(f"YOLOv8检测错误: {e}")
            return []

    def _calculate_bbox_yolo(self, keypoints: List[Keypoint], width: int, height: int) -> Tuple[
        float, float, float, float]:
        """为YOLOv8计算边界框"""
        valid_points = [(kp.x, kp.y) for kp in keypoints if kp.visible]

        if not valid_points:
            return (0, 0, width, height)

        xs, ys = zip(*valid_points)
        return (min(xs), min(ys), max(xs), max(ys))

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "YOLOv8 Pose",
            "type": DetectorType.YOLOV8.value,
            "keypoints": 17,
            "real_time": True,
            "available": self.available,
            "detection_count": self.detection_count
        }


# =============================================================================
# 5. 混合检测器 (智能切换)
# =============================================================================

class HybridDetector(BasePoseDetector):
    """混合检测器 - 智能选择最优模型"""

    def __init__(self, confidence_threshold: float = 0.5):
        super().__init__(confidence_threshold)

        # 初始化两个检测器
        self.mediapipe = MediaPipeDetector(confidence_threshold)
        self.yolov8 = YOLOv8Detector(confidence_threshold=confidence_threshold)

        # 当前使用的检测器
        self.current_detector = self.mediapipe
        self.switch_threshold = 0.7  # 切换阈值

        logger.info("✅ 混合检测器初始化完成")

    def detect(self, frame: np.ndarray) -> List[Person]:
        """智能检测"""
        try:
            # 分析场景选择最优检测器
            optimal_detector = self._select_optimal_detector(frame)

            if optimal_detector != self.current_detector:
                self.current_detector = optimal_detector
                logger.info(f"🔄 检测器切换到: {self.current_detector.get_model_info()['name']}")

            return self.current_detector.detect(frame)

        except Exception as e:
            logger.error(f"混合检测器错误: {e}")
            # 回退到MediaPipe
            return self.mediapipe.detect(frame)

    def _select_optimal_detector(self, frame: np.ndarray) -> BasePoseDetector:
        """根据场景选择最优检测器"""
        # 简单的启发式规则
        height, width = frame.shape[:2]

        # 计算亮度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0

        # 选择策略
        if brightness < 0.3:  # 低光环境
            return self.mediapipe  # MediaPipe在低光下表现更好
        elif width * height > 1920 * 1080:  # 高分辨率
            return self.yolov8 if self.yolov8.available else self.mediapipe
        else:
            return self.mediapipe  # 默认使用MediaPipe

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "Hybrid Detector",
            "type": DetectorType.HYBRID.value,
            "current": self.current_detector.get_model_info(),
            "mediapipe_available": True,
            "yolov8_available": self.yolov8.available,
            "detection_count": self.detection_count
        }

    def cleanup(self):
        """清理所有检测器资源"""
        self.mediapipe.cleanup()
        if self.yolov8.available:
            self.yolov8.cleanup()


# =============================================================================
# 6. 检测器工厂
# =============================================================================

class DetectorFactory:
    """检测器工厂类"""

    @staticmethod
    def create_detector(detector_type: DetectorType, **kwargs) -> BasePoseDetector:
        """创建检测器实例"""
        try:
            if detector_type == DetectorType.MEDIAPIPE:
                return MediaPipeDetector(**kwargs)
            elif detector_type == DetectorType.YOLOV8:
                return YOLOv8Detector(**kwargs)
            elif detector_type == DetectorType.HYBRID:
                return HybridDetector(**kwargs)
            else:
                raise ValueError(f"不支持的检测器类型: {detector_type}")

        except Exception as e:
            logger.error(f"创建检测器失败: {e}")
            # 回退到MediaPipe
            logger.info("回退到MediaPipe检测器")
            return MediaPipeDetector(**kwargs)

    @staticmethod
    def get_available_detectors() -> List[DetectorType]:
        """获取可用的检测器类型"""
        available = [DetectorType.MEDIAPIPE]  # MediaPipe总是可用

        # 检查YOLOv8是否可用
        try:
            import ultralytics
            available.append(DetectorType.YOLOV8)
            available.append(DetectorType.HYBRID)
        except ImportError:
            pass

        return available


# =============================================================================
# 7. 主检测管理器
# =============================================================================

class PoseDetectionManager:
    """姿态检测管理器 - 统一接口"""

    def __init__(self, detector_type: DetectorType = DetectorType.MEDIAPIPE,
                 confidence_threshold: float = 0.5):

        self.detector = DetectorFactory.create_detector(detector_type,
                                                        confidence_threshold=confidence_threshold)
        self.detection_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'average_persons_per_frame': 0,
            'start_time': time.time()
        }

        logger.info(f"🚀 姿态检测管理器启动: {self.detector.get_model_info()['name']}")

    def detect_poses(self, frame: np.ndarray) -> Tuple[List[Person], Dict[str, Any]]:
        """检测姿态并返回统计信息"""
        start_time = time.time()

        # 执行检测
        persons = self.detector.detect(frame)

        # 更新统计
        self._update_stats(persons, time.time() - start_time)

        # 检测信息
        detection_info = {
            'persons_count': len(persons),
            'processing_time_ms': (time.time() - start_time) * 1000,
            'detector_info': self.detector.get_model_info(),
            'frame_size': frame.shape[:2]
        }

        return persons, detection_info

    def _update_stats(self, persons: List[Person], processing_time: float):
        """更新检测统计"""
        self.detection_stats['total_detections'] += 1
        if persons:
            self.detection_stats['successful_detections'] += 1

        # 计算平均人数
        total_persons = sum(len(persons) for persons in [persons])  # 简化版本
        self.detection_stats['average_persons_per_frame'] = (
                total_persons / self.detection_stats['total_detections']
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        runtime = time.time() - self.detection_stats['start_time']

        return {
            **self.detection_stats,
            'runtime_seconds': runtime,
            'detection_rate': self.detection_stats['total_detections'] / runtime if runtime > 0 else 0,
            'success_rate': (self.detection_stats['successful_detections'] /
                             max(self.detection_stats['total_detections'], 1)) * 100,
            'detector_info': self.detector.get_model_info()
        }

    def switch_detector(self, new_detector_type: DetectorType):
        """切换检测器"""
        try:
            old_detector = self.detector
            self.detector = DetectorFactory.create_detector(new_detector_type)
            old_detector.cleanup()
            logger.info(f"🔄 检测器已切换到: {self.detector.get_model_info()['name']}")
        except Exception as e:
            logger.error(f"切换检测器失败: {e}")

    def cleanup(self):
        """清理资源"""
        if self.detector:
            self.detector.cleanup()
        logger.info("姿态检测管理器已清理")


# =============================================================================
# 8. 使用示例和测试
# =============================================================================

def test_pose_detector():
    """测试姿态检测器"""

    print("🧪 开始测试姿态检测器...")

    # 创建检测管理器
    manager = PoseDetectionManager(DetectorType.MEDIAPIPE)

    # 模拟检测
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    try:
        persons, detection_info = manager.detect_poses(test_frame)

        print(f"✅ 检测完成:")
        print(f"   检测到人数: {detection_info['persons_count']}")
        print(f"   处理时间: {detection_info['processing_time_ms']:.2f}ms")
        print(f"   检测器: {detection_info['detector_info']['name']}")

        # 性能统计
        stats = manager.get_performance_stats()
        print(f"📊 性能统计:")
        print(f"   成功率: {stats['success_rate']:.1f}%")
        print(f"   检测频率: {stats['detection_rate']:.1f} FPS")

    except Exception as e:
        print(f"❌ 测试失败: {e}")

    finally:
        manager.cleanup()


# =============================================================================
# 9.
# =============================================================================


if __name__ == "__main__":
    test_pose_detector()