from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
import time


class Keypoint(BaseModel):
    """关键点数据模型"""
    x: float = Field(default=0.0, description="X坐标")
    y: float = Field(default=0.0, description="Y坐标")
    z: float = Field(default=0.0, description="Z坐标（深度）")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="置信度")
    visible: bool = Field(default=True, description="是否可见")

    @validator('confidence')
    def validate_confidence(cls, v):
        return max(0.0, min(1.0, v))

    @validator('x', 'y', 'z')
    def validate_coordinates(cls, v):
        return float(v) if v is not None else 0.0


class Landmarks(BaseModel):
    """关键点集合"""
    landmark: List[Keypoint] = Field(default_factory=list, description="关键点列表")

    @validator('landmark')
    def validate_landmarks(cls, v):
        # 确保至少有一个关键点
        if not v:
            return [Keypoint()]
        return v


class HandLandmarks(BaseModel):
    """手部关键点"""
    landmarks: List[Keypoint] = Field(default_factory=list, description="手部关键点")
    handedness: str = Field(default="Unknown", description="左手或右手")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="检测置信度")


class FaceLandmarks(BaseModel):
    """面部关键点"""
    landmarks: List[Keypoint] = Field(default_factory=list, description="面部关键点")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="检测置信度")


class Person(BaseModel):
    """人物检测结果"""
    keypoints: List[Keypoint] = Field(default_factory=list, description="身体关键点")
    hands: List[HandLandmarks] = Field(default_factory=list, description="手部关键点")
    face: Optional[FaceLandmarks] = Field(default=None, description="面部关键点")
    bbox: List[float] = Field(default_factory=lambda: [0, 0, 0, 0], description="边界框")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="总体置信度")
    id: Optional[int] = Field(default=None, description="追踪ID")


class PoseResult(BaseModel):
    """姿态检测结果"""
    persons: List[Landmarks] = Field(default_factory=list, description="检测到的人物")
    processing_time_ms: float = Field(default=0.0, description="处理时间（毫秒）")
    success: bool = Field(default=True, description="是否成功")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    frame_id: Optional[int] = Field(default=None, description="帧ID")
    timestamp: float = Field(default_factory=time.time, description="时间戳")


class SimilarityResult(BaseModel):
    """相似度计算结果"""
    similarity: float = Field(default=0.0, ge=0.0, le=1.0, description="相似度")
    pose_score: float = Field(default=0.0, ge=0.0, le=1.0, description="姿态得分")
    rhythm_score: float = Field(default=0.0, ge=0.0, le=1.0, description="节奏得分")
    hand_score: float = Field(default=0.0, ge=0.0, le=1.0, description="手势得分")
    total_score: float = Field(default=0.0, ge=0.0, le=1.0, description="总得分")
    delta_t: float = Field(default=1.0, description="时间差")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="置信度")


class AnnotatedImage(BaseModel):
    """标注图片"""
    image_base64: str = Field(description="Base64编码的图片")
    annotations: Optional[Dict[str, Any]] = Field(default=None, description="标注信息")
    processing_time_ms: float = Field(default=0.0, description="处理时间")


class GameState(Enum):
    """游戏状态枚举"""
    IDLE = "idle"
    PREPARING = "preparing"
    PLAYING = "playing"
    PAUSED = "paused"
    FINISHED = "finished"


class DanceMove(BaseModel):
    """舞蹈动作"""
    name: str = Field(description="动作名称")
    emoji: str = Field(description="动作表情符号")
    duration: float = Field(default=2.0, description="动作持续时间（秒）")
    difficulty: str = Field(default="Medium", description="难度等级")
    keyframes: List[Landmarks] = Field(default_factory=list, description="关键帧")


class Dance(BaseModel):
    """舞蹈"""
    id: int = Field(description="舞蹈ID")
    name: str = Field(description="舞蹈名称")
    emoji: str = Field(description="舞蹈表情符号")
    difficulty: str = Field(default="Medium", description="难度等级")
    duration: float = Field(default=30.0, description="舞蹈时长（秒）")
    moves: List[DanceMove] = Field(default_factory=list, description="舞蹈动作序列")
    beat_times: List[float] = Field(default_factory=list, description="节拍时间点")
    reference_video: Optional[str] = Field(default=None, description="参考视频路径")


class GameSession(BaseModel):
    """游戏会话"""
    session_id: str = Field(description="会话ID")
    user_id: Optional[str] = Field(default=None, description="用户ID")
    state: GameState = Field(default=GameState.IDLE, description="游戏状态")
    selected_dance: Optional[Dance] = Field(default=None, description="选中的舞蹈")
    start_time: Optional[float] = Field(default=None, description="开始时间")
    end_time: Optional[float] = Field(default=None, description="结束时间")
    current_score: float = Field(default=0.0, description="当前得分")
    average_score: float = Field(default=0.0, description="平均得分")
    max_score: float = Field(default=0.0, description="最高得分")
    frame_count: int = Field(default=0, description="帧计数")
    settings: Dict[str, Any] = Field(default_factory=dict, description="游戏设置")


class ScoreUpdate(BaseModel):
    """分数更新"""
    session_id: str = Field(description="会话ID")
    frame_id: int = Field(description="帧ID")
    pose_score: float = Field(default=0.0, description="姿态得分")
    rhythm_score: float = Field(default=0.0, description="节奏得分")
    hand_score: float = Field(default=0.0, description="手势得分")
    total_score: float = Field(default=0.0, description="总得分")
    average_score: float = Field(default=0.0, description="平均得分")
    timestamp: float = Field(default_factory=time.time, description="时间戳")


class FrameData(BaseModel):
    """视频帧数据"""
    image_base64: str = Field(description="Base64编码的图片")
    frame_type: str = Field(default="webcam", description="帧类型")
    timestamp: float = Field(default_factory=time.time, description="时间戳")
    frame_id: Optional[int] = Field(default=None, description="帧ID")
    current_time: float = Field(default=0.0, description="当前时间")


class WebSocketMessage(BaseModel):
    """WebSocket消息"""
    event: str = Field(description="事件类型")
    data: Optional[Dict[str, Any]] = Field(default=None, description="消息数据")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    timestamp: float = Field(default_factory=time.time, description="时间戳")


class FrameResult(BaseModel):
    """帧处理结果"""
    event: str = Field(default="frame_result", description="事件类型")
    type: str = Field(description="帧类型")
    image: str = Field(description="处理后的图片")
    persons_detected: int = Field(default=0, description="检测到的人数")
    processing_time_ms: float = Field(default=0.0, description="处理时间")
    landmarks: Optional[List[Landmarks]] = Field(default=None, description="关键点数据")
    confidence: float = Field(default=1.0, description="检测置信度")
    frame_id: Optional[int] = Field(default=None, description="帧ID")


class BeatTiming(BaseModel):
    """节拍时间信息"""
    beat_times: List[float] = Field(default_factory=list, description="节拍时间点")
    tempo: float = Field(default=120.0, description="节拍速度(BPM)")
    confidence: float = Field(default=1.0, description="节拍检测置信度")
    total_beats: int = Field(default=0, description="总节拍数")


class PerformanceStats(BaseModel):
    """性能统计"""
    session_id: str = Field(description="会话ID")
    total_frames: int = Field(default=0, description="总帧数")
    average_fps: float = Field(default=0.0, description="平均FPS")
    average_latency: float = Field(default=0.0, description="平均延迟")
    total_score: float = Field(default=0.0, description="总得分")
    max_score: float = Field(default=0.0, description="最高得分")
    rhythm_accuracy: float = Field(default=0.0, description="节奏准确率")
    pose_accuracy: float = Field(default=0.0, description="姿态准确率")
    stability: float = Field(default=0.0, description="稳定性")
    consistency: float = Field(default=0.0, description="一致性")


class ErrorInfo(BaseModel):
    """错误信息"""
    error_code: str = Field(description="错误代码")
    message: str = Field(description="错误消息")
    details: Optional[Dict[str, Any]] = Field(default=None, description="错误详情")
    timestamp: float = Field(default_factory=time.time, description="时间戳")
    session_id: Optional[str] = Field(default=None, description="会话ID")


class SystemStatus(BaseModel):
    """系统状态"""
    status: str = Field(description="系统状态")
    active_sessions: int = Field(default=0, description="活跃会话数")
    cpu_usage: float = Field(default=0.0, description="CPU使用率")
    memory_usage: float = Field(default=0.0, description="内存使用率")
    fps: float = Field(default=0.0, description="平均FPS")
    detector_type: str = Field(default="mediapipe", description="检测器类型")
    version: str = Field(default="1.0.0", description="版本号")
    uptime: float = Field(default=0.0, description="运行时间")


class CalibrateRequest(BaseModel):
    """校准请求"""
    reference_poses: List[Landmarks] = Field(description="参考姿态")
    user_poses: List[Landmarks] = Field(description="用户姿态")
    calibration_type: str = Field(default="auto", description="校准类型")


class CalibrateResult(BaseModel):
    """校准结果"""
    success: bool = Field(description="校准是否成功")
    scale_factor: float = Field(default=1.0, description="缩放因子")
    offset_x: float = Field(default=0.0, description="X偏移")
    offset_y: float = Field(default=0.0, description="Y偏移")
    rotation: float = Field(default=0.0, description="旋转角度")
    confidence: float = Field(default=1.0, description="校准置信度")
    message: str = Field(default="", description="校准信息")


class VideoInfo(BaseModel):
    """视频信息"""
    width: int = Field(description="视频宽度")
    height: int = Field(description="视频高度")
    fps: float = Field(description="帧率")
    duration: float = Field(description="时长（秒）")
    total_frames: int = Field(description="总帧数")
    format: str = Field(description="视频格式")
    size_bytes: int = Field(description="文件大小（字节）")


class UploadVideoRequest(BaseModel):
    """上传视频请求"""
    video_base64: str = Field(description="Base64编码的视频")
    filename: str = Field(description="文件名")
    dance_id: Optional[int] = Field(default=None, description="舞蹈ID")
    extract_audio: bool = Field(default=True, description="是否提取音频")
    analyze_beats: bool = Field(default=True, description="是否分析节拍")


class UploadVideoResult(BaseModel):
    """上传视频结果"""
    success: bool = Field(description="上传是否成功")
    video_path: str = Field(description="视频路径")
    video_info: VideoInfo = Field(description="视频信息")
    beat_timing: Optional[BeatTiming] = Field(default=None, description="节拍信息")
    message: str = Field(default="", description="结果消息")


# 配置模型
class DetectorConfig(BaseModel):
    """检测器配置"""
    type: str = Field(default="mediapipe", description="检测器类型")
    confidence_threshold: float = Field(default=0.5, description="置信度阈值")
    max_persons: int = Field(default=1, description="最大人数")
    enable_hand_detection: bool = Field(default=False, description="启用手部检测")
    enable_face_detection: bool = Field(default=False, description="启用面部检测")


class GameConfig(BaseModel):
    """游戏配置"""
    max_session_duration: float = Field(default=300.0, description="最大会话时长（秒）")
    auto_save_interval: float = Field(default=10.0, description="自动保存间隔（秒）")
    score_update_interval: float = Field(default=1.0, description="分数更新间隔（秒）")
    enable_perfect_effects: bool = Field(default=True, description="启用完美特效")
    rhythm_window: float = Field(default=0.4, description="节奏窗口（秒）")


class AppConfig(BaseModel):
    """应用配置"""
    detector: DetectorConfig = Field(default_factory=DetectorConfig, description="检测器配置")
    game: GameConfig = Field(default_factory=GameConfig, description="游戏配置")
    debug_mode: bool = Field(default=False, description="调试模式")
    log_level: str = Field(default="INFO", description="日志级别")
    max_concurrent_sessions: int = Field(default=10, description="最大并发会话数")
