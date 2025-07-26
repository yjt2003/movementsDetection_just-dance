from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import time
import numpy as np
import sys
import os

# 添加detector路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'detector'))

from .model import Keypoint, Landmarks, PoseResult, AnnotatedImage
from .utils import decode_base64_image, encode_image_to_base64, pad_landmarks, draw_landmarks
from detector.pose_detector import DetectorType, PoseDetectionManager

router = APIRouter()

# 移除全局检测器实例，每次请求创建新的检测器实例


class ImageRequest(BaseModel):
    image_base64: str
    detector_type: str = "mediapipe"  # mediapipe, yolov8, hybrid


class BatchImageRequest(BaseModel):
    images: List[str]
    detector_type: str = "mediapipe"


@router.post('/detect', response_model=PoseResult)
async def detect_pose(req: ImageRequest):
    """
    检测单张图片中的人体姿态
    """
    try:
        start_time = time.time()

        # 解码图片
        frame = decode_base64_image(req.image_base64)
        if frame is None:
            raise HTTPException(status_code=400, detail="无效的图片数据")

        # 设置检测器类型
        detector_type = DetectorType.MEDIAPIPE
        if req.detector_type.lower() == "yolov8":
            detector_type = DetectorType.YOLOV8
        elif req.detector_type.lower() == "hybrid":
            detector_type = DetectorType.HYBRID

        # 每次请求创建新的检测器实例
        pose_manager = PoseDetectionManager(detector_type)
        print(f"🔍 API检测器创建: {id(pose_manager)} (类型: {detector_type.value})")

        # 进行姿态检测
        persons, det_info = pose_manager.detect_poses(frame)

        # 转换为API格式
        result_persons = []
        for person in persons:
            keypoints = []
            for kp in person.keypoints:
                keypoints.append(Keypoint(
                    x=float(kp.x),
                    y=float(kp.y),
                    z=getattr(kp, 'z', 0.0),
                    confidence=getattr(kp, 'confidence', 1.0),
                    visible=getattr(kp, 'visible', True)
                ))

            # 补齐到33个关键点
            keypoints = pad_landmarks(keypoints, 33)
            result_persons.append(Landmarks(landmark=keypoints))

        processing_time = (time.time() - start_time) * 1000

        return PoseResult(
            persons=result_persons,
            processing_time_ms=processing_time,
            success=True
        )

    except Exception as e:
        print(f"姿态检测错误: {e}")
        raise HTTPException(status_code=500, detail=f"姿态检测失败: {str(e)}")


@router.post('/detect_batch', response_model=List[PoseResult])
async def detect_pose_batch(req: BatchImageRequest):
    """
    批量检测多张图片中的人体姿态
    """
    results = []

    for image_data in req.images:
        try:
            single_req = ImageRequest(
                image_base64=image_data,
                detector_type=req.detector_type
            )
            result = await detect_pose(single_req)
            results.append(result)
        except Exception as e:
            # 对于失败的图片，返回空结果
            results.append(PoseResult(
                persons=[],
                processing_time_ms=0,
                success=False
            ))

    return results


@router.post('/annotate', response_model=AnnotatedImage)
async def annotate_pose(req: ImageRequest):
    """
    在图片上绘制姿态骨架
    """
    try:
        # 先检测姿态
        pose_result = await detect_pose(req)

        # 解码原始图片
        frame = decode_base64_image(req.image_base64)
        if frame is None:
            raise HTTPException(status_code=400, detail="无效的图片数据")

        # 绘制所有人的姿态
        for person in pose_result.persons:
            if person.landmark:
                frame = draw_landmarks(frame, person.landmark)

        # 编码结果图片
        result_image = encode_image_to_base64(frame)

        return AnnotatedImage(image_base64=result_image)

    except Exception as e:
        print(f"姿态标注错误: {e}")
        raise HTTPException(status_code=500, detail=f"姿态标注失败: {str(e)}")


@router.get('/health')
async def health_check():
    """
    健康检查接口
    """
    return {
        "status": "healthy",
        "detector_available": True,
        "timestamp": time.time()
    }


@router.get('/info')
async def get_detector_info():
    """
    获取检测器信息
    """
    return {
        "available_detectors": [dt.value for dt in DetectorType],
        "default_detector": DetectorType.MEDIAPIPE.value,
        "version": "1.0.0",
        "note": "每次请求创建新的检测器实例，避免冲突"
    }


@router.post('/test_detector')
async def test_detector(detector_type: str):
    """
    测试检测器类型
    """
    try:
        if detector_type.lower() == "mediapipe":
            new_type = DetectorType.MEDIAPIPE
        elif detector_type.lower() == "yolov8":
            new_type = DetectorType.YOLOV8
        elif detector_type.lower() == "hybrid":
            new_type = DetectorType.HYBRID
        else:
            raise HTTPException(status_code=400, detail="不支持的检测器类型")

        # 创建新检测器实例进行测试
        test_manager = PoseDetectionManager(new_type)
        
        return {
            "status": "success",
            "detector_type": new_type.value,
            "detector_id": id(test_manager),
            "message": f"检测器 {new_type.value} 测试成功"
        }

    except Exception as e:
        print(f"检测器测试错误: {e}")
        raise HTTPException(status_code=500, detail=f"检测器测试失败: {str(e)}")