from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time
import numpy as np
import sys
import os

# 添加score路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'score'))

from .model import Keypoint, SimilarityResult
from .utils import pad_landmarks
from score.similarity import calculate_pose_similarity, center_landmarks, normalize_landmarks
from score.score_pose import score_pose
from score.motion_match import match_motion_to_beats
from score.average_similarity import CumulativeScore

router = APIRouter()


class CompareRequest(BaseModel):
    user_landmarks: List[Keypoint]
    ref_landmarks: List[Keypoint]
    beat_times: List[float] = []
    current_time: float = 0.0
    window: int = 0  # 时间窗口（帧数）


class BatchCompareRequest(BaseModel):
    user_landmarks_sequence: List[List[Keypoint]]
    ref_landmarks_sequence: List[List[Keypoint]]
    beat_times: List[float] = []
    fps: float = 30.0


class ScoreAnalysisRequest(BaseModel):
    scores: List[float]
    beat_times: List[float] = []
    fps: float = 30.0


class DetailedScoreResult(BaseModel):
    similarity: float
    pose_score: float
    rhythm_score: float
    hand_score: float
    total_score: float
    delta_t: float
    beat_match: bool
    confidence: float


class BatchScoreResult(BaseModel):
    frame_scores: List[DetailedScoreResult]
    average_score: float
    max_score: float
    min_score: float
    rhythm_accuracy: float
    total_frames: int


@router.post('/compare', response_model=SimilarityResult)
async def compare_pose(req: CompareRequest):
    """
    比较两个姿态的相似度
    """
    try:
        # 补齐关键点到33个
        user_lm = pad_landmarks(req.user_landmarks, 33)
        ref_lm = pad_landmarks(req.ref_landmarks, 33)

        # 计算姿态相似度
        pose_score = calculate_pose_similarity(user_lm, ref_lm) or 0.0

        # 计算节奏分数
        rhythm_score = 0.0
        delta_t = 1.0

        if req.beat_times:
            # 找到最近的节拍点
            delta_t = min([abs(req.current_time - t) for t in req.beat_times])
            # 在0.4秒窗口内的节拍认为是准确的
            rhythm_score = max(0, 1 - delta_t / 0.4)

        # 计算总分
        total_score = score_pose(pose_score, delta_t if req.beat_times else 1.0)

        return SimilarityResult(
            similarity=pose_score,
            pose_score=pose_score,
            rhythm_score=rhythm_score,
            total_score=total_score,
            delta_t=delta_t
        )

    except Exception as e:
        print(f"姿态比较错误: {e}")
        raise HTTPException(status_code=500, detail=f"姿态比较失败: {str(e)}")


@router.post('/compare_batch', response_model=BatchScoreResult)
async def compare_pose_batch(req: BatchCompareRequest):
    """
    批量比较姿态序列
    """
    try:
        if len(req.user_landmarks_sequence) != len(req.ref_landmarks_sequence):
            raise HTTPException(status_code=400, detail="用户和参考姿态序列长度不匹配")

        frame_scores = []
        total_frames = len(req.user_landmarks_sequence)

        for i, (user_lm, ref_lm) in enumerate(zip(req.user_landmarks_sequence, req.ref_landmarks_sequence)):
            # 计算当前帧的时间
            current_time = i / req.fps

            # 创建单帧比较请求
            frame_req = CompareRequest(
                user_landmarks=user_lm,
                ref_landmarks=ref_lm,
                beat_times=req.beat_times,
                current_time=current_time
            )

            # 计算分数
            result = await compare_pose(frame_req)

            # 额外计算手势分数和置信度
            hand_score = result.pose_score * 0.8  # 简化的手势分数
            confidence = min(1.0, result.pose_score + 0.1)  # 简化的置信度
            beat_match = result.rhythm_score > 0.6

            frame_scores.append(DetailedScoreResult(
                similarity=result.similarity,
                pose_score=result.pose_score,
                rhythm_score=result.rhythm_score,
                hand_score=hand_score,
                total_score=result.total_score,
                delta_t=result.delta_t,
                beat_match=beat_match,
                confidence=confidence
            ))

        # 计算统计信息
        total_scores = [score.total_score for score in frame_scores]
        rhythm_scores = [score.rhythm_score for score in frame_scores]

        avg_score = sum(total_scores) / len(total_scores) if total_scores else 0.0
        max_score = max(total_scores) if total_scores else 0.0
        min_score = min(total_scores) if total_scores else 0.0

        # 计算节奏准确率
        rhythm_accuracy = sum(1 for score in rhythm_scores if score > 0.6) / len(
            rhythm_scores) if rhythm_scores else 0.0

        return BatchScoreResult(
            frame_scores=frame_scores,
            average_score=avg_score,
            max_score=max_score,
            min_score=min_score,
            rhythm_accuracy=rhythm_accuracy,
            total_frames=total_frames
        )

    except Exception as e:
        print(f"批量姿态比较错误: {e}")
        raise HTTPException(status_code=500, detail=f"批量姿态比较失败: {str(e)}")


@router.post('/analyze_performance')
async def analyze_performance(req: ScoreAnalysisRequest):
    """
    分析表现统计
    """
    try:
        scores = req.scores
        if not scores:
            raise HTTPException(status_code=400, detail="分数列表为空")

        # 基本统计
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        std_score = np.std(scores)

        # 计算趋势
        if len(scores) >= 2:
            trend = (scores[-1] - scores[0]) / len(scores)
        else:
            trend = 0.0

        # 计算稳定性
        stability = 1.0 - (std_score / max(avg_score, 0.1))
        stability = max(0.0, min(1.0, stability))

        # 性能等级
        if avg_score >= 0.9:
            performance_level = "传奇"
        elif avg_score >= 0.75:
            performance_level = "大师"
        elif avg_score >= 0.6:
            performance_level = "优秀"
        elif avg_score >= 0.4:
            performance_level = "良好"
        else:
            performance_level = "需要练习"

        # 节奏分析
        rhythm_analysis = {}
        if req.beat_times:
            frame_times = [i / req.fps for i in range(len(scores))]
            rhythm_matches = []

            for frame_time in frame_times:
                # 找到最近的节拍点
                if req.beat_times:
                    delta_t = min([abs(frame_time - bt) for bt in req.beat_times])
                    rhythm_matches.append(delta_t < 0.4)  # 0.4秒内算匹配

            rhythm_analysis = {
                "rhythm_accuracy": sum(rhythm_matches) / len(rhythm_matches) if rhythm_matches else 0.0,
                "total_beats": len(req.beat_times),
                "matched_beats": sum(rhythm_matches)
            }

        return {
            "statistics": {
                "average_score": avg_score,
                "max_score": max_score,
                "min_score": min_score,
                "standard_deviation": std_score,
                "trend": trend,
                "stability": stability,
                "total_frames": len(scores)
            },
            "performance": {
                "level": performance_level,
                "score_distribution": {
                    "excellent": sum(1 for s in scores if s >= 0.9),
                    "good": sum(1 for s in scores if 0.7 <= s < 0.9),
                    "average": sum(1 for s in scores if 0.5 <= s < 0.7),
                    "poor": sum(1 for s in scores if s < 0.5)
                }
            },
            "rhythm_analysis": rhythm_analysis,
            "recommendations": generate_recommendations(avg_score, stability, rhythm_analysis)
        }

    except Exception as e:
        print(f"性能分析错误: {e}")
        raise HTTPException(status_code=500, detail=f"性能分析失败: {str(e)}")


@router.post('/calculate_similarity')
async def calculate_similarity_only(user_landmarks: List[Keypoint], ref_landmarks: List[Keypoint]):
    """
    仅计算姿态相似度
    """
    try:
        user_lm = pad_landmarks(user_landmarks, 33)
        ref_lm = pad_landmarks(ref_landmarks, 33)

        similarity = calculate_pose_similarity(user_lm, ref_lm) or 0.0

        return {
            "similarity": similarity,
            "normalized": similarity,
            "confidence": min(1.0, similarity + 0.1)
        }

    except Exception as e:
        print(f"相似度计算错误: {e}")
        raise HTTPException(status_code=500, detail=f"相似度计算失败: {str(e)}")


@router.get('/score_metrics')
async def get_score_metrics():
    """
    获取评分指标说明
    """
    return {
        "metrics": {
            "pose_score": {
                "description": "姿态相似度分数",
                "range": [0.0, 1.0],
                "weight": 0.6
            },
            "rhythm_score": {
                "description": "节奏匹配分数",
                "range": [0.0, 1.0],
                "weight": 0.3
            },
            "hand_score": {
                "description": "手势准确度分数",
                "range": [0.0, 1.0],
                "weight": 0.1
            }
        },
        "total_score_formula": "weighted_sum(pose_score, rhythm_score, hand_score)",
        "performance_levels": {
            "传奇": [0.9, 1.0],
            "大师": [0.75, 0.9],
            "优秀": [0.6, 0.75],
            "良好": [0.4, 0.6],
            "需要练习": [0.0, 0.4]
        }
    }


def generate_recommendations(avg_score: float, stability: float, rhythm_analysis: dict) -> List[str]:
    """
    生成改进建议
    """
    recommendations = []

    if avg_score < 0.5:
        recommendations.append("建议多练习基本姿态，提高动作准确性")

    if stability < 0.7:
        recommendations.append("注意保持动作的稳定性和连贯性")

    if rhythm_analysis and rhythm_analysis.get("rhythm_accuracy", 0) < 0.6:
        recommendations.append("加强节奏感训练，跟上音乐节拍")

    if avg_score >= 0.8:
        recommendations.append("表现很棒！可以尝试更复杂的舞蹈动作")

    if not recommendations:
        recommendations.append("继续保持，稳步提升")

    return recommendations

