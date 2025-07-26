
def score_pose(similarity, delta_t, timing_weight=0.4):
    """
    similarity: 姿态相似度 (0~1)
    delta_t: 用户动作相对于参考动作的时间差（秒）
    timing_weight: 时间因素在评分中的权重
    """
    max_tolerance = 0.4  # 允许最大时间偏差（秒）
    timing_score = max(0, 1 - abs(delta_t) / max_tolerance)

    final_score = similarity * (1 - timing_weight) + timing_score * timing_weight
    return final_score
