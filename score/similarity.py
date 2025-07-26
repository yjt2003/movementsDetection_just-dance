import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# =================== 欧氏距离法（原有） ===================
def center_landmarks(landmarks):
    """
    将关键点坐标以鼻子为原点中心化，返回 numpy 数组格式的中心化坐标。
    :param landmarks: MediaPipe landmarks
    :return: (33, 3) numpy array 中心化后的坐标（x, y, z）
    """
    if not landmarks or len(getattr(landmarks, 'landmark', [])) != 33:
        return None
    center = landmarks.landmark[0]
    cx, cy, cz = center.x, center.y, center.z
    centered = np.array([
        [lm.x - cx, lm.y - cy, lm.z - cz] for lm in landmarks.landmark
    ])
    return centered

def normalize_landmarks(landmarks_np):
    """
    对 landmark 坐标进行归一化，以 y 方向范围（估计身高）为标准。
    :param landmarks_np: (33, 3) numpy array，中心化后的坐标
    :return: 归一化后的 numpy array，单位一致
    """
    y_coords = landmarks_np[:, 1]
    height = y_coords.max() - y_coords.min()
    if height < 1e-5:
        return None
    normalized = landmarks_np / height
    return normalized

def euclidean_similarity(landmarks1, landmarks2):
    """
    欧氏距离法相似度
    """
    if (
        not landmarks1 or not landmarks2 or
        len(getattr(landmarks1, 'landmark', [])) != 33 or
        len(getattr(landmarks2, 'landmark', [])) != 33
    ):
        print("Landmarks not fully detected or incomplete.")
        return None
    lms1 = center_landmarks(landmarks1)
    lms2 = center_landmarks(landmarks2)
    if lms1 is None or lms2 is None:
        print("Centering failed.")
        return None
    norm1 = normalize_landmarks(lms1)
    norm2 = normalize_landmarks(lms2)
    if norm1 is None or norm2 is None:
        print("Normalization failed.")
        return None
    distances = np.linalg.norm(norm1 - norm2, axis=1)  # shape: (33,)
    LANDMARK_WEIGHTS = np.array([
        1.0, 0.5, 0.5, 0.5, 0.5, 1.5, 2.0, 2.0, 2.0, 2.0,
        1.0, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2,
        0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2,
        0.2, 0.2
    ])
    weighted_avg_distance = np.average(distances, weights=LANDMARK_WEIGHTS)
    max_reasonable_distance = 0.3
    similarity = max(0.0, 1.0 - (weighted_avg_distance / max_reasonable_distance))
    return similarity

# =================== 余弦法 ===================
def compute_cosine_similarity(landmarks1, landmarks2):
    """
    landmarks1/2: list of 33个关键点对象，需有x/y属性
    """
    if len(landmarks1) != len(landmarks2):
        print("You must have the vectors of same length of vector")
        return None
    vec1 = np.array([[lm.x, lm.y] for lm in landmarks1]).flatten()
    vec2 = np.array([[lm.x, lm.y] for lm in landmarks2]).flatten()
    vec1 = vec1 - np.mean(vec1)
    vec2 = vec2 - np.mean(vec2)
    similarity = cosine_similarity([vec1], [vec2])[0][0]
    # 归一化到[0,1]
    similarity = (similarity + 1) / 2
    return similarity

# =================== 角度法 ===================
MP_LANDMARKS = {
    "nose": 0, "left_eye_inner": 1, "left_eye": 2, "left_eye_outer": 3, "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6,
    "left_ear": 7, "right_ear": 8, "mouth_left": 9, "mouth_right": 10, "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14, "left_wrist": 15, "right_wrist": 16, "left_pinky": 17, "right_pinky": 18,
    "left_index": 19, "right_index": 20, "left_thumb": 21, "right_thumb": 22, "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26, "left_ankle": 27, "right_ankle": 28, "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}
LIMBS = [
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]
LIMB_IDXS = np.array([(MP_LANDMARKS[a], MP_LANDMARKS[b]) for a, b in LIMBS])

def _unit_vectors(points: np.ndarray, idx_pairs: np.ndarray) -> np.ndarray:
    vecs = points[idx_pairs[:, 1]] - points[idx_pairs[:, 0]]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    return vecs / norms

def _angle_diffs(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    cos = np.sum(u * v, axis=1).clip(-1.0, 1.0)
    return np.arccos(cos)

def pose_similarity_2d(landmarks1, landmarks2, aggregate="mean"):
    """
    landmarks1/2: list of 33个关键点对象，需有x/y属性
    """
    arr1 = np.array([[lm.x, lm.y] for lm in landmarks1])
    arr2 = np.array([[lm.x, lm.y] for lm in landmarks2])
    if arr1.shape != (33, 2) or arr2.shape != (33, 2):
        print("Each landmark set must have shape (33, 2)")
        return None
    u = _unit_vectors(arr1, LIMB_IDXS)
    v = _unit_vectors(arr2, LIMB_IDXS)
    angles = _angle_diffs(u, v)
    sims = 1.0 - angles / np.pi
    if aggregate == "mean":
        score = sims.mean()
    elif aggregate == "median":
        score = float(np.median(sims))
    else:
        raise ValueError("aggregate must be 'mean' or 'median'")
    return float(score)

# =================== 统一接口 ===================
def calculate_pose_similarity(landmarks1, landmarks2, method="hybrid"):
    """
    method: 'euclidean'（原有欧氏距离法）、'cosine'（余弦法）、'angle'（角度法）、'hybrid'（三者平均）
    """
    results = {}
    # 欧氏距离法
    results['euclidean'] = euclidean_similarity(landmarks1, landmarks2)
    # 余弦法
    results['cosine'] = compute_cosine_similarity(landmarks1, landmarks2)
    # 角度法
    results['angle'] = pose_similarity_2d(landmarks1, landmarks2)
    if method == "hybrid":
        valid_scores = [v for v in results.values() if v is not None]
        if not valid_scores:
            return None
        return float(np.mean(valid_scores))
    else:
        return results.get(method)