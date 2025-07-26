import numpy as np
import cv2
import os
import mediapipe as mp

def extract_ref_frames(beat_frame_indices: np.ndarray, video_path: str, output_dir="ref_frames"):
    """
    根据节拍帧号（np.ndarray）提取参考视频中的图像帧。
    :param beat_frame_indices: ndarray，每个节拍对应的帧号（整数）
    :param video_path: 视频路径
    :param output_dir: 保存帧图像的文件夹
    :return: list[dict]，每帧包含 frame_path 和 frame_index
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    ref_frames = []

    for i, frame_idx in enumerate(beat_frame_indices):
        frame_idx = int(frame_idx)  # 保证为整数
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            filename = os.path.join(output_dir, f"ref_{i:03d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"✅ Saved {filename} (frame {frame_idx})")
            ref_frames.append({
                "frame_path": filename,
                "index": frame_idx
            })
        else:
            print(f"⚠️ Failed to read frame {frame_idx}")

    cap.release()
    return ref_frames


def extract_user_frames(video_path: str, output_dir="user_frames"):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_list = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = frame_idx / fps
        filename = f"{output_dir}/user_{frame_idx:04d}.jpg"
        cv2.imwrite(filename, frame)

        frame_list.append({
            "frame_path": filename,
            "time": time_sec,
            "index": frame_idx
        })

        frame_idx += 1

    cap.release()
    return frame_list  # 列表中包含每一帧的路径和时间信息


def match_user_frames_to_ref(ref_frames, user_frames, fps, pose_similarity):# video vs video
    """
    对每个参考帧，在用户帧序列中寻找最相似的姿态，并计算时间偏差。

    参数：
        ref_frames: List[Dict]，每个元素是 {"frame_path": ..., "index": ...}
        user_frames: List[Dict]，结构同上
        fps: float，视频帧率
        pose_similarity: 一个类或对象，包含 calculate_pose_similarity 方法

    返回：
        List[Dict]，每项包含：
            {
                "ref_index": int,
                "user_index": int,
                "similarity": float,
                "delta_t": float
            }
    """
    results = []

    mp_pose = mp.solutions.pose

    with mp_pose.Pose(static_image_mode=True) as pose_detector:
        for ref in ref_frames:
            ref_img = cv2.imread(ref["frame_path"])
            ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            ref_result = pose_detector.process(ref_rgb)
            ref_landmarks = ref_result.pose_landmarks

            best_match = None
            best_similarity = -1

            for user in user_frames:
                user_img = cv2.imread(user["frame_path"])
                user_rgb = cv2.cvtColor(user_img, cv2.COLOR_BGR2RGB)
                user_result = pose_detector.process(user_rgb)
                user_landmarks = user_result.pose_landmarks

                score = pose_similarity.calculate_pose_similarity(ref_landmarks, user_landmarks)
                if score is not None and score > best_similarity:
                    best_similarity = score
                    best_match = user

            if best_match is not None:
                delta_frame = best_match["index"] - ref["index"]
                delta_t = delta_frame / fps
                results.append({
                    "ref_index": ref["index"],
                    "user_index": best_match["index"],
                    "similarity": round(best_similarity, 4),
                    "delta_t": round(delta_t, 4)
                })
            else:
                results.append({
                    "ref_index": ref["index"],
                    "user_index": None,
                    "similarity": None,
                    "delta_t": None
                })

    return results


def match_live_cam_to_ref(ref_frames, fps, pose_similarity):
    """
    从实时摄像头捕获帧，与参考帧进行姿态相似度匹配并计算时间差。
    """
    cap = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose
    frame_index = 0
    user_time_records = []

    with mp_pose.Pose(static_image_mode=False) as pose_detector:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose_detector.process(rgb_frame)

            user_landmarks = result.pose_landmarks
            if not user_landmarks:
                frame_index += 1
                continue

            best_ref = None
            best_score = -1
            best_ref_index = -1

            # 对所有参考帧做匹配
            for ref in ref_frames:
                ref_img = cv2.imread(ref["frame_path"])
                ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                ref_result = pose_detector.process(ref_rgb)
                ref_landmarks = ref_result.pose_landmarks

                score = pose_similarity.calculate_pose_similarity(ref_landmarks, user_landmarks)
                if score is not None and score > best_score:
                    best_score = score
                    best_ref = ref
                    best_ref_index = ref["index"]

            # 记录用户帧与其最相似的参考帧的匹配信息
            if best_ref is not None:
                delta_frame = frame_index - best_ref_index
                delta_t = delta_frame / fps
                user_time_records.append({
                    "frame_index": frame_index,
                    "ref_index": best_ref_index,
                    "similarity": round(best_score, 4),
                    "delta_t": round(delta_t, 4)
                })
                print(f"🟢 Frame {frame_index} matched Ref {best_ref_index} | "
                      f"Score={best_score:.3f}, Δt={delta_t:.3f}s")

            frame_index += 1

            # ESC 退出
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    return user_time_records


def match_motion_to_beats(motion_times, beat_times, threshold=0.4):
    '''
    motion_times: list[float]，每个动作发生的时间（秒）
    beat_times: list[float]，节拍锚点时间（秒）
    threshold: float，匹配阈值（秒）
    返回：匹配分数（0~1），每个动作与最近节拍的时间差
    '''
    if not motion_times or not beat_times:
        return 0.0, []
    deltas = []
    for t in motion_times:
        delta = min([abs(t - b) for b in beat_times])
        deltas.append(delta)
    # 统计有多少动作在阈值内
    match_count = sum([1 for d in deltas if d <= threshold])
    score = match_count / len(motion_times)
    return score, deltas