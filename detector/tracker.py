import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Any, Optional

class Track:
    def __init__(self, id: int, keypoints: np.ndarray, bbox: np.ndarray, frame_idx: int):
        self.id = id
        self.keypoints = keypoints  # shape: (num_keypoints, dim)
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.last_frame = frame_idx
        self.miss_count = 0

class Tracker:
    def __init__(self, max_missing: int = 10, dist_threshold: float = 80.0):
        self.max_missing = max_missing
        self.dist_threshold = dist_threshold
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0

    def reset(self):
        self.tracks = {}
        self.next_id = 0

    def update(self, detections: List[Dict[str, Any]], frame_idx: int) -> List[Dict[str, Any]]:
        """
        detections: List[{'keypoints': np.ndarray, 'bbox': np.ndarray}]
        返回: List[{'id': int, 'keypoints':..., 'bbox':...}]
        """
        if not detections:
            # 所有track miss计数+1
            for track in self.tracks.values():
                track.miss_count += 1
            self._remove_lost_tracks()
            return []

        det_kps = [det['keypoints'] for det in detections]
        det_bboxes = [det['bbox'] for det in detections]
        det_num = len(detections)
        track_ids = list(self.tracks.keys())
        track_kps = [self.tracks[tid].keypoints for tid in track_ids]

        # 计算距离矩阵（用关键点均值欧氏距离）
        if track_kps:
            cost_matrix = np.zeros((len(track_kps), det_num))
            for i, tkp in enumerate(track_kps):
                for j, dkp in enumerate(det_kps):
                    cost_matrix[i, j] = np.linalg.norm(np.mean(tkp, axis=0) - np.mean(dkp, axis=0))
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)

        assigned_tracks = set()
        assigned_dets = set()
        results = []
        # 分配ID
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.dist_threshold:
                tid = track_ids[r]
                self.tracks[tid].keypoints = det_kps[c]
                self.tracks[tid].bbox = det_bboxes[c]
                self.tracks[tid].last_frame = frame_idx
                self.tracks[tid].miss_count = 0
                assigned_tracks.add(tid)
                assigned_dets.add(c)
                results.append({'id': tid, 'keypoints': det_kps[c], 'bbox': det_bboxes[c]})
        # 新增未分配的检测
        for i in range(det_num):
            if i not in assigned_dets:
                tid = self.next_id
                self.tracks[tid] = Track(tid, det_kps[i], det_bboxes[i], frame_idx)
                self.next_id += 1
                results.append({'id': tid, 'keypoints': det_kps[i], 'bbox': det_bboxes[i]})
        # 未分配的track计数+1
        for tid in track_ids:
            if tid not in assigned_tracks:
                self.tracks[tid].miss_count += 1
        self._remove_lost_tracks()
        return results

    def _remove_lost_tracks(self):
        lost = [tid for tid, track in self.tracks.items() if track.miss_count > self.max_missing]
        for tid in lost:
            del self.tracks[tid]
