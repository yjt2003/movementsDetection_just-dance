# Import necessary libraries
import sys
import os
import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import mediapipe as mp
import importlib.util
sys.path.append(os.path.join(os.path.dirname(__file__), 'detector'))
from detector.pose_detector import DetectorType, PoseDetectionManager
from detector.smoother import smooth_keypoints
from detector.fixer import fix_keypoints
from detector.tracker import Tracker
# 导入相似度计算
import numpy as np
from score.similarity import calculate_pose_similarity, center_landmarks, normalize_landmarks
from score.music_beat import mp4_2_mp3, get_beats
from score.motion_match import match_motion_to_beats
from score.score_pose import score_pose
from score.average_similarity import CumulativeScore
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from types import SimpleNamespace

# Suppress MediaPipe warnings
import logging

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 或 'Microsoft YaHei'，需本地有该字体
matplotlib.rcParams['axes.unicode_minus'] = False

logging.getLogger('mediapipe').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 标准骨架连接（MediaPipe 33点）
MEDIAPIPE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(27,29),
    (29,31),(26,28),(28,30),(30,32)
]
# YOLOv8/COCO 17点骨架连接
YOLOV8_CONNECTIONS = [
    (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),(6,8),(8,10),
    (5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
]

# 优化后的可视化函数
def draw_persons(frame, persons, detector_type):
    color_list = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]
    # 选择骨架连接
    if detector_type.value == 'mediapipe':
        connections = MEDIAPIPE_CONNECTIONS
    else:
        connections = YOLOV8_CONNECTIONS
    for idx, p in enumerate(persons):
        color = color_list[idx % len(color_list)]
        # 画骨架
        for pt1, pt2 in connections:
            if pt1 < len(p.keypoints) and pt2 < len(p.keypoints):
                kp1, kp2 = p.keypoints[pt1], p.keypoints[pt2]
                if kp1.visible and kp2.visible:
                    cv2.line(frame, (int(kp1.x), int(kp1.y)), (int(kp2.x), int(kp2.y)), color, 2)
        # 画关键点
        for kp in p.keypoints:
            if kp.visible:
                cv2.circle(frame, (int(kp.x), int(kp.y)), 4, color, -1)
        # 画手部
        for hand in getattr(p, 'hands', []):
            for kp in hand.keypoints:
                cv2.circle(frame, (int(kp.x), int(kp.y)), 2, (255,0,255), -1)
        # 显示ID
        if hasattr(p, 'id'):
            bbox = p.bbox
            cv2.putText(frame, f'ID:{p.id}', (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def calc_similarity(lm1, lm2):
    return calculate_pose_similarity(lm1, lm2)

def center_landmarks_safe(landmarks):
    return center_landmarks(landmarks)
def normalize_landmarks_safe(landmarks_np):
    return normalize_landmarks(landmarks_np)

class MediaPipePoseApp:
    def __init__(self, root,
                 detector_type=DetectorType.MEDIAPIPE,
                 smoother_method=None,
                 fixer_method=None,
                 tracker_enable=False):
        self.root = root
        self.root.title("MediaPipe Dance GUI - 多人评分")
        # 自动最大化窗口
        try:
            self.root.state('zoomed')  # Windows
        except:
            try:
                self.root.attributes('-zoomed', True)  # Linux/Mac
            except:
                pass
        self.root.configure(bg='lightgray')

        # 参数
        self.detector_type = detector_type
        self.smoother_method = smoother_method  # 'ema'/'kalman'/None
        self.fixer_method = fixer_method        # 'linear'/'symmetric'/None
        self.tracker_enable = tracker_enable

        # 分别为左/右流维护独立对象
        self.file_pose_manager = PoseDetectionManager(detector_type)
        self.file_tracker = Tracker() if tracker_enable else None
        self.cam_pose_manager = PoseDetectionManager(detector_type)
        self.cam_tracker = Tracker() if tracker_enable else None

        # ===== 新增：为每个ID创建独立的检测器 =====
        self.cam_pose_managers = {}  # 存储每个ID的独立检测器
        self.id_to_score = {}        # 存储每个ID的分数变量
        self.id_to_labels = {}       # 存储每个ID的标签控件
        self.score_container = None  # 分数显示容器
        self.cumulative_scores = {}  # 每个ID的累计分数

        # ===== 新增：参与人数设置 =====
        self.participant_count = 1   # 默认1人
        self.max_participants = 3    # 最大3人
        self.fixed_person_slots = {} # 固定的人员槽位

        self.running_file = False
        self.running_cam = False
        self.video_path = ""
        self.cap_file = None
        self.cap_cam = None
        self.show_video_frame = True

        self.similarity_calculator = SimilarityCalculator()
        self.last_file_landmarks = None  # 记录参考视频主舞者landmarks
        self.last_cam_landmarks = None   # 记录webcam主舞者landmarks
        self.last_similarity = None

        self.beat_times = []
        
        # 多人评分追踪（简化版）
        self.id_to_score = {}        # 存储每个ID的分数变量
        self.id_to_labels = {}       # 存储每个ID的标签控件
        self.cumulative_scores = {}  # 每个ID的累计分数
        
        # 总体分数图表
        self.fig, self.ax = plt.subplots(figsize=(6,3), dpi=100)
        self.score_canvas = None

        self.setup_gui()

    def setup_gui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg='lightgray')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left frame for reference video
        self.left_frame = tk.Frame(main_frame, bg='white', relief=tk.RAISED, bd=2)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Right frame for webcam
        self.right_frame = tk.Frame(main_frame, bg='white', relief=tk.RAISED, bd=2)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Left side - Reference Video
        tk.Label(self.left_frame, text="Reference Video", font=("Arial", 16, "bold"), bg='white').pack(pady=5)

        video_frame = tk.Frame(self.left_frame, bg='white')
        video_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Canvas自适应
        self.canvas_file = tk.Canvas(video_frame, bg="black")
        self.canvas_file.pack(fill=tk.BOTH, expand=True, pady=5)
        self.canvas_file.create_text(350, 262, text="No video loaded", fill="white", font=("Arial", 12))

        self.controls_file = tk.Frame(self.left_frame, bg='white')
        self.controls_file.pack(side=tk.BOTTOM, pady=10)
        tk.Button(self.controls_file, text="📁 Open Video", command=self.load_video,
                  bg='lightblue', font=("Arial", 10), width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(self.controls_file, text="▶️ Start Video", command=self.start_video,
                  bg='lightgreen', font=("Arial", 10), width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(self.controls_file, text="⏹️ Stop Video", command=self.stop_video,
                  bg='lightcoral', font=("Arial", 10), width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(self.controls_file, text="🔄 Show/Hide", command=self.toggle_video_display,
                  bg='lightyellow', font=("Arial", 10), width=12).pack(side=tk.LEFT, padx=2)

        # Right side - Webcam
        tk.Label(self.right_frame, text="Your Webcam", font=("Arial", 16, "bold"), bg='white').pack(pady=5)
        cam_frame = tk.Frame(self.right_frame, bg='white')
        cam_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.canvas_cam = tk.Canvas(cam_frame, bg="black")
        self.canvas_cam.pack(fill=tk.BOTH, expand=True, pady=5)
        self.canvas_cam.create_text(350, 262, text="Webcam not started", fill="white", font=("Arial", 12))

        self.controls_cam = tk.Frame(self.right_frame, bg='white')
        self.controls_cam.pack(side=tk.BOTTOM, pady=10)
        
        # 人数选择控件
        participant_frame = tk.Frame(self.controls_cam, bg='white')
        participant_frame.pack(pady=5)
        tk.Label(participant_frame, text="参与人数:", font=("Arial", 10), bg='white').pack(side=tk.LEFT, padx=5)
        self.participant_var = tk.StringVar(value="1")
        participant_combo = ttk.Combobox(participant_frame, textvariable=self.participant_var, 
                                       values=["1", "2", "3"], width=5, state="readonly")
        participant_combo.pack(side=tk.LEFT, padx=5)
        participant_combo.bind("<<ComboboxSelected>>", self.on_participant_count_changed)
        
        # 摄像头控制按钮
        button_frame = tk.Frame(self.controls_cam, bg='white')
        button_frame.pack(pady=5)
        tk.Button(button_frame, text="📷 Start Webcam", command=self.start_cam,
                  bg='lightgreen', font=("Arial", 10), width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="⏹️ Stop Webcam", command=self.stop_cam,
                  bg='lightcoral', font=("Arial", 10), width=15).pack(side=tk.LEFT, padx=5)

        # Bottom info frame
        self.info_frame = tk.Frame(self.root, bg='lightgray')
        self.info_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        self.info_label = tk.Label(self.info_frame,
                                   text="✅ Ready to start! Click 'Start Webcam' to test pose detection.",
                                   font=("Arial", 12), bg='lightgray', fg='darkgreen')
        self.info_label.pack()
        self.status_frame = tk.Frame(self.info_frame, bg='lightgray')
        self.status_frame.pack(pady=5)
        self.video_status = tk.Label(self.status_frame, text="📹 Video: Stopped",
                                     font=("Arial", 10), bg='lightgray')
        self.video_status.pack(side=tk.LEFT, padx=10)
        self.cam_status = tk.Label(self.status_frame, text="📷 Webcam: Stopped",
                                   font=("Arial", 10), bg='lightgray')
        self.cam_status.pack(side=tk.LEFT, padx=10)
        
        # ===== 修改：多人分数区域 - 改用可滚动的Frame =====
        self.score_frame = tk.Frame(self.root, bg='white', relief=tk.RAISED, bd=2)
        self.score_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        tk.Label(self.score_frame, text="多人评分统计", font=("Arial", 14, "bold"), bg='white').pack(pady=5)
        
        # 添加滚动条
        scrollbar = tk.Scrollbar(self.score_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 创建Canvas作为容器
        self.score_canvas_container = tk.Canvas(
            self.score_frame, 
            yscrollcommand=scrollbar.set,
            bg='white',
            width=300
        )
        self.score_canvas_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.score_canvas_container.yview)
        
        # 在Canvas内创建Frame用于放置分数块
        self.score_container = tk.Frame(self.score_canvas_container, bg='white')
        self.score_canvas_container.create_window(
            (0, 0), 
            window=self.score_container, 
            anchor="nw"
        )
        
        # 绑定滚动事件
        self.score_container.bind(
            "<Configure>",
            lambda e: self.score_canvas_container.configure(
                scrollregion=self.score_canvas_container.bbox("all")
            )
        )
        
        # 总体分数图表
        self.fig, self.ax = plt.subplots(figsize=(4,2), dpi=100)
        self.score_canvas = FigureCanvasTkAgg(self.fig, master=self.score_frame)
        self.score_canvas.get_tk_widget().pack(pady=10, fill=tk.BOTH, expand=True)
        
        # 初始化参与者槽位
        self.initialize_participant_slots()

    def on_participant_count_changed(self, event=None):
        """当参与人数改变时重新初始化槽位"""
        try:
            self.participant_count = int(self.participant_var.get())
            print(f"Participant count changed to: {self.participant_count}")
            self.initialize_participant_slots()
        except ValueError:
            self.participant_count = 1
            
    def initialize_participant_slots(self):
        """初始化固定的参与者槽位"""
        print(f"Initializing {self.participant_count} participant slots")
        
        # 清理现有的显示
        self.clear_all_score_displays()
        
        # 为每个参与者创建固定槽位
        for i in range(self.participant_count):
            person_id = i  # 使用0, 1, 2作为固定ID
            
            # 创建独立的检测器
            self.cam_pose_managers[person_id] = PoseDetectionManager(self.detector_type)
            
            # 初始化累计分数
            self.cumulative_scores[person_id] = CumulativeScore()
            
            # 创建分数变量
            self.id_to_score[person_id] = {
                'similarity': tk.StringVar(value="相似度: 0.00"),
                'pose_score': tk.StringVar(value="姿态分: 0.00"),
                'rhythm_score': tk.StringVar(value="节奏分: 0.00"),
                'total_score': tk.StringVar(value="总分: 0.00"),
                'avg_score': tk.StringVar(value="平均分: 0.00")
            }
            
            # 创建分数显示块
            self.create_score_block(person_id)
            
            # 初始化槽位状态
            self.fixed_person_slots[person_id] = {
                'assigned': False,
                'last_position': None,
                'frame_count': 0
            }
    
    def clear_all_score_displays(self):
        """清理所有分数显示"""
        for person_id in list(self.id_to_labels.keys()):
            if person_id in self.id_to_labels:
                try:
                    self.id_to_labels[person_id]['block'].destroy()
                except:
                    pass
        
        # 重置所有数据结构
        self.id_to_labels.clear()
        self.id_to_score.clear()
        self.cumulative_scores.clear()
        self.cam_pose_managers.clear()
        self.fixed_person_slots.clear()

    def create_score_block(self, pid):
        """为每个ID创建独立的分数显示块"""
        if not self.score_container or pid in self.id_to_labels:
            return
            
        # 创建分数块框架
        score_block = tk.Frame(
            self.score_container, 
            bd=2, 
            relief=tk.GROOVE, 
            padx=5, 
            pady=5,
            bg='#f0f0f0'
        )
        score_block.pack(fill=tk.X, padx=5, pady=5, ipadx=5, ipady=5)
        
        # 添加ID标签
        id_label = tk.Label(
            score_block, 
            text=f"Person ID: {pid}", 
            font=("Arial", 12, "bold"),
            bg='#f0f0f0'
        )
        id_label.pack(anchor=tk.W)
        
        # 添加分数标签
        labels = []
        for key, var in self.id_to_score[pid].items():
            label = tk.Label(
                score_block, 
                textvariable=var,
                font=("Arial", 10),
                bg='#f0f0f0'
            )
            label.pack(anchor=tk.W)
            labels.append(label)
        
        # 存储引用
        self.id_to_labels[pid] = {
            'block': score_block,
            'labels': labels
        }

    def update_score_display(self, pid, scores):
        """更新指定ID的分数显示"""
        try:
            if pid in self.id_to_score:
                # 确保分数值有效
                similarity = scores.get('similarity', 0.0)
                pose_score = scores.get('pose_score', 0.0)
                rhythm_score = scores.get('rhythm_score', 0.0)
                total_score = scores.get('total_score', 0.0)
                avg_score = scores.get('avg_score', 0.0)
                
                # 安全地更新StringVar
                self.id_to_score[pid]['similarity'].set(f"相似度: {similarity:.2f}")
                self.id_to_score[pid]['pose_score'].set(f"姿态分: {pose_score:.2f}")
                self.id_to_score[pid]['rhythm_score'].set(f"节奏分: {rhythm_score:.2f}")
                self.id_to_score[pid]['total_score'].set(f"总分: {total_score:.2f}")
                self.id_to_score[pid]['avg_score'].set(f"平均分: {avg_score:.2f}")
                
                print(f"Updated display for person {pid}: similarity={similarity:.2f}, total={total_score:.2f}")
            else:
                print(f"Warning: No score display found for person {pid}")
        except Exception as e:
            print(f"Error updating score display for person {pid}: {e}")

    def calculate_person_score(self, person, pid):
        """计算单个人的分数"""
        try:
            kps = person.keypoints
            # 补齐关键点
            if len(kps) < 33:
                class DummyKP: 
                    x, y, z = 0.0, 0.0, 0.0
                kps = list(kps) + [DummyKP() for _ in range(33 - len(kps))]
            # 创建landmarks对象
            lm = SimpleNamespace()
            lm.landmark = [SimpleNamespace() for _ in range(33)]
            for i in range(33):
                kp = kps[i]
                lm.landmark[i].x = getattr(kp, 'x', 0.0)
                lm.landmark[i].y = getattr(kp, 'y', 0.0)
                lm.landmark[i].z = getattr(kp, 'z', 0.0)
            # 计算相似度
            similarity = calc_similarity(self.last_file_landmarks, lm)
            similarity = similarity if similarity is not None else 0.0
            pose_score = similarity
            # 计算节奏分
            rhythm_score = 0.0
            delta_t = 1.0
            if self.beat_times:
                frame_idx = len(self.cumulative_scores[pid].scores) if pid in self.cumulative_scores else 0
                current_time = frame_idx / 30.0
                try:
                    delta_t = min([abs(current_time - t) for t in self.beat_times])
                except:
                    delta_t = 1.0
                rhythm_score = max(0, 1 - delta_t / 0.4)
            # 计算总分
            # 修正：score_pose参数为(pose_score, delta_t)
            total_score = score_pose(pose_score, delta_t if self.beat_times else 1.0)
            # 更新累计分数
            if pid not in self.cumulative_scores:
                self.cumulative_scores[pid] = CumulativeScore()
            if total_score > 0:
                self.cumulative_scores[pid].update(total_score)
            avg_score = self.cumulative_scores[pid].average
            return {
                'similarity': similarity,
                'pose_score': pose_score,
                'rhythm_score': rhythm_score,
                'total_score': total_score,
                'avg_score': avg_score
            }
        except Exception as e:
            print(f"Error in calculate_person_score for person {pid}: {e}")
            return {
                'similarity': 0.0,
                'pose_score': 0.0,
                'rhythm_score': 0.0,
                'total_score': 0.0,
                'avg_score': 0.0
            }

    def create_person_score_widget(self, person_id):
        """为新检测到的人创建分数显示控件"""
        if person_id in self.id_to_score:
            return
            
        print(f"Creating score widget for person {person_id}")  # 调试信息
        
        # 初始化分数变量 - 确保在主线程中创建
        self.id_to_score[person_id] = {
            'similarity': tk.StringVar(value="相似度: 0.00"),
            'pose_score': tk.StringVar(value="姿态分: 0.00"),
            'rhythm_score': tk.StringVar(value="节奏分: 0.00"),
            'total_score': tk.StringVar(value="总分: 0.00"),
            'avg_score': tk.StringVar(value="平均分: 0.00")
        }
        
        # 创建分数显示块
        self.create_score_block(person_id)
    
    def update_person_score_display(self, person_id, scores):
        """更新指定人员的分数显示"""
        # 修正：去除score_widgets，统一用id_to_labels和id_to_score
        if person_id not in self.id_to_labels:
            self.create_person_score_widget(person_id)
        self.update_score_display(person_id, scores)
    
    def cleanup_inactive_persons(self, active_person_ids):
        """清理不再活跃的人员UI"""
        # 只清理不在0~N-1的ID
        inactive_ids = set(self.id_to_labels.keys()) - set(active_person_ids)
        for person_id in inactive_ids:
            if person_id in self.id_to_labels:
                try:
                    self.id_to_labels[person_id]['block'].destroy()
                    del self.id_to_labels[person_id]
                except Exception as e:
                    print(f"Error destroying UI for person {person_id}: {e}")
            if person_id in self.id_to_score:
                del self.id_to_score[person_id]
            if person_id in self.cumulative_scores:
                del self.cumulative_scores[person_id]
            if person_id in self.cam_pose_managers:
                del self.cam_pose_managers[person_id]

    def load_video(self):
        path = filedialog.askopenfilename(
            title="Select Dance Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if path:
            self.video_path = path
            filename = os.path.basename(path)
            self.info_label.config(text=f"✅ Video loaded: {filename}")
            # 在Canvas上显示文本
            self.canvas_file.delete("all")
            self.canvas_file.create_text(350, 262, text=f"Video loaded:\n{filename}",
                                         fill="white", font=("Arial", 12))
            # 新增：自动提取节拍锚点
            try:
                audio_path = mp4_2_mp3(path)
                tempo, beats, beat_times = get_beats(audio_path)
                self.beat_times = beat_times.tolist() if hasattr(beat_times, 'tolist') else list(beat_times)
                self.info_label.config(text=f"🎵 节拍锚点提取完成，共{len(self.beat_times)}个")
            except Exception as e:
                self.beat_times = []
                self.info_label.config(text=f"⚠️ 节拍提取失败: {e}")

    def start_video(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "Please select a video file first!")
            return
        if not self.running_file:
            self.running_file = True
            self.video_status.config(text="📹 Video: Playing", fg='green')
            self.info_label.config(text="🎬 Video playing! You can now compare with webcam.")
            threading.Thread(target=self.process_video_file, daemon=True).start()

    def stop_video(self):
        self.running_file = False
        self.video_status.config(text="📹 Video: Stopped", fg='red')
        self.info_label.config(text="⏹️ Video stopped.")
        # 在Canvas上显示停止文本
        self.canvas_file.delete("all")
        self.canvas_file.create_text(350, 262, text="Video stopped", fill="white", font=("Arial", 12))
        if self.cap_file:
            self.cap_file.release()

    def toggle_video_display(self):
        self.show_video_frame = not self.show_video_frame
        mode = "Original video" if self.show_video_frame else "Pose skeleton only"
        self.info_label.config(text=f"🔄 Display mode: {mode}")

    def start_cam(self):
        if not self.running_cam:
            self.running_cam = True
            self.cam_status.config(text="📷 Webcam: Running", fg='green')
            if self.last_file_landmarks is None:
                self.info_label.config(text="📷 Webcam started! Load and start video for comparison.")
            else:
                self.info_label.config(text="📷 Webcam started! Move around to test pose detection.")
            threading.Thread(target=self.process_webcam, daemon=True).start()

    def stop_cam(self):
        self.running_cam = False
        self.cam_status.config(text="📷 Webcam: Stopped", fg='red')
        self.info_label.config(text="📷 Webcam stopped.")
        # 在Canvas上显示停止文本
        self.canvas_cam.delete("all")
        self.canvas_cam.create_text(350, 262, text="Webcam stopped", fill="white", font=("Arial", 12))
        if self.cap_cam:
            self.cap_cam.release()

    def process_video_file(self):
        try:
            self.cap_file = cv2.VideoCapture(self.video_path)
            if not self.cap_file.isOpened():
                messagebox.showerror("Error", "Cannot open video file!")
                return

            fps = self.cap_file.get(cv2.CAP_PROP_FPS) or 30
            delay = max(1, int(1000 / fps))

            while self.cap_file.isOpened() and self.running_file:
                ret, frame = self.cap_file.read()
                if not ret:
                    # Video ended, restart from beginning
                    self.cap_file.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                processed_frame = self.process_pose(frame, which='file')
                self.update_canvas(self.canvas_file, processed_frame)

                cv2.waitKey(delay)

        except Exception as e:
            messagebox.showerror("Video Error", f"Error processing video: {str(e)}")
        finally:
            if self.cap_file:
                self.cap_file.release()

    def process_webcam(self):
        try:
            self.cap_cam = cv2.VideoCapture(0)
            if not self.cap_cam.isOpened():
                messagebox.showerror("Error", "Cannot open webcam!")
                return

            while self.cap_cam.isOpened() and self.running_cam:
                ret, frame = self.cap_cam.read()
                if not ret:
                    continue

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                processed_frame = self.process_pose(frame, which='cam')
                self.update_canvas(self.canvas_cam, processed_frame)

        except Exception as e:
            messagebox.showerror("Webcam Error", f"Error processing webcam: {str(e)}")
        finally:
            if self.cap_cam:
                self.cap_cam.release()
    
    def process_pose(self, frame, which='file'):
        """集成自定义检测器/滤波/补全/追踪，支持多人、手势、优化前后对比"""
        import copy
        height, width = frame.shape[:2]
        
        # 选择对应对象
        if which == 'file':
            pose_manager = self.file_pose_manager
            tracker = self.file_tracker
        else:
            # 使用主检测器进行初始检测
            pose_manager = self.cam_pose_manager
            tracker = self.cam_tracker
        
        # 检测
        persons, det_info = pose_manager.detect_poses(frame)
        
        # ===== 新增：为每个检测到的人创建独立的MediaPipe对象 =====
        if which == 'cam' and persons:
            for p in persons:
                pid = getattr(p, 'id', 0)
                if pid not in self.cam_pose_managers:
                    print(f"Creating independent detector for person {pid}")  # 调试信息
                    # 为新人创建独立的检测器
                    self.cam_pose_managers[pid] = PoseDetectionManager(self.detector_type)
                    
                    # 使用root.after确保在主线程中创建UI元素
                    self.root.after(0, self.create_person_score_widget, pid)
        
        # 使用独立的检测器进行二次检测
        if which == 'cam' and persons:
            for p in persons:
                pid = getattr(p, 'id', 0)
                if pid in self.cam_pose_managers:
                    # 使用该ID专用的检测器进行精确检测
                    try:
                        p_persons, _ = self.cam_pose_managers[pid].detect_poses(frame)
                        if p_persons:
                            # 更新关键点
                            p.keypoints = p_persons[0].keypoints
                    except Exception as e:
                        print(f"Individual detector error for person {pid} (ignored): {e}")
        
        # ===== 关键修改：左侧只保留主舞者，右侧保留所有人 =====
        if which == 'file' and persons:
            # 左侧：只保留距离中心最近的主舞者
            def get_center_distance(person):
                if not person.keypoints:
                    return float('inf')
                xs = [kp.x for kp in person.keypoints if kp.visible]
                ys = [kp.y for kp in person.keypoints if kp.visible]
                if not xs or not ys:
                    return float('inf')
                px, py = sum(xs)/len(xs), sum(ys)/len(ys)
                cx, cy = width/2, height/2
                return (px-cx)**2 + (py-cy)**2
            
            main_person = min(persons, key=get_center_distance)
            persons = [main_person]  # 只保留主舞者
            main_person.id = 0  # 给主舞者固定ID
        
        # 记录原始关键点
        orig_persons = copy.deepcopy(persons)
        
        # 安全的遮挡补全 - 修复numpy错误
        if self.fixer_method and persons:
            try:
                # 确保数据格式正确
                conf_seq = []
                keypoints_seq = []
                for p in persons:
                    person_conf = []
                    person_kps = []
                    for kp in p.keypoints:
                        person_conf.append(float(getattr(kp, 'confidence', 0.5)))
                        person_kps.append(np.array([float(kp.x), float(kp.y)]))
                    conf_seq.append(np.array(person_conf))
                    keypoints_seq.append(person_kps)
                if keypoints_seq and all(len(kps) >= 1 for kps in keypoints_seq):
                    # 修正：直接传keypoints_seq和conf_seq，不加[]
                    fixed_kps = fix_keypoints(keypoints_seq, conf_seq, method=self.fixer_method)
                    for i, p in enumerate(persons):
                        if i < len(fixed_kps):
                            num_kps = min(len(p.keypoints), len(fixed_kps[i]))
                            for j in range(num_kps):
                                if j < len(fixed_kps[i]):
                                    kp = p.keypoints[j]
                                    fixed_point = fixed_kps[i][j]
                                    kp.x, kp.y = float(fixed_point[0]), float(fixed_point[1])
            except Exception as e:
                print(f"Fixer error (ignored): {e}")

        # 安全的平滑滤波 - 修复numpy错误
        if self.smoother_method and persons:
            try:
                keypoints_seq = []
                for p in persons:
                    person_kps = []
                    for kp in p.keypoints:
                        person_kps.append(np.array([float(kp.x), float(kp.y)]))
                    keypoints_seq.append(person_kps)
                if keypoints_seq and all(len(kps) >= 1 for kps in keypoints_seq):
                    # 修正：直接传keypoints_seq，不加[]
                    smoothed_kps = smooth_keypoints(keypoints_seq, method=self.smoother_method)
                    for i, p in enumerate(persons):
                        if i < len(smoothed_kps):
                            num_kps = min(len(p.keypoints), len(smoothed_kps[i]))
                            for j in range(num_kps):
                                if j < len(smoothed_kps[i]):
                                    kp = p.keypoints[j]
                                    smoothed_point = smoothed_kps[i][j]
                                    kp.x, kp.y = float(smoothed_point[0]), float(smoothed_point[1])
            except Exception as e:
                print(f"Smoother error (ignored): {e}")

        # 多人追踪 - 只对右侧webcam启用
        if self.tracker_enable and persons and tracker is not None and which == 'cam':
            try:
                dets = []
                for p in persons:
                    kp_array = np.array([[float(kp.x), float(kp.y)] for kp in p.keypoints])
                    bbox_array = np.array(p.bbox, dtype=float)
                    dets.append({'keypoints': kp_array, 'bbox': bbox_array})
                
                tracked = tracker.update(dets, 0)
                for i, p in enumerate(persons):
                    if i < len(tracked):
                        p.id = tracked[i]['id']
                    else:
                        # 如果追踪器没有返回足够的ID，使用简单分配
                        p.id = getattr(p, 'id', i)
            except Exception as e:
                print(f"Tracker error: {e}")
                # 回退：简单分配ID，但尽量保持稳定
                for i, p in enumerate(persons):
                    if not hasattr(p, 'id'):
                        p.id = i
        else:
            # 如果没有追踪器，给每个人分配简单但稳定的ID
            for i, p in enumerate(persons):
                if not hasattr(p, 'id'):
                    # 尝试基于位置分配稳定ID
                    if which == 'cam':
                        # 基于人的中心位置分配ID（从左到右）
                        if p.keypoints:
                            xs = [kp.x for kp in p.keypoints if kp.visible]
                            if xs:
                                center_x = sum(xs) / len(xs)
                                p.id = int(center_x // 100)  # 粗略的位置ID
                            else:
                                p.id = i
                        else:
                            p.id = i
                    else:
                        p.id = i
                    
        # 可视化
        if self.show_video_frame:
            output_frame = frame.copy()
        else:
            output_frame = np.ones_like(frame) * 255
        output_frame = draw_persons(output_frame, persons, self.detector_type)
        fps = 1000.0 / max(det_info.get('processing_time_ms', 1), 1)
        cv2.putText(output_frame, f"Detector: {self.detector_type.value} | Smoother: {self.smoother_method or 'None'} | Fixer: {self.fixer_method or 'None'} | Tracker: {self.tracker_enable}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)
        cv2.putText(output_frame, f"FPS: {fps:.1f} | Persons: {len(persons)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)
        
        # 左侧视频：记录主舞者landmarks
        if which == 'file' and persons:
            main_person_file = persons[0]  # 已经是主舞者了
            # 记录参考主舞者landmarks
            kps = main_person_file.keypoints
            if len(kps) < 33:
                class DummyKP:
                    x, y, z = 0.0, 0.0, 0.0
                kps = list(kps) + [DummyKP() for _ in range(33 - len(kps))]
            lm = SimpleNamespace()
            lm.landmark = [SimpleNamespace() for _ in range(33)]
            for i in range(33):
                kp = kps[i]
                lm.landmark[i].x = getattr(kp, 'x', 0.0)
                lm.landmark[i].y = getattr(kp, 'y', 0.0)
                lm.landmark[i].z = getattr(kp, 'z', 0.0)
            self.last_file_landmarks = lm
            
            # 在视频上标注"主舞者"
            cv2.putText(output_frame, "Main Dancer", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        elif which == 'file':
            self.last_file_landmarks = None
            
        # ===== 新增：右侧webcam多人分数计算和UI更新 =====
        # ===== 关键修改：只处理0~N-1的ID =====
        if which == 'cam' and self.last_file_landmarks and persons:
            active_person_ids = list(range(self.participant_count))
            detected_ids = [getattr(p, 'id', -1) for p in persons]
            for pid in active_person_ids:
                # 找到对应ID的人
                p = next((pp for pp in persons if getattr(pp, 'id', -1) == pid), None)
                if p is not None:
                    # 计算分数
                    scores = self.calculate_person_score(p, pid)
                else:
                    # 未检测到，分数清零
                    scores = {'similarity': 0.0, 'pose_score': 0.0, 'rhythm_score': 0.0, 'total_score': 0.0, 'avg_score': 0.0}
                self.update_score_display(pid, scores)
            # 清理不活跃ID
            self.cleanup_inactive_persons(active_person_ids)
        elif which == 'cam' and not self.last_file_landmarks:
            print("Warning: No reference landmarks from video file")
        elif which == 'cam' and not persons:
            print("Warning: No persons detected in webcam frame")
            
        # 更新总体分数图表
        if which == 'cam':
            self.update_score_plot()

        return output_frame

    def update_canvas(self, canvas, frame):
        try:
            canvas.update_idletasks()
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            if w <= 1 or h <= 1:
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = img.resize((w, h))
            imgtk = ImageTk.PhotoImage(image=img)
            canvas.delete("all")
            canvas.create_image(w//2, h//2, image=imgtk)
            canvas.imgtk = imgtk
        except Exception as e:
            print(f"Error updating canvas: {e}")

    def update_score_plot(self):
        """更新总体分数图表，显示所有人的分数曲线"""
        self.ax.clear()
        
        # 为每个人绘制分数曲线 - 使用累计分数数据
        colors = ['orange', 'blue', 'green', 'red', 'purple', 'brown']
        plot_count = 0
        for person_id, cumulative_score in self.cumulative_scores.items():
            if cumulative_score.scores:
                color = colors[plot_count % len(colors)]
                self.ax.plot(cumulative_score.scores, color=color, label=f'Person {person_id}', linewidth=2)
                plot_count += 1
        
        if plot_count > 0:
            self.ax.set_ylim(0, 1.05)
            self.ax.set_ylabel('分数')
            self.ax.set_xlabel('帧数')
            self.ax.legend(loc='upper right')
            self.ax.grid(True, linestyle='--', alpha=0.3)
            self.ax.set_title('多人分数变化')
        else:
            self.ax.text(0.5, 0.5, '暂无分数数据', transform=self.ax.transAxes, 
                        ha='center', va='center', fontsize=12)
            self.ax.set_title('多人分数变化')
        
        self.fig.tight_layout()
        # 修正：只有score_canvas不为None时才调用draw
        if self.score_canvas is not None:
            self.score_canvas.draw()

    def cleanup(self):
        """Cleanup resources"""
        self.running_file = False
        self.running_cam = False
        if self.cap_file:
            self.cap_file.release()
        if self.cap_cam:
            self.cap_cam.release()


# 用于姿态相似度计算的工具类
class SimilarityCalculator:
    def __init__(self):
        pass

    def calculate(self, lm1, lm2):
        return calculate_pose_similarity(lm1, lm2)

    def center_landmarks(self, landmarks):
        return center_landmarks(landmarks)

    def normalize_landmarks(self, landmarks_np):
        return normalize_landmarks(landmarks_np)


def main():
    # 可通过命令行或配置文件切换参数，这里用默认参数
    root = tk.Tk()
    app = MediaPipePoseApp(root,
                          detector_type=DetectorType.MEDIAPIPE,  # 可选: MEDIAPIPE/YOLOV8/HYBRID
                          smoother_method='kalman',                   # 可选: 'ema'/'kalman'/None
                          fixer_method='symmetric',                      # 可选: 'linear'/'symmetric'/None
                          tracker_enable=True)                   # True/False
    def on_closing():
        app.cleanup()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    print("🚀 Starting MediaPipe Dance GUI... (with multi-person scoring)")
    root.mainloop()


if __name__ == "__main__":
    main()