from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import base64
import numpy as np
import time
import cv2
import tempfile
import os
from typing import Dict, List, Optional
import sys
import traceback
import asyncio

# 添加项目根路径到sys.path
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'score'))  # 添加score文件夹

# 从 api 文件夹导入
from api.model import Keypoint, Landmarks
from api.utils import decode_base64_image, encode_image_to_base64, pad_landmarks, draw_landmarks

# 从 score 文件夹导入障碍物管理 - 使用try-except避免导入错误
try:
    from extra_obstacles import ObstacleManager
    print("✅ 障碍物管理器导入成功")
    obstacle_available = True
except ImportError as e:
    print(f"⚠️ 障碍物管理器导入失败: {e}")
    obstacle_available = False

# 导入检测器模块
try:
    from detector.pose_detector import DetectorType, PoseDetectionManager
    print("✅ 姿态检测模块导入成功")
    detector_available = True
except ImportError as e:
    print(f"❌ 姿态检测模块导入失败: {e}")
    traceback.print_exc()
    detector_available = False

# 导入评分模块
try:
    from score.similarity import calculate_pose_similarity
    from score.score_pose import score_pose
    from score.music_beat import mp4_2_mp3, get_beats
    from score.motion_match import match_motion_to_beats
    from score.average_similarity import CumulativeScore
    print("✅ 评分模块导入成功")
    score_available = True
except ImportError as e:
    print(f"❌ 评分模块导入失败: {e}")
    traceback.print_exc()
    score_available = False

# 提供备用实现
if not detector_available:
    print("⚠️ 使用备用检测器实现")
    class DetectorType:
        MEDIAPIPE = "mediapipe"
    
    class PoseDetectionManager:
        def __init__(self, detector_type):
            self.detector_type = detector_type
            print(f"⚠️ 使用备用检测器: {detector_type}")
        
        def detect_poses(self, frame):
            return [], {"processing_time_ms": 10}

if not score_available:
    print("⚠️ 使用备用评分实现")
    def calculate_pose_similarity(lm1, lm2):
        return 0.8
    
    def score_pose(pose_score, delta_t):
        return pose_score * 0.9
    
    def mp4_2_mp3(video_path):
        return ""
    
    def get_beats(audio_path):
        return 120, [], []
    
    def match_motion_to_beats(motion, beats):
        return []
    
    class CumulativeScore:
        def __init__(self):
            self.scores = []
            self.average = 0.0
        
        def update(self, score):
            self.scores.append(score)
            self.average = sum(self.scores) / len(self.scores)
        
        def reset(self):
            self.scores = []
            self.average = 0.0

if not obstacle_available:
    print("⚠️ 使用备用障碍物管理器")
    class ObstacleManager:
        def __init__(self, frame_size=(640, 480)):
            self.frame_width, self.frame_height = frame_size
            self.active_obstacles = []
            self.sync_time = 0.0
            self.sync_enabled = False
        
        def set_sync_time(self, sync_time):
            self.sync_time = sync_time
            self.sync_enabled = True
        
        def set_difficulty(self, level):
            pass
        
        def spawn_obstacle(self):
            return None
        
        def update_obstacles(self):
            return []
        
        def check_collision(self, obstacle, landmarks):
            return None
        
        def deactivate_obstacle(self, obstacle_id):
            pass
        
        def reset(self):
            self.active_obstacles = []

ws_router = APIRouter()


# 连接管理和会话类
class PersonScoreTracker:
    """每个人的独立评分追踪器"""

    def __init__(self, person_id):
        self.person_id = person_id
        self.cumulative_score = CumulativeScore()
        self.score_history = []
        self.obstacle_score = 0  # 障碍得分
        self.current_scores = {
            'similarity': 0.0,
            'pose_score': 0.0,
            'rhythm_score': 0.0,
            'total_score': 0.0,
            'obstacle_score': 0.0,
            'avg_score': 0.0
        }

    def update_scores(self, similarity, pose_score, rhythm_score, total_score, obstacle_score=0):
        self.current_scores['similarity'] = similarity
        self.current_scores['pose_score'] = pose_score
        self.current_scores['rhythm_score'] = rhythm_score
        self.current_scores['total_score'] = total_score
        self.current_scores['obstacle_score'] = obstacle_score
        self.obstacle_score = obstacle_score

        if total_score > 0:
            self.cumulative_score.update(total_score)
            self.current_scores['avg_score'] = self.cumulative_score.average
            self.score_history.append(total_score)


class GameSession:
    def __init__(self):
        # 参考视频相关
        self.reference_landmarks: Optional[List[Keypoint]] = None
        self.reference_video_path: Optional[str] = None
        self.beat_times: List[float] = []
        
        # 游戏状态
        self.game_started: bool = False
        self.game_paused: bool = False
        self.selected_dance: Dict = {'id': 1, 'name': 'Easy'}
        self.level: str = 'Easy'
        self.start_time: Optional[float] = None
        self.frame_count: int = 0
        
        # 🔄 同步状态管理
        self.sync_enabled: bool = False
        self.sync_start_time: Optional[float] = None
        self.webcam_first_frame: bool = False
        self.reference_synced: bool = False
        
        # 创建两个完全独立的检测器实例
        try:
            self.pose_manager_reference = PoseDetectionManager(DetectorType.MEDIAPIPE)
            self.pose_manager_webcam = PoseDetectionManager(DetectorType.MEDIAPIPE)
            
            print(f"✅ 参考视频检测器: {id(self.pose_manager_reference)}")
            print(f"✅ 用户视频检测器: {id(self.pose_manager_webcam)}")
            print("✅ 两个独立检测器初始化成功")
        except Exception as e:
            print(f"❌ 检测器初始化失败: {e}")
            self.pose_manager_reference = None
            self.pose_manager_webcam = None
        
        # 多人评分追踪
        self.person_trackers: Dict[int, PersonScoreTracker] = {}
        self.cumulative_score = CumulativeScore()
        self.score_history: List[float] = []

        # 障碍物管理器
        self.obstacle_manager = ObstacleManager(frame_size=(640, 480))
        
        # 🔄 新增：为障碍物管理器设置默认难度
        self.obstacle_manager.set_difficulty('Easy')
        
        # 🔄 新增：同步状态管理（如果你之前没有添加）
        self.sync_enabled = False
        self.sync_start_time = None
        self.time_offset = 0.0 # 用于同步时间校正
        
        
    
    def reset_sync_state(self):
        """重置同步状态"""
        self.sync_enabled = False
        self.sync_start_time = None
        self.webcam_first_frame = False
        self.reference_synced = False
        self.time_offset = 0.0
        print("🔄 同步状态已重置")

    # 🔄 新增：添加同步方法
    def enable_sync(self, sync_start_time: float):
        """启用同步模式"""
        self.sync_enabled = True
        self.sync_start_time = sync_start_time
        self.time_offset = time.time() - sync_start_time
        print(f"🔄 同步模式已启用: {sync_start_time}")
    
    def get_sync_time(self, client_time: float = None) -> float:
        """获取同步时间"""
        if client_time is not None and self.sync_enabled:
            return client_time
        
        if self.sync_enabled and self.sync_start_time:
            current_time = time.time()
            return current_time - self.time_offset
        
        return time.time() - (self.start_time or time.time())
    
    def reset_sync(self):
        """重置同步状态"""
        self.sync_enabled = False
        self.sync_start_time = None
        self.time_offset = 0.0


# 连接管理
active_sessions: Dict[str, GameSession] = {}


@ws_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = id(websocket)
    active_sessions[session_id] = GameSession()
    session = active_sessions[session_id]

    print(f"✅ WebSocket连接成功 (Session: {session_id})")

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                await handle_message(msg, websocket, session)
            except Exception as e:
                print(f"❌ 消息处理失败: {e}")
                traceback.print_exc()
                await websocket.send_json({
                    'event': 'error',
                    'message': f'消息处理失败: {str(e)}'
                })

    except WebSocketDisconnect:
        print(f"❌ WebSocket断开 (Session: {session_id})")
        if session_id in active_sessions:
            del active_sessions[session_id]
    except Exception as e:
        print(f"❌ WebSocket错误: {e}")
        traceback.print_exc()
        if session_id in active_sessions:
            del active_sessions[session_id]


async def handle_message(msg: Dict, websocket: WebSocket, session: GameSession):
    """处理WebSocket消息"""
    event = msg.get('event')
    print(f"📨 收到WebSocket消息: {event}")

    if event == 'frame':
        await handle_frame(msg, websocket, session)
    elif event == 'upload_reference_video':
        await handle_upload_reference_video(msg, websocket, session)
    elif event == 'start_game':
        await handle_start_game(msg, websocket, session)
    elif event == 'sync_start':
        await handle_sync_start(msg, websocket, session)
    elif event == 'pause_game':
        await handle_pause_game(websocket, session)
    elif event == 'resume_game':
        await handle_resume_game(websocket, session)
    elif event == 'stop_game':
        await handle_stop_game(websocket, session)
    else:
        print(f"❓ 未知事件: {event}")


async def handle_sync_start(msg: Dict, websocket: WebSocket, session: GameSession):
    """处理同步开始信号"""
    timestamp = msg.get('timestamp')
    
    if timestamp and session.sync_enabled:
        session.sync_start_time = timestamp / 1000.0  # 转换为秒
        session.webcam_first_frame = True
        session.time_offset = time.time() - session.sync_start_time
        
        print(f"🔄 同步开始 - 时间基准: {session.sync_start_time}, 偏移: {session.time_offset:.3f}s")
        
        # 通知前端同步已建立
        await websocket.send_json({
            'event': 'sync_established',
            'sync_time': session.sync_start_time,
            'time_offset': session.time_offset
        })


async def handle_frame(msg: Dict, websocket: WebSocket, session: GameSession):
    """处理视频帧 - 支持同步"""
    frame_type = msg.get('frame_type', 'webcam')
    image_data = msg.get('image', '')
    current_time = msg.get('current_time', 0.0)
    sync_enabled = msg.get('sync_enabled', False)
    sync_timestamp = msg.get('sync_timestamp')


    # 🔄 在帧处理开始时添加同步时间设置
    sync_time = session.get_sync_time(msg.get('current_time'))
    
    # 🔄 为障碍物管理器设置同步时间
    if hasattr(session.obstacle_manager, 'set_sync_time'):
        session.obstacle_manager.set_sync_time(sync_time)

    # 🔄 同步时间处理
    if sync_enabled and sync_timestamp:
        # 使用前端传来的同步时间
        sync_time = current_time
    else:
        # 使用服务器时间
        sync_time = session.get_sync_time(current_time)

    print(f"🎬 处理{frame_type}帧 - 同步时间: {sync_time:.3f}s")

    if not image_data:
        print("❌ 图片数据为空")
        return

    # 根据帧类型选择不同的检测器实例
    if frame_type == 'reference':
        pose_manager = session.pose_manager_reference
    else:
        pose_manager = session.pose_manager_webcam

    if not pose_manager:
        print("❌ 检测器未初始化")
        return

    try:
        # 解码图片
        frame = decode_base64_image(image_data)
        if frame is None:
            print("❌ 图片解码失败")
            return

        height, width = frame.shape[:2]
        print(f"✅ 图片解码成功，尺寸: {frame.shape}")

        # 摄像头webcam帧做水平镜像
        if frame_type == 'webcam':
            frame = cv2.flip(frame, 1)

        # 姿态检测
        start_time = time.time()
        persons, det_info = pose_manager.detect_poses(frame)
        processing_time = (start_time - time.time()) * 1000

        print(f"🔍 姿态检测完成，检测到 {len(persons) if persons else 0} 人")

        # 参考视频：只保留距离中心最近的主舞者
        if frame_type == 'reference' and persons:
            def get_center_distance(person):
                if not person.keypoints:
                    return float('inf')
                xs = [kp.x for kp in person.keypoints if kp.visible]
                ys = [kp.y for kp in person.keypoints if kp.visible]
                if not xs or not ys:
                    return float('inf')
                px, py = sum(xs) / len(xs), sum(ys) / len(ys)
                cx, cy = width / 2, height / 2
                return (px - cx) ** 2 + (py - cy) ** 2

            main_person = min(persons, key=get_center_distance)
            persons = [main_person]
            main_person.id = 0

        # 提取关键点
        all_landmarks = []
        if persons:
            for person in persons:
                kps = []
                for kp in person.keypoints:
                    kps.append(Keypoint(
                        x=float(kp.x),
                        y=float(kp.y),
                        z=getattr(kp, 'z', 0.0),
                        confidence=getattr(kp, 'confidence', 1.0),
                        visible=getattr(kp, 'visible', True)
                    ))
                landmarks = pad_landmarks(kps, 33)
                all_landmarks.append(landmarks)

        # 绘制姿态
        vis_frame = frame.copy()
        if all_landmarks:
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
            for i, landmarks in enumerate(all_landmarks):
                color = colors[i % len(colors)]
                vis_frame = draw_landmarks(vis_frame, landmarks, color=color)

        # 如果是参考视频，添加标注
        if frame_type == 'reference' and all_landmarks:
            cv2.putText(vis_frame, "Main Dancer", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        # 编码图片
        vis_img_b64 = encode_image_to_base64(vis_frame)

        # 发送帧结果
        await websocket.send_json({
            'event': 'frame_result',
            'type': frame_type,
            'image': vis_img_b64,
            'persons_detected': len(persons) if persons else 0,
            'processing_time_ms': abs(processing_time),
            'sync_time': sync_time,
            'persons_keypoints_count': [
                sum(1 for kp in landmarks if getattr(kp, 'visible', True))
                for landmarks in all_landmarks
            ] if all_landmarks else []
        })

        # 处理参考帧
        if frame_type == 'reference' and all_landmarks:
            session.reference_landmarks = all_landmarks[0]
            print(f"📹 参考视频主舞者关键点已保存")

        # 处理用户帧并计算多人分数 + 障碍物（使用同步时间）
        elif frame_type == 'webcam' and session.reference_landmarks and all_landmarks:
            print(f"🎯 开始计算多人分数+障碍物，共{len(all_landmarks)}人，同步时间: {sync_time:.3f}s")
            await calculate_multi_person_scores_with_obstacles(
                all_landmarks, sync_time, websocket, session, persons
            )

    except Exception as e:
        print(f"❌ 帧处理错误: {e}")
        traceback.print_exc()
        await websocket.send_json({
            'event': 'error',
            'message': f'帧处理失败: {str(e)}'
        })


async def calculate_multi_person_scores_with_obstacles(all_landmarks: List[List[Keypoint]], 
                                                     sync_time: float,
                                                     websocket: WebSocket, 
                                                     session: GameSession, 
                                                     persons):
    """计算多人分数 + 障碍物检测 - 支持同步时间"""
    if not session.game_started or session.game_paused:
        return

    try:
        # 🎯 1. 基于同步时间更新障碍物状态
        if hasattr(session.obstacle_manager, 'set_sync_time'):
            session.obstacle_manager.set_sync_time(sync_time)
        
        obstacles = session.obstacle_manager.update_obstacles()
        
        # 2. 尝试生成新障碍物（基于同步时间）
        new_obstacle = session.obstacle_manager.spawn_obstacle()
        if new_obstacle:
            print(f"🎯 生成新障碍物: {new_obstacle['id']} at {sync_time:.3f}s")
            await websocket.send_json({
                'event': 'obstacle_spawn',
                'obstacle': new_obstacle,
                'sync_time': sync_time
            })

        # 3. 发送障碍物更新
        if obstacles:
            await websocket.send_json({
                'event': 'obstacle_update',
                'obstacles': obstacles,
                'sync_time': sync_time
            })

        # 4. 处理每个人的分数
        person_scores = {}
        total_obstacle_score = 0

        for i, user_landmarks in enumerate(all_landmarks):
            person_id = getattr(persons[i], 'id', i) if i < len(persons) else i

            # 为新人创建tracker
            if person_id not in session.person_trackers:
                session.person_trackers[person_id] = PersonScoreTracker(person_id)

            # 计算基础分数
            try:
                pose_score = calculate_pose_similarity(session.reference_landmarks, user_landmarks) or 0.0

                # 🎵 基于同步时间计算节奏分数
                rhythm_score = 0.0
                if session.beat_times:
                    if session.beat_times:
                        # 使用同步时间计算节拍匹配
                        delta_t = min([abs(sync_time - bt) for bt in session.beat_times])
                        rhythm_score = max(0, 1 - delta_t / 0.4)

                # 手势分数
                hand_score = pose_score * 0.8

                # 5. 障碍物碰撞检测
                obstacle_score_change = 0
                for obstacle in obstacles:
                    if obstacle.get('active', True):
                        collision_result = session.obstacle_manager.check_collision(obstacle, user_landmarks)
                        if collision_result:
                            obstacle_score_change += collision_result['score_change']
                            
                            print(f"🎯 Person {person_id} 障碍物碰撞: {collision_result['result']}, 分数变化: {collision_result['score_change']}")
                            
                            # 发送障碍物得分事件
                            await websocket.send_json({
                                'event': 'obstacle_score',
                                'person_id': person_id,
                                'result': collision_result['result'],
                                'score_change': collision_result['score_change'],
                                'display': collision_result['display'],
                                'sync_time': sync_time
                            })
                            
                            # 禁用已碰撞的障碍物
                            session.obstacle_manager.deactivate_obstacle(obstacle['id'])

                # 更新总障碍得分
                tracker = session.person_trackers[person_id]
                tracker.obstacle_score += obstacle_score_change
                total_obstacle_score += tracker.obstacle_score

                # 难度权重
                LEVEL_WEIGHTS = {
                    'Easy': (0.8, 0.15, 0.05),
                    'Medium': (0.6, 0.3, 0.1),
                    'Hard': (0.5, 0.4, 0.1),
                    'Expert': (0.4, 0.5, 0.1)
                }
                w_pose, w_rhythm, w_hand = LEVEL_WEIGHTS.get(session.level, (0.8, 0.15, 0.05))
                total_score = w_pose * pose_score + w_rhythm * rhythm_score + w_hand * hand_score

                # 更新该人的分数
                tracker.update_scores(pose_score, pose_score, rhythm_score, total_score, tracker.obstacle_score)
                person_scores[person_id] = tracker.current_scores

            except Exception as e:
                print(f"❌ Person {person_id} 分数计算错误: {e}")
                continue

        # 清理不活跃的人员
        active_person_ids = set(person_scores.keys())
        inactive_ids = set(session.person_trackers.keys()) - active_person_ids
        for person_id in inactive_ids:
            del session.person_trackers[person_id]

        # 计算整体平均分数
        if person_scores:
            all_total_scores = [scores['total_score'] for scores in person_scores.values()]
            avg_total_score = sum(all_total_scores) / len(all_total_scores)
            session.cumulative_score.update(avg_total_score)

        session.frame_count += 1

        # 发送多人分数更新
        await websocket.send_json({
            'event': 'score_update',
            'person_scores': person_scores,
            'current_scores': {
                'pose_score': round(list(person_scores.values())[0]['pose_score'] * 100, 2) if person_scores else 0,
                'rhythm_score': round(list(person_scores.values())[0]['rhythm_score'] * 100, 2) if person_scores else 0,
                'hand_score': round(list(person_scores.values())[0]['rhythm_score'] * 80, 2) if person_scores else 0,
                'total_score': round(list(person_scores.values())[0]['total_score'] * 100, 2) if person_scores else 0,
                'obstacle_score': total_obstacle_score
            },
            'average_score': round(session.cumulative_score.average * 100, 2),
            'frame_count': session.frame_count,
            'persons_detected': len(person_scores),
            'sync_time': sync_time
        })

        # 🕒 基于同步时间的自动结束游戏逻辑
        if session.sync_enabled and session.sync_start_time:
            game_duration = sync_time
            if game_duration > 60:  # 60秒后自动结束
                print(f"⏰ 游戏时间达到60秒，自动结束 (实际时长: {game_duration:.1f}s)")
                await handle_stop_game(websocket, session)
        elif session.start_time and (time.time() - session.start_time) > 60:
            await handle_stop_game(websocket, session)

    except Exception as e:
        print(f"❌ 多人分数+障碍物计算错误: {e}")
        traceback.print_exc()


async def handle_upload_reference_video(msg: Dict, websocket: WebSocket, session: GameSession):
    """处理参考视频上传"""
    video_data = msg.get('video', '')
    if not video_data:
        print("❌ 视频数据为空")
        return

    try:
        print("📤 开始处理参考视频上传...")

        # 解码视频数据
        if ',' in video_data:
            video_bytes = base64.b64decode(video_data.split(',')[1])
        else:
            video_bytes = base64.b64decode(video_data)

        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(video_bytes)
            session.reference_video_path = tmp_file.name

        print(f"✅ 视频保存成功: {session.reference_video_path}")

        # 提取音频和节拍
        try:
            print("🎵 开始提取音频和节拍...")
            audio_path = mp4_2_mp3(session.reference_video_path)
            tempo, beats, beat_times = get_beats(audio_path)
            session.beat_times = beat_times.tolist() if hasattr(beat_times, 'tolist') else list(beat_times)

            # 清理临时音频文件
            if os.path.exists(audio_path):
                os.remove(audio_path)

            print(f"✅ 节拍提取成功，共{len(session.beat_times)}个节拍点")

            await websocket.send_json({
                'event': 'reference_ready',
                'beat_count': len(session.beat_times),
                'tempo': float(tempo) if hasattr(tempo, 'item') else tempo,
                'beat_times': session.beat_times[:10]  # 发送前10个节拍点用于调试
            })

        except Exception as e:
            print(f"⚠️ 节拍提取失败: {e}")
            session.beat_times = []
            await websocket.send_json({
                'event': 'reference_ready',
                'beat_count': 0,
                'tempo': 120.0,
                'message': f'视频上传成功，但节拍提取失败: {str(e)}'
            })

    except Exception as e:
        print(f"❌ 视频上传处理失败: {e}")
        traceback.print_exc()
        await websocket.send_json({
            'event': 'error',
            'message': f'视频上传失败: {str(e)}'
        })


# 在 handle_start_game 函数中，添加难度设置：
async def handle_start_game(msg: Dict, websocket: WebSocket, session: GameSession):
    """开始游戏"""
    session.selected_dance = msg.get('dance', session.selected_dance)
    session.level = msg.get('level', session.selected_dance.get('name', 'Easy'))
    session.game_started = True
    session.game_paused = False
    session.start_time = time.time()
    session.frame_count = 0
    session.score_history = []
    session.cumulative_score.reset()
    session.person_trackers.clear()
    
    # 🔄 新增：启用同步模式
    session.sync_enabled = True
    
    # 🎯 新增：设置障碍物管理器难度
    session.obstacle_manager.set_difficulty(session.level)
    
    # 重置障碍物管理器
    session.obstacle_manager.reset()

    await websocket.send_json({
        'event': 'game_started',
        'dance': session.selected_dance,
        'level': session.level,
        'sync_enabled': session.sync_enabled
    })
    print(f"🎮 游戏开始: {session.selected_dance['name']} 难度: {session.level}")


async def handle_pause_game(websocket: WebSocket, session: GameSession):
    """暂停游戏"""
    session.game_paused = True
    await websocket.send_json({'event': 'game_paused'})
    print("⏸️ 游戏暂停")


async def handle_resume_game(websocket: WebSocket, session: GameSession):
    """恢复游戏"""
    session.game_paused = False
    await websocket.send_json({'event': 'game_resumed'})
    print("▶️ 游戏恢复")


async def handle_stop_game(websocket: WebSocket, session: GameSession):
    """停止游戏"""
    final_score = session.cumulative_score.average * 100

    session.game_started = False
    session.game_paused = False
    
    # 重置同步状态
    session.reset_sync_state()
    
    # 重置障碍物管理器
    session.obstacle_manager.reset()

    await websocket.send_json({
        'event': 'game_stopped',
        'final_score': round(final_score, 2),
        'total_persons': len(session.person_trackers),
        'sync_disabled': True
    })

    print(f"🛑 游戏结束，最终得分: {final_score:.2f}，共{len(session.person_trackers)}人参与")