import numpy as np
import random
import time
from typing import List, Dict, Tuple, Optional
from api.model import Keypoint


def generate_obstacle(frame_width: int, frame_height: int, 
                     obstacle_size: Tuple[int, int] = (100, 100)) -> Dict:
    """
    生成随机位置的障碍/奖励框
    
    Args:
        frame_width: 画面宽度
        frame_height: 画面高度  
        obstacle_size: 障碍框大小 (width, height)
        
    Returns:
        障碍信息字典
    """
    w, h = obstacle_size
    
    # 确保框不会超出画面边界
    max_x = max(0, frame_width - w)
    max_y = max(0, frame_height - h)
    
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    
    return {
        'start_pos': (x, y),
        'size': (w, h),
        'center': (x + w // 2, y + h // 2)
    }


def is_point_inside_rectangle(point: Tuple[float, float], 
                            rect: Tuple[float, float, float, float]) -> bool:
    """
    判断点是否在矩形内
    
    Args:
        point: 点坐标 (x, y)
        rect: 矩形 (x_min, y_min, x_max, y_max)，值为[0,1]归一化坐标
        
    Returns:
        是否在矩形内
    """
    x, y = point
    x_min, y_min, x_max, y_max = rect
    return x_min <= x <= x_max and y_min <= y <= y_max


def is_face_inside_rectangle(landmarks: List[Keypoint], 
                           rect: Tuple[float, float, float, float]) -> bool:
    """
    判断整个脸部是否完全在矩形内
    
    Args:
        landmarks: 关键点列表
        rect: 矩形边界 (x_min, y_min, x_max, y_max)，值为[0,1]归一化坐标
        
    Returns:
        脸部是否完全在矩形内
    """
    if not landmarks or len(landmarks) < 33:
        return False
    
    try:
        # MediaPipe面部关键点索引
        # 0: 鼻子, 1-4: 眼部周围, 7-8: 嘴角, 9-10: 眉毛
        face_indices = [0, 1, 2, 3, 4, 7, 8, 9, 10]
        
        face_points = []
        for idx in face_indices:
            if idx < len(landmarks):
                kp = landmarks[idx]
                if kp.visible and kp.confidence > 0.5:
                    face_points.append((kp.x, kp.y))
        
        if len(face_points) < 3:  # 至少需要3个可见的面部点
            return False
        
        # 计算面部边界框
        xs = [p[0] for p in face_points]
        ys = [p[1] for p in face_points]
        
        face_x_min, face_x_max = min(xs), max(xs)
        face_y_min, face_y_max = min(ys), max(ys)
        
        # 检查面部边界框是否完全在目标矩形内
        rect_x_min, rect_y_min, rect_x_max, rect_y_max = rect
        
        return (rect_x_min <= face_x_min and face_x_max <= rect_x_max and
                rect_y_min <= face_y_min and face_y_max <= rect_y_max)
        
    except Exception as e:
        print(f"面部检测错误: {e}")
        return False


def is_any_landmark_inside_rectangle(landmarks: List[Keypoint], 
                                   rect: Tuple[float, float, float, float]) -> bool:
    """
    判断是否有任何关键点在矩形内
    
    Args:
        landmarks: 关键点列表
        rect: 矩形边界 (x_min, y_min, x_max, y_max)，值为[0,1]归一化坐标
        
    Returns:
        是否有关键点在矩形内
    """
    if not landmarks:
        return False
    
    try:
        for kp in landmarks:
            if kp.visible and kp.confidence > 0.5:
                if is_point_inside_rectangle((kp.x, kp.y), rect):
                    return True
        return False
        
    except Exception as e:
        print(f"关键点碰撞检测错误: {e}")
        return False


def create_obstacle_pattern(obstacle_type: str, size: Tuple[int, int]) -> Dict:
    """
    创建不同类型的障碍图案配置
    
    Args:
        obstacle_type: 障碍类型 ('reward', 'penalty')
        size: 尺寸 (width, height)
        
    Returns:
        图案配置字典
    """
    w, h = size
    
    if obstacle_type == 'reward':
        return {
            'type': 'reward',
            'color': '#4CAF50',  # 绿色
            'border_color': '#2E7D32',
            'icon': '🎯',
            'glow_color': '#81C784',
            'animation': 'pulse',
            'description': 'Face Target'
        }
    elif obstacle_type == 'penalty':
        return {
            'type': 'penalty', 
            'color': '#F44336',  # 红色
            'border_color': '#C62828',
            'icon': '⚠️',
            'glow_color': '#E57373',
            'animation': 'shake',
            'description': 'Avoid Zone'
        }
    else:
        return {
            'type': 'neutral',
            'color': '#FF9800',  # 橙色
            'border_color': '#F57C00',
            'icon': '❓',
            'glow_color': '#FFB74D',
            'animation': 'rotate',
            'description': 'Unknown'
        }


def calculate_obstacle_score(result_type: str, obstacle_type: str) -> int:
    """
    计算障碍得分
    
    Args:
        result_type: 结果类型 ('hit', 'miss', 'safe', 'fail')
        obstacle_type: 障碍类型 ('reward', 'penalty')
        
    Returns:
        得分变化
    """
    if obstacle_type == 'reward':
        return 2 if result_type == 'hit' else 0
    elif obstacle_type == 'penalty':
        return -2 if result_type == 'fail' else 0
    else:
        return 0


def format_score_display(score_change: int) -> Dict:
    """
    格式化分数显示效果
    
    Args:
        score_change: 分数变化
        
    Returns:
        显示效果配置
    """
    if score_change > 0:
        return {
            'text': f'+{score_change}',
            'color': '#4CAF50',
            'icon': '✨',
            'animation': 'scoreUp',
            'duration': 2000
        }
    elif score_change < 0:
        return {
            'text': f'{score_change}',
            'color': '#F44336', 
            'icon': '💥',
            'animation': 'scoreDown',
            'duration': 2000
        }
    else:
        return {
            'text': 'Miss',
            'color': '#9E9E9E',
            'icon': '😐',
            'animation': 'scoreMiss',
            'duration': 1500
        }


class ObstacleManager:
    """障碍管理器 - 升级版，支持同步"""
    
    def __init__(self, frame_size: Tuple[int, int] = (640, 480)):
        self.frame_width, self.frame_height = frame_size
        self.active_obstacles: List[Dict] = []
        self.obstacle_id_counter = 0
        self.last_spawn_time = 0
        self.spawn_interval = 3.0  # 3秒间隔
        self.obstacle_duration = 4.0  # 障碍持续4秒
        
        # 🔄 新增：同步相关属性
        self.sync_time: float = 0.0
        self.sync_enabled: bool = False
        self.time_offset: float = 0.0
        
        # 🎯 新增：难度相关属性
        self.difficulty_level = 'Easy'
        self.spawn_probability = 0.7
        self.max_active_obstacles = 3
        
        print("✅ 障碍管理器初始化完成")
    
    def set_sync_time(self, sync_time: float):
        """🔄 新增：设置同步时间"""
        self.sync_time = sync_time
        self.sync_enabled = True
        if not hasattr(self, 'sync_start_time'):
            self.sync_start_time = sync_time
    
    def set_difficulty(self, level: str):
        """🎯 新增：设置难度等级"""
        self.difficulty_level = level
        
        difficulty_settings = {
            'Easy': {'interval': 3.0, 'probability': 0.5, 'max_obstacles': 2},
            'Medium': {'interval': 2.5, 'probability': 0.6, 'max_obstacles': 3},
            'Hard': {'interval': 2.0, 'probability': 0.7, 'max_obstacles': 4},
            'Expert': {'interval': 1.5, 'probability': 0.8, 'max_obstacles': 5}
        }
        
        settings = difficulty_settings.get(level, difficulty_settings['Easy'])
        self.spawn_interval = settings['interval']
        self.spawn_probability = settings['probability']
        self.max_active_obstacles = settings['max_obstacles']
        
        print(f"🎯 难度设置: {level} - 生成间隔: {self.spawn_interval}s")
    
    def should_spawn_obstacle(self) -> bool:
        """判断是否应该生成新的障碍 - 支持同步时间"""
        if self.sync_enabled:
            # 使用同步时间
            return (self.sync_time - self.last_spawn_time) >= self.spawn_interval
        else:
            # 使用系统时间（向后兼容）
            current_time = time.time()
            return (current_time - self.last_spawn_time) >= self.spawn_interval
    
    def spawn_obstacle(self) -> Optional[Dict]:
        """生成新的障碍 - 支持同步和难度"""
        if not self.should_spawn_obstacle():
            return None
        
        # 检查活跃障碍数量限制
        if len(self.active_obstacles) >= self.max_active_obstacles:
            return None
            
        # 随机决定是否生成（基于难度）
        if random.random() > self.spawn_probability:
            return None
        
        # 获取当前时间
        current_time = self.sync_time if self.sync_enabled else time.time()
        self.last_spawn_time = current_time
        
        # 随机选择障碍类型
        obstacle_type = random.choice(['reward', 'penalty'])
        
        # 生成位置
        obstacle_info = generate_obstacle(
            self.frame_width, self.frame_height, 
            obstacle_size=(120, 120)
        )
        
        # 创建障碍对象
        obstacle = {
            'id': f'obstacle_{self.obstacle_id_counter}',
            'type': obstacle_type,
            'start_time': current_time,
            'duration': self.obstacle_duration,
            'position': obstacle_info['start_pos'],
            'size': obstacle_info['size'],
            'center': obstacle_info['center'],
            'pattern': create_obstacle_pattern(obstacle_type, obstacle_info['size']),
            'active': True,
            'blinking': False,
            'progress': 0.0
        }
        
        self.obstacle_id_counter += 1
        self.active_obstacles.append(obstacle)
        
        return obstacle
    
    def update_obstacles(self) -> List[Dict]:
        """更新障碍状态，返回活跃的障碍 - 支持同步时间"""
        current_time = self.sync_time if self.sync_enabled else time.time()
        
        # 移除过期的障碍
        self.active_obstacles = [
            obs for obs in self.active_obstacles 
            if (current_time - obs['start_time']) < obs['duration']
        ]
        
        # 更新障碍状态
        for obstacle in self.active_obstacles:
            elapsed_time = current_time - obstacle['start_time']
            obstacle['progress'] = elapsed_time / obstacle['duration']
            
            # 添加闪烁效果（接近结束时）
            if obstacle['progress'] > 0.7:
                obstacle['blinking'] = True
            else:
                obstacle['blinking'] = False
        
        return self.active_obstacles
    
    def get_obstacle_rect(self, obstacle: Dict) -> Tuple[float, float, float, float]:
        """获取障碍的归一化矩形坐标"""
        x, y = obstacle['position']
        w, h = obstacle['size']
        
        # 转换为归一化坐标 [0, 1]
        x_min = x / self.frame_width
        y_min = y / self.frame_height
        x_max = (x + w) / self.frame_width
        y_max = (y + h) / self.frame_height
        
        return (x_min, y_min, x_max, y_max)
    
    def check_collision(self, obstacle: Dict, landmarks: List[Keypoint]) -> Optional[Dict]:
        """检查碰撞并返回结果 - 增强版"""
        if not obstacle['active']:
            return None
        
        rect = self.get_obstacle_rect(obstacle)
        
        if obstacle['type'] == 'reward':
            # 奖励框：需要全脸在框内
            if is_face_inside_rectangle(landmarks, rect):
                score_change = calculate_obstacle_score('hit', 'reward')
                return {
                    'obstacle_id': obstacle['id'],
                    'result': 'hit',
                    'score_change': score_change,
                    'display': format_score_display(score_change)
                }
            else:
                return {
                    'obstacle_id': obstacle['id'],
                    'result': 'miss',
                    'score_change': 0,
                    'display': format_score_display(0)
                }
        
        elif obstacle['type'] == 'penalty':
            # 惩罚框：任何关键点碰到就扣分
            if is_any_landmark_inside_rectangle(landmarks, rect):
                score_change = calculate_obstacle_score('fail', 'penalty')
                return {
                    'obstacle_id': obstacle['id'],
                    'result': 'fail',
                    'score_change': score_change,
                    'display': format_score_display(score_change)
                }
            else:
                return {
                    'obstacle_id': obstacle['id'],
                    'result': 'safe',
                    'score_change': 0,
                    'display': None
                }
        
        return None
    
    def deactivate_obstacle(self, obstacle_id: str):
        """禁用障碍（避免重复计分）"""
        for obstacle in self.active_obstacles:
            if obstacle['id'] == obstacle_id:
                obstacle['active'] = False
                print(f"🎯 障碍物已停用: {obstacle_id}")
                break
    
    def reset(self):
        """重置管理器"""
        self.active_obstacles.clear()
        self.obstacle_id_counter = 0
        self.last_spawn_time = 0
        self.sync_time = 0.0
        self.sync_enabled = False
        print("🔄 障碍物管理器已重置")
    
    def get_statistics(self) -> Dict:
        """🔄 新增：获取统计信息"""
        return {
            'active_count': len([obs for obs in self.active_obstacles if obs['active']]),
            'total_spawned': self.obstacle_id_counter,
            'sync_time': self.sync_time,
            'sync_enabled': self.sync_enabled,
            'difficulty': self.difficulty_level
        }


# 为了向后兼容，保留原有的别名
SyncedObstacleManager = ObstacleManager