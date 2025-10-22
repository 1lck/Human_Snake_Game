import cv2
import mediapipe as mp
import numpy as np
import random
import time


class HandGestureGame:
    def __init__(self):
        # 初始化 MediaPipe 手部检测
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 配置手部检测器
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # 摄像头初始化
        self.cap = cv2.VideoCapture(0)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 游戏状态
        self.score = 0
        self.start_time = time.time()
        
        # 目标点（蓝色）
        self.target_pos = self.generate_random_target()
        self.target_radius = 20
        self.collision_distance = 30
        
        # 食指位置（红色点）
        self.finger_pos = None
        self.finger_radius = 10
        
    def generate_random_target(self):
        """生成随机目标点位置"""
        margin = 50  # 边距，避免目标点太靠近边缘
        x = random.randint(margin, self.frame_width - margin)
        y = random.randint(margin, self.frame_height - margin)
        return (x, y)
    
    def calculate_distance(self, pos1, pos2):
        """计算两点之间的欧氏距离"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def check_collision(self):
        """检测碰撞"""
        if self.finger_pos is None:
            return False
        
        distance = self.calculate_distance(self.finger_pos, self.target_pos)
        return distance < self.collision_distance
    
    def format_time(self, seconds):
        """格式化时间为 分:秒"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def draw_ui(self, frame):
        """绘制游戏UI"""
        # 计算游戏时长
        elapsed_time = time.time() - self.start_time
        time_str = self.format_time(elapsed_time)
        
        # 绘制分数（左上角）
        cv2.putText(
            frame, 
            f"Score: {self.score}", 
            (10, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.2, 
            (0, 255, 0), 
            3
        )
        
        # 绘制时间（右上角）
        time_text = f"Time: {time_str}"
        text_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        cv2.putText(
            frame, 
            time_text, 
            (self.frame_width - text_size[0] - 10, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.2, 
            (0, 255, 255), 
            3
        )
        
        # 绘制蓝色目标点
        cv2.circle(frame, self.target_pos, self.target_radius, (255, 0, 0), -1)
        cv2.circle(frame, self.target_pos, self.target_radius, (255, 255, 255), 2)
        
        # 绘制食指红点
        if self.finger_pos:
            cv2.circle(frame, self.finger_pos, self.finger_radius, (0, 0, 255), -1)
            cv2.circle(frame, self.finger_pos, self.finger_radius, (255, 255, 255), 2)
        
        # 绘制提示信息
        cv2.putText(
            frame, 
            "Press 'q' to quit", 
            (10, self.frame_height - 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (200, 200, 200), 
            2
        )
        
    def run(self):
        """主游戏循环"""
        print("游戏开始！用你的食指触碰蓝色点来得分。")
        print("按 'q' 键退出游戏。")
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("无法读取摄像头画面")
                break
            
            # 翻转画面（镜像效果）
            frame = cv2.flip(frame, 1)
            
            # 转换颜色空间（MediaPipe 需要 RGB）
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 处理手部检测
            results = self.hands.process(rgb_frame)
            
            # 重置食指位置
            self.finger_pos = None
            
            # 如果检测到手部
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 绘制手部骨架线
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # 获取食指尖端坐标（landmark 8）
                    index_finger_tip = hand_landmarks.landmark[8]
                    
                    # 转换为像素坐标
                    h, w, c = frame.shape
                    finger_x = int(index_finger_tip.x * w)
                    finger_y = int(index_finger_tip.y * h)
                    self.finger_pos = (finger_x, finger_y)
            
            # 检测碰撞
            if self.check_collision():
                self.score += 1
                self.target_pos = self.generate_random_target()
                print(f"得分！当前分数: {self.score}")
            
            # 绘制UI
            self.draw_ui(frame)
            
            # 显示画面
            cv2.imshow('Hand Gesture Game', frame)
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 游戏结束
        elapsed_time = time.time() - self.start_time
        print(f"\n游戏结束！")
        print(f"最终分数: {self.score}")
        print(f"游戏时长: {self.format_time(elapsed_time)}")
        
        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


if __name__ == "__main__":
    game = HandGestureGame()
    game.run()

