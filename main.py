import cv2
import mediapipe as mp
import numpy as np
import random
import time
import os
import json


class HandGestureGame:
    def __init__(self, game_mode='normal', num_targets=3):
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
        self.cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 窗口尺寸（初始化为摄像头分辨率）
        self.window_width = self.cam_width
        self.window_height = self.cam_height
        
        # 游戏模式
        self.game_mode = game_mode  # 'normal' 或 'countdown'
        self.num_targets = num_targets  # 目标点数量
        
        # 游戏状态
        self.score = 0
        self.high_score = self.load_high_score()
        self.start_time = time.time()
        self.paused = False
        self.game_over = False
        
        # 倒计时模式设置
        if self.game_mode == 'countdown':
            self.countdown_duration = 60  # 60秒倒计时
            self.time_left = self.countdown_duration
        
        # 目标点列表（蓝色）
        self.targets = []
        self.target_radius = 20
        self.collision_distance = 30
        for _ in range(self.num_targets):
            self.targets.append(self.generate_random_target())
        
        # 食指位置（红色点）
        self.finger_pos = None
        self.finger_radius = 10
        
        # 创建可调整大小的窗口
        cv2.namedWindow('Hand Gesture Game', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Hand Gesture Game', self.window_width, self.window_height)
        
    def load_high_score(self):
        """从文件加载最高分"""
        try:
            if os.path.exists('highscore.json'):
                with open('highscore.json', 'r') as f:
                    data = json.load(f)
                    return data.get('high_score', 0)
        except:
            pass
        return 0
    
    def save_high_score(self):
        """保存最高分到文件"""
        try:
            with open('highscore.json', 'w') as f:
                json.dump({'high_score': self.high_score}, f)
        except:
            pass
    
    def generate_random_target(self):
        """生成随机目标点位置"""
        margin = 50  # 边距，避免目标点太靠近边缘
        x = random.randint(margin, max(margin + 10, self.window_width - margin))
        y = random.randint(margin, max(margin + 10, self.window_height - margin))
        return (x, y)
    
    def calculate_distance(self, pos1, pos2):
        """计算两点之间的欧氏距离"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def check_collision(self):
        """检测碰撞，返回被碰撞的目标点索引"""
        if self.finger_pos is None:
            return None
        
        for i, target_pos in enumerate(self.targets):
            distance = self.calculate_distance(self.finger_pos, target_pos)
            if distance < self.collision_distance:
                return i
        return None
    
    def format_time(self, seconds):
        """格式化时间为 分:秒"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def draw_ui(self, frame):
        """绘制游戏UI"""
        # 计算时间
        if self.game_mode == 'countdown':
            elapsed = time.time() - self.start_time
            self.time_left = max(0, self.countdown_duration - elapsed)
            time_str = self.format_time(self.time_left)
            time_label = "Time Left: "
            time_color = (0, 255, 255) if self.time_left > 10 else (0, 0, 255)  # 少于10秒变红色
        else:
            elapsed_time = time.time() - self.start_time
            time_str = self.format_time(elapsed_time)
            time_label = "Time: "
            time_color = (0, 255, 255)
        
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
        
        # 绘制最高分（左上角第二行）
        cv2.putText(
            frame, 
            f"High Score: {self.high_score}", 
            (10, 85), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (255, 255, 0), 
            2
        )
        
        # 绘制时间（右上角）
        time_text = f"{time_label}{time_str}"
        text_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        cv2.putText(
            frame, 
            time_text, 
            (self.window_width - text_size[0] - 10, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.2, 
            time_color, 
            3
        )
        
        # 绘制所有蓝色目标点
        for target_pos in self.targets:
            cv2.circle(frame, target_pos, self.target_radius, (255, 0, 0), -1)
            cv2.circle(frame, target_pos, self.target_radius, (255, 255, 255), 2)
        
        # 绘制食指红点
        if self.finger_pos:
            cv2.circle(frame, self.finger_pos, self.finger_radius, (0, 0, 255), -1)
            cv2.circle(frame, self.finger_pos, self.finger_radius, (255, 255, 255), 2)
        
        # 绘制暂停提示
        if self.paused:
            # 半透明暂停遮罩
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.window_width, self.window_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            # 暂停文字
            pause_text = "PAUSED"
            text_size = cv2.getTextSize(pause_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
            text_x = (self.window_width - text_size[0]) // 2
            text_y = (self.window_height + text_size[1]) // 2
            cv2.putText(
                frame, 
                pause_text, 
                (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                3, 
                (255, 255, 255), 
                5
            )
            
            # 继续提示
            continue_text = "Press SPACE to continue"
            text_size2 = cv2.getTextSize(continue_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x2 = (self.window_width - text_size2[0]) // 2
            text_y2 = text_y + 60
            cv2.putText(
                frame, 
                continue_text, 
                (text_x2, text_y2), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (200, 200, 200), 
                2
            )
        
        # 绘制游戏结束界面
        if self.game_over:
            # 半透明遮罩
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.window_width, self.window_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # 游戏结束文字
            game_over_text = "GAME OVER"
            text_size = cv2.getTextSize(game_over_text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 5)[0]
            text_x = (self.window_width - text_size[0]) // 2
            text_y = (self.window_height - 100) // 2
            cv2.putText(
                frame, 
                game_over_text, 
                (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                2.5, 
                (0, 0, 255), 
                5
            )
            
            # 最终分数
            final_score_text = f"Final Score: {self.score}"
            text_size2 = cv2.getTextSize(final_score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x2 = (self.window_width - text_size2[0]) // 2
            text_y2 = text_y + 60
            cv2.putText(
                frame, 
                final_score_text, 
                (text_x2, text_y2), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, 
                (255, 255, 255), 
                3
            )
            
            # 如果打破纪录
            if self.score >= self.high_score and self.score > 0:
                new_record_text = "NEW HIGH SCORE!"
                text_size3 = cv2.getTextSize(new_record_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                text_x3 = (self.window_width - text_size3[0]) // 2
                text_y3 = text_y2 + 50
                cv2.putText(
                    frame, 
                    new_record_text, 
                    (text_x3, text_y3), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.2, 
                    (0, 255, 255), 
                    2
                )
            
            # 退出提示
            quit_text = "Press 'q' to quit"
            text_size4 = cv2.getTextSize(quit_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x4 = (self.window_width - text_size4[0]) // 2
            text_y4 = self.window_height - 50
            cv2.putText(
                frame, 
                quit_text, 
                (text_x4, text_y4), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (200, 200, 200), 
                2
            )
        else:
            # 正常游戏提示信息
            cv2.putText(
                frame, 
                "SPACE: Pause | Q: Quit", 
                (10, self.window_height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (200, 200, 200), 
                2
            )
        
    def run(self):
        """主游戏循环"""
        mode_text = "倒计时挑战模式（60秒）" if self.game_mode == 'countdown' else "普通模式"
        print(f"游戏开始！模式: {mode_text}")
        print(f"目标点数量: {self.num_targets}")
        print("用你的食指触碰蓝色点来得分。")
        print("按空格键暂停游戏，按 'q' 键退出游戏。")
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("无法读取摄像头画面")
                break
            
            # 翻转画面（镜像效果）
            frame = cv2.flip(frame, 1)
            
            # 尝试获取当前窗口大小（添加错误处理）
            try:
                window_rect = cv2.getWindowImageRect('Hand Gesture Game')
                if window_rect[2] > 0 and window_rect[3] > 0:
                    new_width = window_rect[2]
                    new_height = window_rect[3]
                    
                    # 如果窗口大小改变，更新相关参数
                    if new_width != self.window_width or new_height != self.window_height:
                        self.window_width = new_width
                        self.window_height = new_height
                        
                        # 重新生成所有目标点确保在窗口范围内
                        margin = 50
                        for i in range(len(self.targets)):
                            if self.targets[i][0] > self.window_width - margin or \
                               self.targets[i][1] > self.window_height - margin:
                                self.targets[i] = self.generate_random_target()
            except:
                pass  # 忽略窗口获取错误
            
            # 调整帧大小以适应窗口
            frame_resized = cv2.resize(frame, (self.window_width, self.window_height))
            
            # 只在游戏未暂停且未结束时进行游戏逻辑
            if not self.paused and not self.game_over:
                # 检查倒计时是否结束
                if self.game_mode == 'countdown' and self.time_left <= 0:
                    self.game_over = True
                    if self.score > self.high_score:
                        self.high_score = self.score
                        self.save_high_score()
                        print(f"\n🎉 新纪录！最高分: {self.high_score}")
                
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
                            frame_resized,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # 获取食指尖端坐标（landmark 8）
                        index_finger_tip = hand_landmarks.landmark[8]
                        
                        # 转换为像素坐标（基于调整后的窗口大小）
                        finger_x = int(index_finger_tip.x * self.window_width)
                        finger_y = int(index_finger_tip.y * self.window_height)
                        self.finger_pos = (finger_x, finger_y)
                
                # 检测碰撞
                collision_index = self.check_collision()
                if collision_index is not None:
                    self.score += 1
                    # 替换被碰撞的目标点
                    self.targets[collision_index] = self.generate_random_target()
                    print(f"得分！当前分数: {self.score}")
                    
                    # 更新最高分
                    if self.score > self.high_score:
                        self.high_score = self.score
            
            # 绘制UI
            self.draw_ui(frame_resized)
            
            # 显示画面
            cv2.imshow('Hand Gesture Game', frame_resized)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # 空格键暂停/继续
                if not self.game_over:
                    self.paused = not self.paused
                    if self.paused:
                        print("游戏已暂停")
                    else:
                        print("游戏继续")
                        # 暂停后继续时，调整开始时间
                        if self.game_mode == 'countdown':
                            pause_duration = time.time() - self.start_time - (self.countdown_duration - self.time_left)
                            self.start_time += pause_duration
        
        # 游戏结束
        if self.game_mode == 'countdown':
            print(f"\n⏱️ 时间到！")
        
        elapsed_time = time.time() - self.start_time
        print(f"\n游戏结束！")
        print(f"最终分数: {self.score}")
        if self.game_mode == 'normal':
            print(f"游戏时长: {self.format_time(elapsed_time)}")
        print(f"最高分: {self.high_score}")
        
        # 保存最高分
        if self.score > self.high_score:
            self.high_score = self.score
        self.save_high_score()
        
        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


def main():
    """主函数，选择游戏模式"""
    print("=" * 50)
    print("欢迎来到手势控制游戏！")
    print("=" * 50)
    print("\n请选择游戏模式：")
    print("1. 普通模式（无时间限制）")
    print("2. 倒计时挑战模式（60秒）")
    
    try:
        mode_choice = input("\n请输入模式编号 (1 或 2，默认为 1): ").strip()
        if mode_choice == '2':
            game_mode = 'countdown'
        else:
            game_mode = 'normal'
        
        num_targets = input("请输入目标点数量 (1-10，默认为 3): ").strip()
        if num_targets.isdigit() and 1 <= int(num_targets) <= 10:
            num_targets = int(num_targets)
        else:
            num_targets = 3
    except:
        game_mode = 'normal'
        num_targets = 3
    
    print(f"\n启动游戏：{'倒计时模式' if game_mode == 'countdown' else '普通模式'}，{num_targets} 个目标点")
    print("-" * 50)
    
    game = HandGestureGame(game_mode=game_mode, num_targets=num_targets)
    game.run()


if __name__ == "__main__":
    main()
