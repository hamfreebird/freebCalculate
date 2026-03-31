import math
import random
import sys
from typing import List, Tuple

import pygame

# 初始化pygame
pygame.init()

# 屏幕设置
WIDTH, HEIGHT = 1200, 800
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Zen Fractal - 旋转龙分形")

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
COLORS = [
    (255, 100, 100),  # 红色
    (100, 255, 100),  # 绿色
    (100, 100, 255),  # 蓝色
    (255, 255, 100),  # 黄色
    (255, 100, 255),  # 洋红
    (100, 255, 255),  # 青色
]


# 光晕粒子类
class GlowParticle:
    def __init__(self):
        self.x = random.randint(0, WIDTH)
        self.y = random.randint(0, HEIGHT)
        self.size = random.randint(2, 8)
        self.speed_x = random.uniform(-0.5, 0.5)
        self.speed_y = random.uniform(-0.5, 0.5)
        self.color = random.choice(
            [
                (255, 255, 200, 30),  # 淡黄色
                (200, 255, 255, 30),  # 淡青色
                (255, 200, 255, 30),  # 淡洋红
                (255, 255, 255, 20),  # 白色
            ]
        )
        self.pulse_speed = random.uniform(0.01, 0.03)
        self.pulse_offset = random.uniform(0, math.pi * 2)

    def update(self):
        self.x += self.speed_x
        self.y += self.speed_y

        # 边界检查
        if self.x < 0 or self.x > WIDTH:
            self.speed_x *= -1
        if self.y < 0 or self.y > HEIGHT:
            self.speed_y *= -1

        # 脉冲效果
        pulse = math.sin(pygame.time.get_ticks() * self.pulse_speed + self.pulse_offset)
        self.current_size = self.size * (0.8 + 0.2 * pulse)

    def draw(self, surface):
        pygame.draw.circle(
            surface, self.color, (int(self.x), int(self.y)), int(self.current_size)
        )


# 龙分形生成器
class DragonFractal:
    def __init__(self):
        self.points = []
        self.rotation_angle = 0
        self.rotation_speed = 0.5  # 度/帧
        self.scale = 1.0
        self.scale_speed = 0.001
        self.max_iterations = 15
        self.current_iterations = 10
        self.color_index = 0
        self.color_change_speed = 0.1

    def generate_dragon_curve(self, iterations: int) -> List[Tuple[float, float]]:
        """生成龙曲线点"""
        # 使用L-system生成龙曲线
        sequence = "FX"
        for _ in range(iterations):
            new_seq = ""
            for char in sequence:
                if char == "X":
                    new_seq += "X+YF+"
                elif char == "Y":
                    new_seq += "-FX-Y"
                else:
                    new_seq += char
            sequence = new_seq

        # 解析序列生成点
        points = []
        x, y = 0, 0
        angle = 0
        step = 8

        for char in sequence:
            if char == "F":
                x += step * math.cos(math.radians(angle))
                y += step * math.sin(math.radians(angle))
                points.append((x, y))
            elif char == "+":
                angle += 90
            elif char == "-":
                angle -= 90

        return points

    def update(self):
        """更新分形状态"""
        # 旋转
        self.rotation_angle += self.rotation_speed
        if self.rotation_angle >= 360:
            self.rotation_angle -= 360

        # 缩放
        self.scale += self.scale_speed
        if self.scale > 1.5 or self.scale < 0.5:
            self.scale_speed *= -1

        # 颜色变化
        self.color_index += self.color_change_speed
        if self.color_index >= len(COLORS):
            self.color_index = 0

        # 更新分形点
        self.points = self.generate_dragon_curve(self.current_iterations)

    def draw(self, surface):
        """绘制分形"""
        if not self.points:
            return

        # 计算变换矩阵
        cos_a = math.cos(math.radians(self.rotation_angle))
        sin_a = math.sin(math.radians(self.rotation_angle))

        # 变换并绘制点
        transformed_points = []
        for x, y in self.points:
            # 缩放
            sx = x * self.scale
            sy = y * self.scale

            # 旋转
            rx = sx * cos_a - sy * sin_a
            ry = sx * sin_a + sy * cos_a

            # 平移至中心
            tx = CENTER_X + rx
            ty = CENTER_Y + ry

            transformed_points.append((tx, ty))

        # 绘制线条
        color = COLORS[int(self.color_index) % len(COLORS)]
        if len(transformed_points) > 1:
            # 绘制主线条
            pygame.draw.lines(surface, color, False, transformed_points, 2)

            # 绘制渐变效果的点
            for i, (x, y) in enumerate(transformed_points):
                alpha = i / len(transformed_points)
                point_color = (
                    int(color[0] * alpha),
                    int(color[1] * alpha),
                    int(color[2] * alpha),
                )
                radius = max(1, int(3 * alpha))
                pygame.draw.circle(surface, point_color, (int(x), int(y)), radius)


# 创建光晕层
def create_glow_surface():
    """创建光晕表面"""
    glow_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    # 创建径向渐变
    for radius in range(200, 0, -10):
        alpha = int(10 * (radius / 200))
        color = (255, 255, 255, alpha)
        pygame.draw.circle(glow_surface, color, (CENTER_X, CENTER_Y), radius)

    return glow_surface


# 主函数
def main():
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # 创建分形
    dragon = DragonFractal()

    # 创建光晕粒子
    glow_particles = [GlowParticle() for _ in range(100)]

    # 创建光晕表面
    glow_surface = create_glow_surface()

    running = True
    show_controls = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    show_controls = not show_controls
                elif event.key == pygame.K_UP:
                    dragon.current_iterations = min(
                        dragon.current_iterations + 1, dragon.max_iterations
                    )
                elif event.key == pygame.K_DOWN:
                    dragon.current_iterations = max(dragon.current_iterations - 1, 3)
                elif event.key == pygame.K_r:
                    dragon.rotation_speed *= -1
                elif event.key == pygame.K_c:
                    dragon.color_change_speed *= -1

        # 清屏
        screen.fill(BLACK)

        # 绘制光晕背景
        screen.blit(glow_surface, (0, 0), special_flags=pygame.BLEND_ADD)

        # 更新和绘制光晕粒子
        for particle in glow_particles:
            particle.update()
            particle.draw(screen)

        # 更新和绘制分形
        dragon.update()
        dragon.draw(screen)

        # 绘制控制提示
        if show_controls:
            controls = [
                "up/down: Increase/reduce fractal iterations",
                "Space: display/hide control prompt",
                "ESC: exit program",
            ]
            # controls = [
            #     "控制说明:",
            #     "↑/↓: 增加/减少分形迭代次数",
            #     "R: 反转旋转方向",
            #     "C: 反转颜色变化方向",
            #     "空格: 显示/隐藏控制提示",
            #     "ESC: 退出程序",
            #     f"当前迭代: {dragon.current_iterations}",
            #     f"旋转速度: {dragon.rotation_speed:.1f}°/帧",
            # ]

            for i, text in enumerate(controls):
                text_surface = font.render(text, True, WHITE)
                screen.blit(text_surface, (10, 10 + i * 25))

        # 绘制标题
        title_font = pygame.font.SysFont(None, 48)
        title = title_font.render("Zen Fractal - Rotary dragon fractal", True, WHITE)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 20))

        # 绘制帧率
        fps_text = font.render(f"FPS: {int(clock.get_fps())}", True, WHITE)
        screen.blit(fps_text, (WIDTH - 100, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
