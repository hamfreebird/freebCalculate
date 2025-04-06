import pygame
import math
import colorsys
import random
from pygame.locals import *
from pygame.math import Vector2

# === 初始化配置 ===
pygame.init()
WIDTH, HEIGHT = 800, 800  # 窗口尺寸
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("freebird fly in the sky~")
clock = pygame.time.Clock()

# === 核心参数 ===
GOLDEN_RATIO = (math.sqrt(5) - 1) / 2  # 黄金分割比例
MAX_DEPTH = 11  # 分形最大递归深度
BASE_ROT_SPEED = 0.03  # 基础旋转速度(弧度/帧)
STAR_COUNT = 100000  # 星尘粒子数量
BLOOM_LAYERS = 5  # 光晕效果层级


def apply_bloom(surface, iterations=2):
    for _ in range(iterations):
        scaled = pygame.transform.smoothscale(surface, (WIDTH // 2, HEIGHT // 2))
        scaled = pygame.transform.smoothscale(scaled, (WIDTH, HEIGHT))
        scaled.set_alpha(80)
        surface.blit(scaled, (0, 0))


class SynestheticExperience:
    """主体验类，整合视觉与音频模拟系统"""

    def __init__(self):
        """初始化系统资源"""
        self.time = 0  # 时间计数器
        self.init_starfield()  # 初始化星尘粒子
        self.init_audio_params()  # 初始化音频参数
        self.bloom_cache = [  # 光晕效果缓存
            pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            for _ in range(BLOOM_LAYERS)
        ]

    def init_starfield(self):
        """生成随机星尘粒子系统"""
        self.starfield = [(
            Vector2(random.random() * WIDTH, random.random() * HEIGHT),
            random.uniform(0.3, 1.0),
            random.choice([0.5, 0.7, 1.0])
        ) for _ in range(STAR_COUNT)]

    def init_audio_params(self):
        """初始化虚拟音频参数"""
        self.bass_beat = 0.0  # 低频节奏强度 (0-1)
        self.treble_peak = 0.0  # 高频能量强度 (0-1)
        self.melody_phase = 0.0  # 旋律相位 (0-1)

    def update_audio_simulation(self):
        """更新虚拟音乐参数生成器"""
        self.time += 1
        # 低频节奏模拟（每2秒一个节拍）
        self.bass_beat = max(0, math.sin(math.radians(self.time * 0.9)) ** 15)

        # 高频节奏模拟（快速变化）
        self.treble_peak = math.sin(self.time * 0.02) ** 3 * 0.8 + 0.2

        # 旋律相位模拟（缓慢循环）
        self.melody_phase = (self.time * 0.005) % 1

    def gradient_background(self, surface):
        """绘制动态渐变背景"""
        for y in range(HEIGHT):
            # 基于垂直位置和时间的色彩变化
            h = 0.6 + math.sin(y / HEIGHT * math.pi * 2 + self.time * 0.001) * 0.02
            l = 0.1 + 0.15 * (1 - y / HEIGHT)  # 亮度渐变
            color = colorsys.hls_to_rgb(h, l, 1)
            pygame.draw.line(surface, [c * 230 for c in color], (0, y), (WIDTH, y))

    def draw_stars(self, surface):
        """绘制音乐驱动的星尘粒子"""
        for i, (pos, size, speed) in enumerate(self.starfield):
            # 粒子漂移运动
            drift = Vector2(
                math.sin(self.time * speed * 0.01 + i),
                math.cos(self.time * speed * 0.008 + i)
            ) * (0.3 + self.treble_peak * 0.5)

            new_pos = pos + drift
            # 粒子强度计算
            intensity = 0.5 + 0.5 * math.sin(self.time * speed * 0.1) * self.treble_peak
            alpha = min(255, 80 * intensity * (0.7 + self.bass_beat * 0.3))

            # 边界循环处理
            if new_pos.x < 0 or new_pos.x > WIDTH:
                new_pos.x = random.random() * WIDTH
            if new_pos.y < 0 or new_pos.y > HEIGHT:
                new_pos.y = random.random() * HEIGHT

            self.starfield[i] = (new_pos, size, speed)
            # 绘制粒子
            pygame.draw.circle(surface,
                               (255, 255, 255, int(alpha)),
                               (int(new_pos.x), int(new_pos.y)),
                               int(size * (1 + self.bass_beat * 0.5)))

    def get_depth_color(self, depth):
        """生成基于景深的颜色"""
        hue = 0.62 + depth * 0.01 + self.melody_phase * 0.1  # 色相偏移
        sat = 0.4 - depth * 0.03  # 饱和度递减
        lum = 0.6 - depth * 0.04  # 亮度递减
        alpha = 180 - depth * 12  # 透明度递减

        rgb = [c * 255 for c in colorsys.hls_to_rgb(hue % 1, lum, sat)]
        return (*rgb, alpha * (0.9 + self.treble_peak * 0.1))

    def recursive_fractal(self, surface, depth, points, angle):
        """递归生成分形结构"""
        if depth > MAX_DEPTH:
            return

        # 景深效果参数
        scale = 0.92 ** depth  # 尺寸缩放
        blur = depth * 0.3  # 模糊强度
        layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

        # 坐标变换与绘制
        rotated = [p.rotate(angle * scale) * scale + Vector2(WIDTH / 2, HEIGHT / 2) for p in points]
        pygame.draw.polygon(layer, self.get_depth_color(depth), rotated,
                width = int(2 - depth * 0.1))

        # 应用模糊效果
        scaled = pygame.transform.smoothscale(layer, (WIDTH // 2, HEIGHT // 2))
        blurred = pygame.transform.smoothscale(scaled, (WIDTH, HEIGHT))
        blurred.set_alpha(80)
        surface.blit(blurred, (0, 0))

        # 生成子分形顶点
        new_points = []
        for i in range(len(points)):
            dir_vec = points[(i + 1) % len(points)] - points[i]
            new_point = points[i] + dir_vec * GOLDEN_RATIO
            # 添加音乐驱动的扰动
            new_point += Vector2.from_polar((self.bass_beat * 3, random.uniform(0, 360)))
            new_points.append(new_point.rotate(angle * 0.2))

        # 递归调用
        self.recursive_fractal(
            surface, depth + 1,
            new_points,
            angle + math.radians(2 + 5 * math.sin(self.time * 0.01 + depth)))


    def generate_light_halo(self):
        """生成多层光晕效果"""
        base = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        # 生成基础分形
        self.recursive_fractal(base, 0, [
            Vector2(0, -200),
            Vector2(173.2, 100),  # 等边三角形顶点
            Vector2(-173.2, 100)
        ], math.radians(self.time * BASE_ROT_SPEED))

        # 生成模糊层级
        for i in range(BLOOM_LAYERS):
            scaled = pygame.transform.smoothscale(base, (WIDTH // 2, HEIGHT // 2))
            scaled = pygame.transform.smoothscale(scaled, (WIDTH, HEIGHT))
            self.bloom_cache[i] = scaled
            self.bloom_cache[i].set_alpha(80 - i * 20)


    def draw(self, surface):
        """主绘制循环"""
        self.update_audio_simulation()
        surface.fill((0, 0, 0))  # 清空画布

        # 绘制背景
        self.gradient_background(surface)
        self.draw_stars(surface)

        # 生成光晕效果
        self.generate_light_halo()

        # 混合光晕层
        for bloom in self.bloom_cache:
            surface.blit(bloom, (0, 0))

        # 添加中心高光
        glow = pygame.Surface((300, 300), pygame.SRCALPHA)
        pygame.draw.circle(glow, (180, 180, 255, 60), (150, 150), 150)
        scaled = pygame.transform.smoothscale(glow, (WIDTH // 2, HEIGHT // 2))
        scaled = pygame.transform.smoothscale(scaled, (WIDTH, HEIGHT))
        scaled.set_alpha(80)
        surface.blit(scaled, (WIDTH // 2 - 150, HEIGHT // 2 - 150))

# === 主程序 ===
if __name__ == "__main__":
    experience = SynestheticExperience()
    running = True


    while running:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False

        experience.draw(screen)
        pygame.display.flip()
        clock.tick(60)  # 锁定60FPS

    pygame.quit()
