from enum import Enum
import random


# 可以扩展
class NPCState(Enum):
    IDLE = "空闲"
    MOVING = "移动中"
    INTERACTING = "交互中"
    COMBAT = "战斗状态"


class Position:
    """三维坐标管理类（二维空间 + 时间维度）"""

    def __init__(self, x=0, y=0, t=0):
        self.x = x
        self.y = y
        self.t = t  # 时间维度

    def update(self, x=None, y=None, t=None):
        """更新坐标参数"""
        if x is not None: self.x = x
        if y is not None: self.y = y
        if t is not None: self.t = t

    def get_coords(self):
        """获取完整坐标信息"""
        return (self.x, self.y, self.t)

    def __str__(self):
        return f"空间坐标({self.x}, {self.y}) @ 时间{self.t}"


class BaseNPC:
    """NPC基类"""

    def __init__(
            self,
            identifier,
            name,
            position=Position(),
            state=NPCState.IDLE,
            nickname=None,
            age=0,
            image_path=None,
            faction="中立",
            quotes=[]
    ):
        self._identifier = identifier
        self._position = position
        self._state = state
        self.name = name
        self.nickname = nickname or name
        self.age = age
        self.image_path = image_path
        self.faction = faction
        self.quotes = quotes.copy()

    # 属性访问方法
    @property
    def identifier(self):
        return self._identifier

    @property
    def position(self):
        return self._position

    @property
    def state(self):
        return self._state

    # 状态修改方法
    def update_state(self, new_state):
        if not isinstance(new_state, NPCState):
            raise ValueError("无效的状态类型")
        self._state = new_state

    # 位置操作
    def move_to(self, x, y):
        """移动NPC到指定空间坐标"""
        self._position.update(x=x, y=y)
        self.update_state(NPCState.MOVING)

    def time_travel(self, t):
        """时间维度移动"""
        self._position.update(t=t)

    # 对话功能
    def speak(self):
        """随机说一句经典语录"""
        if not self.quotes:
            return f"{self.name}保持沉默"
        return f"{self.name}说：'{random.choice(self.quotes)}'"

    # 信息展示
    def get_info(self):
        """获取完整NPC信息"""
        return {
            "ID": self._identifier,
            "姓名": self.name,
            "昵称": self.nickname,
            "年龄": self.age,
            "位置": str(self._position),
            "状态": self._state.value,
            "阵营": self.faction,
            "图片": self.image_path,
            "语录数量": len(self.quotes)
        }

    def __str__(self):
        return f"{self.name} ({self.nickname}) [{self.faction}]"


if __name__ == "__main__":
    # 创建NPC实例
    freebird = BaseNPC(
        identifier=1001,
        name="自由的飞鸟",
        nickname="freebird",
        age=31415926,
        position=Position(1, 9, 2025),
        faction="天空王国",
        image_path=None,
        quotes=[
            "freebird fly in the sky!",
            "又解决了一个bug！",
            "我的邮箱是：freebirdflyinthesky@gmail.com 欢迎报告bug嗷！",
            "叫我freebird！",
            "冷知识：我其实不是人类（是光之子啦）。"
        ]
    )

    # 测试功能
    print(freebird.get_info())
    freebird.move_to(random.randint(-100, 100), random.randint(-100, 100))
    freebird.time_travel(random.randint(1600, 2025))
    print(freebird.speak())
    print("当前位置：", freebird.position.get_coords())

