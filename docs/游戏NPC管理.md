
# 游戏NPC管理系统
本模块提供游戏NPC管理系统，采用模块化设计，包含三维坐标管理、状态机和基础NPC行为实现。主要组件包括：
- `NPCState` 状态枚举
- `Position` 三维时空坐标类
- `BaseNPC` NPC基类
---

## NPCState 枚举
#### 1. 状态类型
| 状态值 | 说明 |
|--------|------|
| IDLE | 空闲状态（默认）|
| MOVING | 空间坐标移动中 |
| INTERACTING | 与玩家/环境交互 |
| COMBAT | 战斗状态 |
注意里面的状态可以扩展。

---

## Position 类
#### 1. 属性说明
| 属性 | 类型 | 说明 |
|------|------|------|
| x | int/float | 横向坐标 |
| y | int/float | 纵向坐标 |
| t | int | 时间维度坐标 |

#### 2. 核心方法
```python
update(x=None, y=None, t=None)  # 更新坐标参数（None值保持不变）
get_coords() -> tuple  # 返回(x, y, t)元组
__str__() -> str  # 返回可读坐标字符串
```

---

## BaseNPC 类
#### 1.  初始化参数
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| identifier | int | 是 | - | 唯一标识符 |
| name | str | 是 | - | 正式名称 |
| position | Position | 否 | Position() | 三维坐标 |
| state | NPCState | 否 | NPCState.IDLE | 当前状态 |
| nickname | str | 否 | None | 显示用昵称 |
| age | int | 否 | 0 | 角色年龄 |
| image_path | str | 否 | "" | PNG素材路径 |
| faction | str | 否 | "中立" | 所属阵营 |
| quotes | list[str] | 否 | [] | 经典语录库 |

#### 2. 核心方法
属性访问（只读）
```python
.identifier  # 获取唯一标识符
.position  # 获取Position对象
.state  # 获取当前NPCState
```
状态管理
```python
update_state(new_state: NPCState)  # 切换NPC状态
```
#### 空间移动
```python
move_to(x: Number, y: Number)  # 空间坐标移动（自动触发MOVING状态）
```
时间操作
```python
time_travel(t: int)  # 修改时间维度坐标
```
交互功能
```python
speak() -> str  # 随机返回一条经典语录
```
信息查询
```python
get_info() -> dict  # 返回包含所有属性的字典
```

---

## 使用示例
#### 1. 创建NPC
```python
npc = BaseNPC(
    identifier=1002,
    name="XXX",
    nickname="xxx",
    age=28,
    position=Position(120, 45, 2023),
    faction="xxxx",
    image_path=None,
    quotes=[
        "xxxx,xxxxx!",
        "xxxxxxx~"
    ]
)
```

#### 2. 执行操作
```python
npc.move_to(130, 50)  # 空间移动
npc.time_travel(2025)  # 时间跳跃
print(npc.speak())  # 随机语录输出
print(npc.get_info())  # 查看完整信息
```

#### 3. 输出示例
```
XXX说：'xxxx,xxxxx!'
{
    'ID': 1002,
    '姓名': 'XXX',
    '昵称': 'xxx',
    '年龄': 28,
    '位置': '空间坐标(130, 50) @ 时间2025',
    '状态': '移动中',
    '阵营': 'xxxx',
    '图片': None,
    '语录数量': 2
}
```

