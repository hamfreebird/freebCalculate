import cmath
import math


class NumberOperations:
    def __init__(self, number):
        # 解析数学表达式字符串
        if isinstance(number, str):
            number = self._parse_math_expression(number)

        # 转换为复数类型
        self.value = complex(number)

        # 判断数值属性
        self.is_real = (self.value.imag == 0)
        self.is_positive_integer = False
        if self.is_real:
            real_part = self.value.real
            if real_part.is_integer() and real_part > 0:
                self.integer = int(real_part)
                self.is_positive_integer = True

    def _parse_math_expression(self, expr):
        """解析数学表达式字符串为数值"""
        # 安全允许的函数和常量
        allowed = {
            'sqrt': cmath.sqrt,  # 支持复数开根
            'π': math.pi,  # 支持希腊字母π
            'pi': math.pi,
            'e': math.e,
            '**': pow  # 支持幂运算
        }

        try:
            # 统一替换数学符号
            expr = expr.replace('^', '**').replace('√', 'sqrt').replace(' ', '')
            # 在安全环境中求值
            return eval(expr, {'__builtins__': None}, allowed)
        except Exception as e:
            raise ValueError(f"无效的数学表达式: {expr} ({str(e)})")

    def power(self, n):
        return self.value ** n

    def nth_root(self, n):
        if n == 0:
            raise ValueError("n cannot be zero for nth_root")
        return self.value ** (1 / n)

    def logarithm(self, n=None):
        if n is None:
            return cmath.log(self.value)
        else:
            return cmath.log(self.value, n)

    def factorize(self):
        if not self.is_positive_integer:
            return []
        factors = []
        num = self.integer
        while num % 2 == 0:
            factors.append(2)
            num //= 2
        i = 3
        max_factor = math.isqrt(num) + 1
        while i <= max_factor and num > 1:
            while num % i == 0:
                factors.append(i)
                num //= i
                max_factor = math.isqrt(num) + 1
            i += 2
        if num > 1:
            factors.append(num)
        return factors


if __name__ == "__main__":
    # 分数和基本运算
    num1 = NumberOperations("2/3 + 1/6")
    print(num1.value)  # (0.8333333333333333+0j)

    # 希腊字母支持
    num2 = NumberOperations("π/2")
    print(num2.value)  # (1.5707963267948966+0j)

    # 复数运算
    num3 = NumberOperations("sqrt(-4) + 3^2")  # 2j + 9
    print(num3.power(0.5))  # 开平方计算结果

    # 因数分解
    num4 = NumberOperations("12")
    print(num4.factorize())  # [2, 2, 3]

