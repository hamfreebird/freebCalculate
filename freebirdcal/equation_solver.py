import cmath


class EquationSolver:
    def solve_linear_1v(self, a, b):
        """解一元一次方程 ax + b = 0
        返回解或解的情况说明"""
        if a == 0:
            return "无解" if b != 0 else "无穷多解"
        return -b / a

    def solve_quadratic_1v(self, a, b, c):
        """解一元二次方程 ax² + bx + c = 0
        返回包含两个解的元组（可能为复数）"""
        if a == 0:
            return self.solve_linear_1v(b, c)
        discriminant = b ** 2 - 4 * a * c
        sqrt_d = cmath.sqrt(discriminant)
        return ((-b + sqrt_d) / (2 * a), (-b - sqrt_d) / (2 * a))

    def solve_linear_2v(self, coeff_matrix, constants):
        """解二元一次方程组
        [[a1, b1], [a2, b2]] 对应方程组：
        a1*x + b1*y = c1
        a2*x + b2*y = c2
        constants = [c1, c2]
        返回解元组或解的情况说明"""
        a1, b1 = coeff_matrix[0]
        a2, b2 = coeff_matrix[1]
        c1, c2 = constants

        det = a1 * b2 - a2 * b1
        if det == 0:
            # Check if equations are consistent
            if (a1 * c2 == a2 * c1) and (b1 * c2 == b2 * c1):
                return "无穷多解"
            return "无解"
        x = (b2 * c1 - b1 * c2) / det
        y = (a1 * c2 - a2 * c1) / det
        return (x, y)

    def solve_linear_3v(self, coeff_matrix, constants):
        """解三元一次方程组
        [[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]] 对应：
        a1*x + b1*y + c1*z = d1
        a2*x + b2*y + c2*z = d2
        a3*x + b3*y + c3*z = d3
        constants = [d1, d2, d3]
        返回解元组或解的情况说明"""
        # 计算系数矩阵的行列式
        a, b, c = coeff_matrix[0]
        d, e, f = coeff_matrix[1]
        g, h, i = coeff_matrix[2]

        det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        if det == 0:
            return "无解或无穷多解"

        # 计算各变量的行列式
        dx = constants[0] * (e * i - f * h) - b * (constants[1] * i - f * constants[2]) + c * (
                    constants[1] * h - e * constants[2])
        dy = a * (constants[1] * i - f * constants[2]) - constants[0] * (d * i - f * g) + c * (
                    d * constants[2] - constants[1] * g)
        dz = a * (e * constants[2] - constants[1] * h) - b * (d * constants[2] - constants[1] * g) + constants[0] * (
                    d * h - e * g)

        x = dx / det
        y = dy / det
        z = dz / det
        return (x, y, z)

    def solve_quadratic_2v(self, linear_eq, quadratic_eq):
        """解由线性方程和二次方程组成的二元二次方程组
        linear_eq: (A, B, C) 对应 Ax + By = C
        quadratic_eq: (a, b, c, d, e, f) 对应 ax² + by² + cxy + dx + ey + f = 0
        返回解的列表（可能包含复数解）"""
        A, B, C = linear_eq
        a, b, c, d, e, f = quadratic_eq

        solutions = []

        if B != 0:  # 用x表示y
            # y = (C - A*x)/B
            def y_sub(x):
                return (C - A * x) / B

            # 代入二次方程并展开
            x2_coeff = a + (b * A ** 2) / B ** 2 - (c * A) / B
            x_coeff = (-2 * b * A * C) / B ** 2 + (c * C) / B + d - (e * A) / B
            const = (b * C ** 2) / B ** 2 - (e * C) / B + f

            # 解二次方程
            x_solutions = self.solve_quadratic_1v(x2_coeff, x_coeff, const)
            for x in x_solutions:
                if isinstance(x, complex):
                    solutions.append((x, y_sub(x)))
                else:
                    solutions.append((x, y_sub(x)))
        elif A != 0:  # 用y表示x
            # x = (C - B*y)/A
            def x_sub(y):
                return (C - B * y) / A

            # 代入二次方程并展开
            y2_coeff = b + (a * B ** 2) / A ** 2 - (c * B) / A
            y_coeff = (-2 * a * B * C) / A ** 2 + (c * C) / A + e - (d * B) / A
            const = (a * C ** 2) / A ** 2 - (d * C) / A + f

            # 解二次方程
            y_solutions = self.solve_quadratic_1v(y2_coeff, y_coeff, const)
            for y in y_solutions:
                solutions.append((x_sub(y), y))
        else:
            return "无效方程"

        return solutions


if __name__ == "__main__":
    solver = EquationSolver()

    # 一元一次方程
    print("一元一次方程解:", solver.solve_linear_1v(2, -4))  # 2x -4 = 0 → x=2

    # 一元二次方程
    print("一元二次方程解:", solver.solve_quadratic_1v(1, -3, 2))  # x²-3x+2=0 → (2,1)

    # 二元一次方程组
    print("二元一次方程组解:",
          solver.solve_linear_2v([[3, 2], [2, -1]], [7, 4]))  # 3x+2y=7, 2x-y=4 → (3,2)

    # 三元一次方程组
    print("三元一次方程组解:",
          solver.solve_linear_3v([[2, 1, 1], [1, 3, 2], [1, 0, 1]], [4, 5, 1]))  # 解为(1,1,1)

    # 二元二次方程组（线性+二次）
    print("二元二次方程组解:",
          solver.solve_quadratic_2v((1, 1, 3), (1, 1, 0, 0, 0, -9)))  # x+y=3, x²+y²=9 → (3,0)和(0,3)

