#calculation.py-step2
import math
import numpy as np
global_vars = globals()
# ------ CONFIGURATION ------
# Define the physical and fluid parameters
PARAMETERS = {
    "rho_20": 915,
    "g": 9.81,
    "t0": 25,
    "K": 14,
    "P_bar": 25,
    "d_4":0.915
}
# ------ AUXILIARY FUNCTIONS ------


def xi_1(rho_20):
    return 1.825e-3 - 1.315e-6 * rho_20

def rho_t(t_x):
    rho_20 = PARAMETERS["rho_20"]
    rho_t = rho_20 - xi_1(rho_20) * (t_x - 20)
    # print(f"Calculated density at {t_x}°C: {rho_t} kg/m³ (using rho_20: {rho_20} kg/m³)")
    return rho_t

# 计算比热容
def c(t_x):
    d_4 = PARAMETERS["d_4"]
    c_rho_t = rho_t(t_x)
    c_value = (1 / math.sqrt(d_4 ** 15)) * (1.687 + 3.39e-3 * c_rho_t) # 使用新的变量名c_value

    return c_value

def a(D, t_x, Q):
    rho = rho_t(t_x)
    C = c(t_x)
    K = PARAMETERS["K"]
    a_value=(K * 3.14 * D) / (rho * Q * C)
    # print(f"a的值为：{a_value}")
    return a_value
def velocity(D, Q):
    if D <= 0:
        raise ValueError(f"Diameter D cannot be zero or negative, got: {D}")
    v = (4 * Q) / (math.pi * D ** 2)
    # print(f"Calculated velocity: {v} (m/s) for Diameter: {D} (m) and Flow rate: {Q} (m³/s)")
    return v

def relative_roughness(D):
    # 假设粗糙度e是已知的，以米为单位
    e = 0.045  # 这只是一个例子，实际值应该基于管道的物理特性
    return e / D

def colebrook_equation(f, D, Q, t_x):
    # 检查无效输入
    if D <= 0 or Q <= 0 or t_x <= 0:
        raise ValueError("Physical parameters must be greater than zero.")

    Re =colebrook_equation(f,D, Q, t_x)
    eD = relative_roughness(D)  # 你可能需要定义这个函数或以其他方式获取这个值

    # 检查Re和f是否有效
    if Re <= 0 or f <= 0:
        raise ValueError("Reynolds number and friction factor must be greater than zero.")

    # 避免数学错误
    try:
        sqrt_f = math.sqrt(f)
        term1 = eD / 3.7
        term2 = 2.51 / (Re * sqrt_f)

        # 如果term1 + term2 大于 1，对数将是负数
        if term1 + term2 >= 1:
            raise ValueError("Logarithm argument out of bounds. Check input values.")

        right_side = -2 * math.log10(term1 + term2)
    except ValueError as e:
        raise ValueError(f"Error in colebrook equation calculation: {e}")

    # 计算方程的左侧
    left_side = 1 / sqrt_f

    # 返回两边之差
    return left_side - right_side

def newton_raphson(D, Q, t_x, initial_guess=0.07, max_iterations=600, tolerance=1e-6, verbose=False):
    # 使用Newton-Raphson方法求解
    f_guess = initial_guess
    for iteration in range(max_iterations):
        try:
            f_val = colebrook_equation(f_guess, D=D, Q=Q, t_x=t_x)
            # 计算导数
            df = (colebrook_equation(f_guess + tolerance, D, Q, t_x) - f_val) / tolerance

            if df == 0:  # 避免除以零
                raise ValueError("Derivative was zero. No solution found.")

            # 更新估计值
            f_next = f_guess - f_val / df

            # 如果需要打印迭代信息，仅在verbose为True时打印
            if verbose:
                print(f"Iteration {iteration}: f_guess = {f_guess}, f_next = {f_next}, f_val = {f_val}, df = {df}")

            # 检查收敛性
            if abs(f_next - f_guess) < tolerance:
                return f_next
            f_guess = f_next

        except ValueError as e:
            # 如果发生错误，并且需要打印错误信息，仅在verbose为True时打印
            if verbose:
                print(f"Error during iteration {iteration}: {e}")
            break  # 退出迭代

    # 如果需要打印未收敛的信息，仅在verbose为True时打印
    if verbose:
        print(f"Did not converge after {max_iterations} iterations, last friction factor: {f_guess}")
    return f_guess


def get_friction_factor(D, Q, t_x):
    friction_factor = newton_raphson(D, Q, t_x)
    if friction_factor is None:  # 这里实际上不再需要，但为了完整性我保留了它
        raise ValueError("Unable to compute friction factor!")
    return friction_factor

def i(D, Q, t_x):
    lambda_m = get_friction_factor(D, Q, t_x)
    V = velocity(D, Q)
    g = PARAMETERS["g"]
    hydraulic_slope = lambda_m * (1 / D) * ((V ** 2) / (2 * g))
    # print(f"速度为: {V}")
    # print(f"水力坡降为: {hydraulic_slope}")
    return hydraulic_slope

def b(t_x, D, Q):
    rho = rho_t(t_x)
    # a_value=a(D, t_x, Q)
    g = PARAMETERS["g"]
    I = i(D, Q, t_x)
    k = PARAMETERS["K"]
    b_value=(rho * Q * g * I) / (k * math.pi * D)
    # print(f"b的值为：{b_value}")
    return b_value
    # return(I*g)/(k*a_value)

    # return (t_x - t0 - b_value + t0 * e_aL + b_value * e_aL) / e_aL
def solve_tL(t_x, D, L, Q):
    b_value = b(t_x, D, Q)
    a_value = a(D, t_x, Q)
    t0 = PARAMETERS["t0"]

    # 设置一个阈值，比如100，来判断当a_value * L的结果非常大时如何处理
    exp_threshold = 100
    if a_value * L > exp_threshold:
        # 这里简化处理，因为exp(a_value * L)会非常大
        return t0  # 或者是其他根据您模型适当的处理方式
    else:
        e_aL = math.exp(a_value * L)
        return (t_x - t0 - b_value + t0 * e_aL + b_value * e_aL) / e_aL

# 质量流量
def Gm(t_x, Q):
    mass_flow = rho_t(t_x) * Q
    return mass_flow


def calculate_P(t_x, D, P_bar_MPa, L, Q):
    # 将输入的MPa转换为Pa
    P_bar = P_bar_MPa * 10 ** 6


    G = Gm(t_x, Q)
    V_m = velocity(D, Q)
    lambda_m = get_friction_factor(D, Q, t_x)
    rho_m = rho_t(t_x)

    # print(f"calculate_P - nu: {nu}, G: {G}, V_m: {V_m}, lambda_m: {lambda_m}, rho_m: {rho_m}")

    numerator = lambda_m * (2 * V_m * G) / (math.pi * D ** 2)
    denominator = 1 - (rho_m * V_m * V_m) / P_bar

    if denominator == 0:
        print("Error: Denominator is zero, the equation is not solvable in this case.")
        return float('nan')

    delta_P_Pa = numerator / denominator * L
    delta_P_MPa = delta_P_Pa / 10 ** 6

    return delta_P_MPa


def calculate_hydrostatic_pressure(t_x, height):
    g = PARAMETERS["g"]  # 重力加速度，单位是 m/s^2
    rho=rho_t(t_x)
    hydrostatic_pressure = rho * g * height
    return hydrostatic_pressure/1000



