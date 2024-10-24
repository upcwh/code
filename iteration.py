# iteration.py
from math_utils import *
class ThresholdChecker:
    def __init__(self, data):
        self.data = data

    def check_pressure(self, node, low_alarm=None, high_alarm=None, lock_threshold=None, high_high_threshold=None, stop_value=None, low_low_threshold=None):
        if node not in self.data:
            return False, None  # 节点不在数据中，静默返回

        pressure = self.data.get(node, 0)  # 获取节点的压力值

        exceeded = False
        alarm_value = None  # 用于存储报警值

        # 检查最严重的阈值：停机阈值和超低压阈值
        if stop_value is not None and pressure > stop_value:
            # print(f"停机警报：{node}大于{stop_value}MPa")
            exceeded = True
            alarm_value = stop_value  # 设置报警值

        elif low_low_threshold is not None and pressure < low_low_threshold:
            # print(f"超低压报警：{node}小于超低低限值{low_low_threshold}MPa")
            exceeded = True
            alarm_value = low_low_threshold

        # 检查其他阈值
        elif high_high_threshold is not None and pressure > high_high_threshold:
            # print(f"超高压报警：{node}大于超高高限值{high_high_threshold}MPa")
            exceeded = True
            alarm_value = high_high_threshold

        elif lock_threshold is not None and pressure < lock_threshold:
            # print(f"联锁警报：{node}小于{lock_threshold}MPa，启动联锁机制")
            exceeded = True
            alarm_value = lock_threshold

        elif high_alarm is not None and pressure > high_alarm:
            # print(f"高压报警：{node}大于{high_alarm}MPa")
            exceeded = True
            alarm_value = high_alarm

        elif low_alarm is not None and pressure < low_alarm:
            # print(f"低压报警：{node}小于{low_alarm}MPa")
            exceeded = True
            alarm_value = low_alarm

        # 如果超过阈值，更新 self.data[node] 的值
        if exceeded:
            self.data[node] = alarm_value

        return exceeded, alarm_value

    def check_liquid_level(self, node, low_limit=None, high_limit=None, low_low_limit=None, high_high_limit=None):
        if node not in self.data:
            return False, None
        level = self.data.get(node, 0)
        exceeded = False
        alarm_value = None  # 用于存储报警值

        # 检查超低和超高液位阈值
        if low_low_limit is not None and level < low_low_limit:

            exceeded = True
            alarm_value = low_low_limit
        elif high_high_limit is not None and level > high_high_limit:

            exceeded = True
            alarm_value = high_high_limit
        elif low_limit is not None and level < low_limit:

            exceeded = True
            alarm_value =low_limit
        elif high_limit is not None and level > high_limit:

            exceeded = True
            alarm_value=high_limit

        if exceeded:
            self.data[node] = alarm_value

        return exceeded, alarm_value
    def check_temperature(self, node, low_temp=None, high_temp=None, high_high_temp=None):
        if node not in self.data:
            return False, None  # 节点不在数据中，静默返回
        temperature = self.data.get(node, 0)
        exceeded = False
        alarm_value = None

        # 检查超高温阈值
        if high_high_temp is not None and temperature > high_high_temp:

            exceeded = True
            alarm_value = high_high_temp
        # 检查一般的高温阈值
        if high_temp is not None and temperature > high_temp:

            exceeded = True
            alarm_value = high_temp
        # 检查低温阈值
        if low_temp is not None and temperature < low_temp:

            exceeded = True
            alarm_value = low_temp

        if exceeded:
            self.data[node] = alarm_value

        return exceeded, alarm_value
MAX_PUMP_BOOST = 0.6
class Graph:
    def __init__(self, graph_dict):
        self.graph_dict = graph_dict

    def get_parents(self, node):

        parents = []
        for parent, children in self.graph_dict.items():
            if any(child[0] == node for child in children):
                parents.append(parent)
        return parents

    def get_children(self, node):
        if node in self.graph_dict:

            return [child[0] for child in self.graph_dict[node]]
        else:

            return []

def calculate_pressure_drop(current_node, parent_node, graph_view, current_temp, P_bar, current_flow_rate):
    # 从 graph_view 获取管道长度和直径
    for connection in graph_view.get(parent_node, []):
        if connection[0] == current_node:
            length_m = connection[1] / 1000.0
            d_m = connection[2] * 0.0254
            break
    else:
        # 如果找不到连接，返回零压降
        return 0

    # 计算压降
    pressure_drop = abs(calculate_P(current_temp, d_m, P_bar, length_m, current_flow_rate))
    # print(f"pressure_drop:{pressure_drop}")
    return pressure_drop


# def iterate_graph(data, filter_data, graph_view, current_temp, P_bar, current_flow_rate):

def reverse_iterate_and_update_pressure(node, filter_data, graph_view, current_temp, P_bar, current_flow_rate, updated_nodes):
    graph = Graph(graph_view)
    # 现在使用 graph 实例来调用 get_parents 方法
    parents = graph.get_parents(node)

    # 查找当前节点在 filter_data 中的压力值
    current_node_pressure = None
    for item in filter_data:
        if item['node'] == node:
            current_node_pressure = item['pressure']
            break

    # 如果当前节点的压力值不存在于 filter_data 中，则跳过更新
    if current_node_pressure is None:
        return

    for parent in parents:
        # 检查父节点是否已被更新
        if parent in updated_nodes:
            continue

        # 计算压降
        pressure_drop = calculate_pressure_drop(node, parent, graph_view, current_temp, P_bar, current_flow_rate)


        if "出口" in parent:
            delta_P_pump = abs(calculate_P_with_pump(current_temp, current_flow_rate))
            delta_P_pump = min(delta_P_pump, MAX_PUMP_BOOST)
            adjusted_pressure_drop = abs(pressure_drop - delta_P_pump)
        else:
            adjusted_pressure_drop = pressure_drop


        # 对父节点进行递归迭代
        reverse_iterate_and_update_pressure(parent, filter_data, graph_view, current_temp, P_bar, current_flow_rate, updated_nodes)
def forward_iterate_from_alert_node(alarm_nodes, filter_data, graph_view, current_temp, P_bar, current_flow_rate, updated_nodes):
    graph = Graph(graph_view)
    def should_update(node):
        return node not in updated_nodes and any(item['node'] == node for item in filter_data)

    def calculate_pressure_drop(current_temp, d_m, P_bar, length_m, current_flow_rate):
           # 这里应用您的压降公式
           return abs(calculate_P(current_temp, d_m, P_bar, length_m, current_flow_rate))

    visited_nodes = set()  # 用于跟踪已经处理过的节点

    for alarm_node in alarm_nodes:
        # 查找与报警节点相关的信息
        alarm_node_data = next((item for item in filter_data if item['node'] == alarm_node), None)
        if alarm_node_data is None:
            continue  # 如果未找到相关数据，则跳过当前报警节点

        parent_pressure = alarm_node_data['pressure']
        queue = [(alarm_node, parent_pressure)]

        while queue:
            node, current_pressure = queue.pop(0)  # 使用队列来保持正确的访问顺序

            if node in visited_nodes:
                continue  # 如果已处理过该节点，则跳过
            visited_nodes.add(node)  # 标记节点为已访问

            if node != alarm_node and should_update(node):  # 不更新报警节点本身的数据
                # 更新当前节点的数据
                for item in filter_data:
                    if item['node'] == node:
                        item['pressure'] = round(abs(current_pressure), 4)
                        break
            for target_node in graph.get_children(node):
                child_data = next((edge for edge in graph_view[node] if edge[0] == target_node), None)
                if child_data:
                    length_m, d_m = child_data[1], child_data[2]
                    length_m = length_m / 1000
                    d_m = d_m * 0.0254
                    pressure_drop = calculate_pressure_drop(current_temp, d_m, P_bar, length_m, current_flow_rate)
                    new_pressure = current_pressure - pressure_drop

                    queue.append((target_node, new_pressure))

