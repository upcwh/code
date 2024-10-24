#math_utils.py-step3
from calculations import *
from collections import deque, defaultdict
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
def calculate_results(data, graph):
    line_name = data.get('名称', '')
    start_node = None  # 初始化start_node
    Q = 0  # 初始化Q

    # 如果start_node为空，则尝试从图结构中获取初始节点
    if start_node is None:
        start_node = next(iter(graph)) if graph else None


    results = []


    # 返回计算结果
    return results

def handle_confluence_and_split_nodes(graph_result):
    if not isinstance(graph_result, dict):
        raise TypeError("Expected a dictionary for 'graph_result', but received {}".format(type(graph_result)))

    confluence_nodes = set()
    split_nodes = set()

    for node_name, children in graph_result.items():
        if len(children) > 1:
            split_nodes.add(node_name)

    in_degrees = {node: 0 for node in graph_result.keys()}

    # 计算每个节点的入度
    for children in graph_result.values():
        for child_node, *_ in children:
            in_degrees[child_node] = in_degrees.get(child_node, 0) + 1

    for node_name, in_degree in in_degrees.items():
        if in_degree > 1:
            confluence_nodes.add(node_name)

    return list(split_nodes), list(confluence_nodes)

def traverse_graph(graph, start_node, start_pressure, P_bar, start_temperature, Q, results, liquid_levels, line_name=None, temperature_data=None):
    split_nodes, confluence_nodes = handle_confluence_and_split_nodes(graph)
    queue = deque([(start_node, start_pressure, start_temperature, None, 0, 0, Q)])
    visited = set()
    flow_rates = defaultdict(float)

    while queue:
        current_node, current_pressure, current_temp, previous_node, previous_pressure_drop, previous_temp_drop, current_flow_rate = queue.popleft()

        # 当前节点是汇合节点时，累加流量；否则直接赋值
        if current_node in confluence_nodes:
            flow_rates[current_node] += current_flow_rate
        else:
            flow_rates[current_node] = current_flow_rate

        # 避免重复访问已经访问过的节点
        if current_node in visited:
            continue
        visited.add(current_node)


        children = graph.get(current_node, [])
        if current_node in split_nodes:
            current_flow_rate = flow_rates[current_node] / len(children)

        for next_node, length_mm, d_inches, height in children:
            length_m = length_mm / 1000
            d_m = d_inches * 0.0254

            pressure_drop = abs(calculate_P(current_temp, d_m, P_bar, length_m, current_flow_rate))
            next_temperature = abs(solve_tL(current_temp, d_m, length_m, current_flow_rate))
