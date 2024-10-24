#graph-step1

import re
from collections import deque
# 定义子拓扑

def extract_valve_numbers(name):
    match = re.search(r'(\d{4})\D*$', name)
    return match.group(1) if match else None


def match_valve_data(data):
    graph_name = data.get("名称")
    structure_data = GRAPHS.get(graph_name)
    matched_valves = {}

def filter_structure_by_valve_opening(data):
    graph_name = data.get("名称")
    structure_data = GRAPHS.get(graph_name)
    matched_valves = match_valve_data(data)

    if not structure_data:
        print(f"No structure data found for graph: {graph_name}")
        return {}, set()  # 修改返回值为字典和集合的元组

    nodes_to_remove = set()

    for valve, opening in matched_valves.items():
        if opening < 80:
            nodes_to_remove.add(valve)
            queue = deque([valve])
            while queue:
                current_node = queue.popleft()
                for child in structure_data.get(current_node, []):
                    if child[0] not in nodes_to_remove:
                        if child[0].__contains__('D|'):
                            continue
                        nodes_to_remove.add(child[0])
                        queue.append(child[0])

    filtered_structure = {node: children for node, children in structure_data.items() if node not in nodes_to_remove}

    for node in filtered_structure:
        filtered_structure[node] = [child for child in filtered_structure[node] if child[0] not in nodes_to_remove]

    filtered_structure = remove_d_prefix(filtered_structure)
    # return filtered_structure, nodes_to_remove  # 返回filtered_structure和nodes_to_remove
    return nodes_to_remove
def remove_d_prefix(structure):
    # 创建一个新的字典，其中节点名称不包含 "D|"
    new_structure = {}
    for node, children in structure.items():
        # 移除节点名称中的 "D|"
        new_node = node.replace('D|', '')
        new_children = []
        for child in children:
            # 替换每个子元组中第一个元素（即节点名称）中的 "D|"
            new_child = (child[0].replace('D|', ''),) + child[1:]
            new_children.append(new_child)
        new_structure[new_node] = new_children
    return new_structure

def structure_data(data):
    graph_name = data.get("名称")
    structure_data = GRAPHS.get(graph_name)
    # print(f"{structure_data}:structurc_data")
    return structure_data