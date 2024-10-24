# receiveService.py--模版动态展示效果
from flask import Flask, request, jsonify
import graph as graph_module
from math_utils import *
from receive import *
import time, re, json
import threading
import yaml
import requests
from ModelPrediction import func
# from threshold import ThresholdChecker
from queue import Queue
import time

from iteration import *
app = Flask(__name__)
global_index = 0

# Flask 应用外部定义全局变量
current_graph_name = None
cached_results, current_states_cache, is_first_data, first_request_flag = None, None, True, True

@app.route('/receive_data', methods=['POST'])
def receive_data():
    global cached_results, current_states_cache, is_first_data, first_request_flag, current_graph_name,global_index


    if first_request_flag:
        # print("程序正在运行.......")
        first_request_flag = False
    output_queue = Queue()
    try:
        data = request.json
        if not data:
            # print("No data provided")  # 输出到终端
            return jsonify({"error": "No data provided"}), 400
        graph_name = data.get("名称")
        if not graph_name:
            # print("No graph name provided")
            return jsonify({"error": "No graph name provided"}), 400
        # 检查线路名称是否与当前处理的线路名称一致
        if current_graph_name is not None and current_graph_name != graph_name:
            # print("Route Name Mismatch Error")
            return jsonify({"error": "Route Name Mismatch Error"}), 400

        current_graph_name = graph_name
        matched_valves = graph_module.match_valve_data(data)
        nodes_to_remove = graph_module.filter_structure_by_valve_opening(data)
        structure_data_0=graph_module.structure_data(data)

        structure_data = graph_module.remove_d_prefix(structure_data_0)
        results = calculate_results(data, structure_data)



        for result in results:
            if result['node'] in nodes_to_remove:
                result.update({'pressure': 0, 'temperature': 0})
        # print(results)

        # 初始过滤和格式化 filtered_results
        filtered_results = []
        for res in results:
            if "K" not in res["node"]:
                formatted_res = {
                    "node": res["node"],
                    "pressure": abs(round(res["pressure"], 4)),
                    "temperature": abs(round(res["temperature"], 4))
                }
                filtered_results.append(formatted_res)

        iterate_graph_output = iterate_graph(data, filtered_results, structure_data)
        iterate_graph_output_list = iterate_graph_output[0]

        # 检查是否存在报警节点
        alarm_node_exists = iterate_graph_output[1]
        # 如果存在报警节点，则更新 filtered_results
        if alarm_node_exists:
            filtered_results = iterate_graph_output_list



        model_result = func(f"{graph_name}: " + str(filtered_results))
        # print(f"model_result:{model_result}")
        input_str = model_result
        # 使用正则表达式匹配第一个冒号及其之前的部分
        result_str = re.sub(r'^[^:]*:', '', input_str, 1).lstrip()
        # 将单引号替换为双引号
        result_0 = result_str.replace("'", "\"")
        # 将字符串转换为列表
        result_list_start = json.loads(result_0)
        # print(f"result_list_start:{result_list_start}")
        result_list_end = calculate_and_return_min_error_nodes(filtered_results, data, result_list_start)

        result_list = []

        # 现在的 filtered_result_list 包含了根据条件筛选后的结果
        print(f"{graph_name}:{result_list}")

        # 根据 is_first_data 确定是否打印结果
        if is_first_data:

            is_first_data = False  # 更新标志

        tempOutput = []

        animation_thread = None
        # 如果有缓存的结果，启动动画展示线程
        if cached_results is not None:
            lock = threading.Lock()
            with lock:
                animate_results(cached_results, result_list, output_queue)
        # 从队列中收集输出
        animation_outputs = []
        while not output_queue.empty():
            animation_outputs.append(output_queue.get())

        # 更新缓存结果
        cached_results = result_list

        # 将动画输出添加到响应中
        # 新的 JSON 结构
        new_json_structure = []
        for output in animation_outputs:
            global_index += 1  # 增加全局索引
            new_json_structure.append({
                "index": global_index,
                "data": output  # 假设 output 已经是所需格式的字典
            })

        # return jsonify(animation_outputs), 200
        return jsonify(new_json_structure), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except KeyError as e:
        return jsonify({"error": f"Key error: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500


def calculate_and_return_min_error_nodes(nodes_1, nodes_2, nodes_3):
    def process_nodes(nodes):
        processed_data = {}
        for node_item in nodes:
            node_key = node_item['node']
            pressure_error = float('inf')
            temperature_error = float('inf')


            processed_data[node_key] = {'pressure_error': pressure_error, 'temperature_error': temperature_error}

        return processed_data

    data_1 = process_nodes(nodes_1)
    data_3 = process_nodes(nodes_3)

    result = []
    for node in nodes_3:
        node_key = node['node']
        pressure_error_1 = data_1[node_key]['pressure_error']
        temperature_error_1 = data_1[node_key]['temperature_error']
        pressure_error_3 = data_3[node_key]['pressure_error']
        temperature_error_3 = data_3[node_key]['temperature_error']


        if pressure_error_1 + temperature_error_1 <= pressure_error_3 + temperature_error_3:
            result.append(nodes_1[[node_item['node'] for node_item in nodes_1].index(node_key)])
        else:
            result.append(node)
    return result

def calculate_differences(initial_results, final_results):
    try:
        diffs = {}
        final_nodes = {node['node']: node for node in final_results}


        return diffs
    except KeyError as e:
        # print(f"Key error: {e}")
        return {}
    except Exception as e:
        # print(f"An error occurred: {e}")
        return {}


def animate_results(initial_results, final_results, output_queue, steps=15):
    global current_states_cache, global_index


def read_config():
    with open("./config.yml", "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
        return config

def register_nacos(nacos_config):
    # 配置 Nacos 服务注册中心的地址和端口
    url = nacos_config['url']
    requests.post(url)

def nacos_beat(nacos_config):
    url = nacos_config['beat']
    result = requests.put(url)
    threading.Timer(5, nacos_beat, [nacos_config]).start()

def unregister_nacos(nacos_config):
    # 配置 Nacos 服务注册中心的地址和端口
    url = nacos_config['url']
    requests.delete(url)

def run_app(server_config):
    app.run(host=server_config['ip'], port=server_config['port'])

if __name__ == '__main__':
    config = read_config()
    # unregister_nacos(config['nacos']);
    try:
        register_nacos(config['nacos'])
        threading.Timer(0, nacos_beat, [config['nacos']]).start()
    except Exception as e:
        print(e)
    run_app(config['server'])
    # run(host='0.0.0.0',port=5000)
