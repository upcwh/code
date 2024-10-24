

def calculate_differences(initial_results, final_results):
    try:
        diffs = {}
        final_nodes = {node['node']: node for node in final_results}

        for node in initial_results:
            node_name = node['node']
            final_node = final_nodes.get(node_name)

            if final_node:
                pressure_diff = final_node['pressure'] - node['pressure']
                temperature_diff = final_node['temperature'] - node['temperature']
            else:
                pressure_diff = 0
                temperature_diff = 0

            diffs[node_name] = {'pressure': pressure_diff, 'temperature': temperature_diff}

        return diffs
    except KeyError as e:
        print(f"Key error: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def animate_results(initial_results, final_results, graph_name, steps=15, sleep_duration=0.1):
    global current_states_cache

    try:
        diffs = calculate_differences(initial_results, final_results)

        # 如果缓存为空，则使用初始结果初始化它
        if current_states_cache is None:
            current_states_cache = {node['node']: node for node in initial_results}

        # 使用缓存的状态来执行动画
        for step in range(steps):
            # 遍历缓存中的节点以更新它们的状态
            for node_name, current_state in current_states_cache.items():
                diff = diffs.get(node_name)

                # 如果节点在最终结果中存在，则更新其状态
                if diff:
                    current_state['pressure'] = round(current_state['pressure'] + diff['pressure'] / steps, 4)
                    current_state['temperature'] = round(current_state['temperature'] + diff['temperature'] / steps, 4)
                # 如果节点在最终结果中不存在，保持当前状态不变

            # 输出当前所有节点的状态
            print(f"{graph_name}: {list(current_states_cache.values())}")
            time.sleep(sleep_duration)

        # 动画结束后，使用最终结果更新缓存
        for node in final_results:
            node_name = node['node']
            if node_name in current_states_cache:
                current_states_cache[node_name].update(node)

    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")