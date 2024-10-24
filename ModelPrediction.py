# -*- coding: utf-8 -*-

import re
import time

import torch
from TCN_Model import TCN


def func(name):
    graph_name = name

    # 程序开始时间
    start_time = time.time()

    # 提取线路名称，即第一个冒号前的部分
    parts = graph_name.split(":")
    first_part = parts[0]

    # 根据名称判断使用哪个模型
    path = './TCN/'
    if first_part == '1':
        path += 'model20231113_1.pth'
    elif first_part == '2':
        path += 'model20231113_2.pth'
    elif first_part == '3':
        path += 'model20231113_3.pth'
    else:
        path += 'model20231113_4.pth'

    # 加载模型
    model = TCN()
    model.load_state_dict(torch.load(path), strict=False)
    model.eval()

    # 匹配浮点数
    float_pattern = r":\s*([-+]?\d*\.\d+|\d+)"
    float_numbers = re.findall(float_pattern, graph_name)

    # 将字符串形式的浮点数转换为真正的浮点数
    float_numbers = [float(number) for number in float_numbers]

    # 打印提取的浮点数
    # print("提取的浮点数:", float_numbers)

    # 处理数据，得到可以训练的数据形式
    data_dic = {}
    cnt = 0
    for i in float_numbers:
        if cnt not in data_dic:
            data_dic[cnt] = {}
            data_dic[cnt] = float_numbers[cnt]
        cnt += 1
    # print(data_dic)

    dic1 = {}
    cnt = 0
    for key, value in data_dic.items():
        if cnt not in dic1:
            dic1[cnt] = {}
            for i in range(0, 1):
                dic1[cnt][i] = value
        cnt += 1
    # print(dic1)

    ids1 = dic1.keys()
    column = set(col for values in dic1.values() for col in values)
    X = [[dic1.get(id, {}).get(col, 0) for col in column] for id in ids1]

    X = torch.Tensor(X)
    X = X.unsqueeze(-1).permute(0, 2, 1)

    dic = {}
    cnt = 0
    for i in float_numbers:
        if cnt not in dic:
            dic[cnt] = {}
            dic[cnt][0] = float_numbers[cnt]
        cnt += 1

    ids1 = dic.keys()
    column = set(col for values in dic.values() for col in values)
    X = [[dic.get(id, {}).get(col, 0) for col in column] for id in ids1]

    X = torch.Tensor(X)
    X = X.unsqueeze(-1).permute(0, 2, 1)
    with torch.no_grad():
        data = model(X)

    dic0 = {}
    cnt = 0
    for i in range(0, data.shape[0]):
        if float_numbers[cnt] not in dic0:
            dic0[float_numbers[cnt]] = round(data[cnt][0].item(), 4)
        cnt += 1

    # 模型预测的数值替换掉原数值
    def replace_match(match):
        value_str = match.group(0)
        value = float(value_str)
        if value in dic0:
            return str(dic0[value])
        else:
            return str(value_str)

    pat = r"[-+]?\d*\.\d+|\d+"
    filtered_results = re.sub(pat, replace_match, graph_name)
    # 程序结束时间
    end_time = time.time()
    return filtered_results
