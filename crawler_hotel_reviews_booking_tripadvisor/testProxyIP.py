# coding: utf8

import requests

# myself IP address
localip = "176.72.191.244"

# 待清洗代理IP数据池
proxies = [
    {"http": "115.215.49.143:8118" },
    {"http": "58.240.205.126:8118" },
    {"http": "110.73.9.10:8123"},
    ]

# 有效代理IP池
proxypool = []

# 清洗代理IP，去除无效的代理IP
for index in range(len(proxies)):
    print(proxies[index])
    try:
        result = requests.get('https://www.google.fi', proxies=proxies[index])
    except Exception as e:
        continue
    # 与本机IP对比，相等则说明没用到代理IP，为无效项
    if result.content != localip:
        proxypool.append(proxies[index])
print("useful IP:")
print(proxypool)
