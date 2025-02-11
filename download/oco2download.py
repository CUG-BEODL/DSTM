import requests
import os

# 打开URL列表文件
with open(r"H:\greenhouse\XCO2\OCO2\subset_OCO2_L2_Lite_FP_11r_20230911_075257_.txt", "r") as file:
    urls = file.readlines()

# 下载文件
for url in urls:
    url = url.strip()  # 去除URL两端的空白字符
    name = url.split("/")[-1]  # 提取URL中的文件名
    start_index = name.find("oco2_LtCO2_") + len("oco2_LtCO2_")
    end_index = name.find("_B11014Ar")
    if start_index != -1 and end_index != -1:
        filename = name[start_index:end_index]
        print(filename)
    else:
        print("未找到匹配的内容")
        break

    file = os.path.join(r'H:\greenhouse\XCO2\OCO2', filename)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url,headers=headers)
    with open(file, "wb") as file:
        file.write(response.content)
