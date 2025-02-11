import datetime

import requests

# 日期范围
start_date = "2015-01-01"
end_date = "2020-12-31"

# 生成URL并下载数据
base_url = "https://gml.noaa.gov/aftp/products/carbontracker/co2/molefractions/xCO2_1330LST/CT2022.xCO2_1330_glb3x2_{}.nc"
current_date = start_date
while current_date <= end_date:
    url = base_url.format(current_date)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.76"}  # 设置请求头部
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        file_name = r'H:\greenhouse\CT\global\CT2022.xCO2_1330_glb3x2_{}.nc'.format(current_date)  # 文件名为日期.nc
        with open(file_name, "wb") as file:
            file.write(response.content)
        print(f"下载成功: {url}")
    else:
        print(f"下载失败: {url}")

    # 日期加1天
    current_date = (datetime.datetime.strptime(current_date, "%Y-%m-%d") + datetime.timedelta(days=1)).strftime(
        "%Y-%m-%d")