import cdsapi
import requests
import os

from tqdm import tqdm


def Downloader(url, save_folder, filename):
    # 创建保存文件夹
    os.makedirs(save_folder, exist_ok=True)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.69'
    }

    response = requests.get(url, headers=headers)

    total_size = int(response.headers.get('content-length', 0))

    if response.status_code == 200:
        file_path = os.path.join(save_folder, filename)
        with open(file_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=4096 * 1024):  # 4MB
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        print(f'{filename}下载完成，保存路径：{file_path}')
    else:
        print('下载失败。')


if __name__ == '__main__':
    c = cdsapi.Client()  # 创建用户

    # 数据信息字典
    dic = {
        'product_type': 'reanalysis',  # 产品类型
        'format': 'netcdf',  # 数据格式
        'variable': '10m_v_component_of_wind',  # 变量名称 10m_u_component_of_wind
        'area': [60, 70, 15, 140],
        'year': '',  # 年，设为空
        'month': [  # 月
            '01', '02',
            # '03',
            # '04', '05', '06',
            # '07', '08', '09',
            # '10', '11', '12',
        ],
        'day': [  # 日
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16',
            '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31'
        ],
        'time': [  # 小时
            '13:00'
        ]
    }
    save_folder = r'C:\Users\Administrator\Downloads\v10'

    for y in range(2021, 2022):  # 遍历年
        dic['year'] = str(y)
        filename = str(y)+'.nc'
        r = c.retrieve('reanalysis-era5-single-levels', dic, )  # 文件下载器
        url = r.location  # 获取文件下载地址
        print(y)
        print(url)
        # break
        # Downloader(url, save_folder, filename)
