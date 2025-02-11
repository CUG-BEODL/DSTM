import re
from tqdm import tqdm
import numpy as np
from osgeo import gdal
import os
from datetime import datetime, timedelta


# 输入图像
def read_tif(path):
    dataset = gdal.Open(path)

    cols = dataset.RasterXSize  # 图像长度
    rows = (dataset.RasterYSize)  # 图像宽度
    im_proj = (dataset.GetProjection())  # 读取投影
    im_Geotrans = (dataset.GetGeoTransform())  # 读取仿射变换
    im_data = dataset.ReadAsArray(0, 0, cols, rows)  # 转为numpy格式
    # im_data[im_data > 0] = 1 #除0以外都等于1
    del dataset
    return im_proj, im_Geotrans, im_data


# 将tif保存
def write_tif(filename, im_proj, im_geotrans, im_data):
    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset


def isTIF(filename):
    return filename.endswith('.tif')


# t: year_day
times = ['2015_1', '2015_17', '2015_33', '2015_49', '2015_65', '2015_81', '2015_97', '2015_113', '2015_129', '2015_145',
         '2015_161', '2015_177', '2015_193', '2015_209', '2015_225', '2015_241', '2015_257', '2015_273', '2015_289',
         '2015_305', '2015_321', '2015_337', '2015_353', '2016_1', '2016_17', '2016_33', '2016_49', '2016_65',
         '2016_81', '2016_97', '2016_113', '2016_129', '2016_145', '2016_161', '2016_177', '2016_193', '2016_209',
         '2016_225', '2016_241', '2016_257', '2016_273', '2016_289', '2016_305', '2016_321', '2016_337', '2016_353',
         '2017_1', '2017_17', '2017_33', '2017_49', '2017_65', '2017_81', '2017_97', '2017_113', '2017_129', '2017_145',
         '2017_161', '2017_177', '2017_193', '2017_209', '2017_225', '2017_241', '2017_257', '2017_273', '2017_289',
         '2017_305', '2017_321', '2017_337', '2017_353', '2018_1', '2018_17', '2018_33', '2018_49', '2018_65',
         '2018_81', '2018_97', '2018_113', '2018_129', '2018_145', '2018_161', '2018_177', '2018_193', '2018_209',
         '2018_225', '2018_241', '2018_257', '2018_273', '2018_289', '2018_305', '2018_321', '2018_337', '2018_353',
         '2019_1', '2019_17', '2019_33', '2019_49', '2019_65', '2019_81', '2019_97', '2019_113', '2019_129', '2019_145',
         '2019_161', '2019_177', '2019_193', '2019_209', '2019_225', '2019_241', '2019_257', '2019_273', '2019_289',
         '2019_305', '2019_321', '2019_337', '2019_353', '2020_1', '2020_17', '2020_33', '2020_49', '2020_65',
         '2020_81', '2020_97', '2020_113', '2020_129', '2020_145', '2020_161', '2020_177', '2020_193', '2020_209',
         '2020_225', '2020_241', '2020_257', '2020_273', '2020_289', '2020_305', '2020_321', '2020_337', '2020_353',
         '2021_1', '2021_17', '2021_33', '2021_49']


def get16index(t):
    def judge(t1, t2):
        year1, day1 = map(int, t1.split("_"))
        year2, day2 = map(int, t2.split("_"))
        if year1 == year2:
            return day1 > day2
        else:
            return year1 > year2

    left = 0
    right = len(times) - 1
    while left <= right:
        mid = (left + right) // 2
        if times[mid] == t:
            return mid
        elif judge(t, times[mid]):
            left = mid + 1
        else:
            right = mid - 1

    return left - 1


def getMonth(t):
    date_str = t[:4] + '-' + t[5:]
    date_obj = datetime.strptime(date_str, '%Y-%j')
    year = date_obj.year
    month = date_obj.month
    return str(year % 100) + str(month).rjust(2, '0')


def getYear(t):
    return int(t.split('_')[0]) - 2015


def getDateCode(t):
    year, day = map(int, t.split('_'))
    return year * 1000 + day


def get_16dates(start_date_str, num_days=15):
    dates = []
    start_date = datetime.strptime(start_date_str, '%Y_%j')

    for i in range(num_days, -1, -1):
        date_str = start_date.strftime('%Y_%j')
        dates.append(date_str)
        start_date -= timedelta(days=1)
    dates = [re.sub(r'(^|_)0+', r'\1', date) for date in sorted(dates)]
    return dates


def extract_patch(t):
    # _, _, xco2 = read_tif(os.path.join(xco2_path, t + '.tif'))

    era5s = []
    dates16 = get_16dates(t)
    for era5_path in era5_paths:
        for date in dates16:
            _, _, era5 = read_tif(os.path.join(era5_path, date + '.tif'))
            era5s.append(era5)

    month = getMonth(t)
    _, _, emission = read_tif(os.path.join(emission_path, 'odiac2022_1km_excl_intl_{}.tif'.format(month)))

    # 获取年份
    yearIDX = getYear(t)
    # print(yearIDX + 2015)

    # 获取16id
    idx16 = get16index(t)
    # print(idx16)

    points = []
    codes = []

    for i in tqdm(range(mask.shape[0])):
        if i <= 1 or i >= mask.shape[0] - 2:
            continue
        for j in range(mask.shape[1]):
            # if np.isnan(xco2[i, j]) or mask[i, j] < 0.5:
            if mask[i, j] < 0.5:
                continue
            # 判断是否在边界上
            if j <= 1 or j >= mask.shape[1] - 2:
                continue

            point = np.empty((0, 5, 5), dtype=np.float32)
            for era5 in era5s:
                # 获取对应区域
                sub_matrix = era5[i - 2:i + 3, j - 2:j + 3]
                point = np.concatenate((point, [sub_matrix]))

            point = np.concatenate((point, srtm[:, i - 2:i + 3, j - 2:j + 3]))

            point = np.concatenate((point, population[yearIDX:yearIDX + 1, i - 2:i + 3, j - 2:j + 3]))
            point = np.concatenate((point, landuse[yearIDX:yearIDX + 1, i - 2:i + 3, j - 2:j + 3]))
            point = np.concatenate((point, ndvi[idx16:idx16 + 1, i - 2:i + 3, j - 2:j + 3]))
            point = np.concatenate((point, [emission[i - 2:i + 3, j - 2:j + 3]]))

            # code = np.array([getDateCode(t), i, j, xco2[i, j]], dtype=np.float32)
            code = np.array([getDateCode(t), i, j, 0], dtype=np.float32)
            codes.append([code])
            points.append([point])

    points = np.concatenate((points))
    codes = np.concatenate((codes))
    return points, codes


def extract_patch_site(t, x, y):
    era5s = []
    dates16 = get_16dates(t)
    for era5_path in era5_paths:
        for date in dates16:
            _, _, era5 = read_tif(os.path.join(era5_path, date + '.tif'))
            era5s.append(era5)

    month = getMonth(t)
    _, _, emission = read_tif(os.path.join(emission_path, 'odiac2022_1km_excl_intl_{}.tif'.format(month)))

    # 获取年份
    yearIDX = getYear(t)
    # print(yearIDX + 2015)

    # 获取16id
    idx16 = get16index(t)
    # print(idx16)

    points = np.empty((0, 87, 5, 5), dtype=np.float32)
    codes = np.empty((0, 4), dtype=np.float32)
    # for i in range(y - 2, y + 3):
    #     for j in range(x - 2, x + 3):
    for i in range(y, y + 1):
        for j in range(x, x + 1):

            if mask[i, j] < 0.5:
                continue
            # 判断是否在边界上
            if i <= 1 or i >= mask.shape[0] - 2 or j <= 1 or j >= mask.shape[1] - 2:
                continue

            point = np.empty((0, 5, 5), dtype=np.float32)
            for era5 in era5s:
                # 获取对应区域
                sub_matrix = era5[i - 2:i + 3, j - 2:j + 3]
                point = np.concatenate((point, [sub_matrix]))

            point = np.concatenate((point, srtm[:, i - 2:i + 3, j - 2:j + 3]))

            point = np.concatenate((point, population[yearIDX:yearIDX + 1, i - 2:i + 3, j - 2:j + 3]))
            point = np.concatenate((point, landuse[yearIDX:yearIDX + 1, i - 2:i + 3, j - 2:j + 3]))
            point = np.concatenate((point, ndvi[idx16:idx16 + 1, i - 2:i + 3, j - 2:j + 3]))
            point = np.concatenate((point, [emission[i - 2:i + 3, j - 2:j + 3]]))

            code = np.array([getDateCode(t), i, j, 0], dtype=np.float32)
            codes = np.concatenate((codes, [code]), axis=0)
            points = np.concatenate((points, [point]), axis=0)

    return points, codes


def extract_patch_without_time(t):
    _, _, xco2 = read_tif(os.path.join(xco2_path, t + '.tif'))

    era5s = []

    for era5_path in era5_paths:
        _, _, era5 = read_tif(os.path.join(era5_path, t + '.tif'))
        era5s.append(era5)

    month = getMonth(t)
    _, _, emission = read_tif(os.path.join(emission_path, 'odiac2022_1km_excl_intl_{}.tif'.format(month)))

    # 获取年份
    yearIDX = getYear(t)
    # print(yearIDX + 2015)

    # 获取16id
    idx16 = get16index(t)
    # print(idx16)

    points = np.empty((0, 12, 5, 5), dtype=np.float32)
    codes = np.empty((0, 4), dtype=np.float32)
    for i in range(xco2.shape[0]):
        for j in range(xco2.shape[1]):
            if np.isnan(xco2[i, j]) or mask[i, j] < 0.5:
                continue
            # 判断是否在边界上
            if i <= 1 or i >= xco2.shape[0] - 2 or j <= 1 or j >= xco2.shape[1] - 2:
                continue

            point = np.empty((0, 5, 5), dtype=np.float32)
            for era5 in era5s:
                # 获取对应区域
                sub_matrix = era5[i - 2:i + 3, j - 2:j + 3]
                point = np.concatenate((point, [sub_matrix]))

            point = np.concatenate((point, srtm[:, i - 2:i + 3, j - 2:j + 3]))

            point = np.concatenate((point, population[yearIDX:yearIDX + 1, i - 2:i + 3, j - 2:j + 3]))
            point = np.concatenate((point, landuse[yearIDX:yearIDX + 1, i - 2:i + 3, j - 2:j + 3]))
            point = np.concatenate((point, ndvi[idx16:idx16 + 1, i - 2:i + 3, j - 2:j + 3]))
            point = np.concatenate((point, [emission[i - 2:i + 3, j - 2:j + 3]]))

            code = np.array([getDateCode(t), i, j, xco2[i, j]], dtype=np.float32)
            codes = np.concatenate((codes, [code]), axis=0)
            points = np.concatenate((points, [point]), axis=0)

    return points, codes


def extract16(t, file):
    _, _, xco2 = read_tif(os.path.join(xco2_path, t + '.tif'))

    era5s = []
    dates16 = get_16dates(t)
    for era5_path in era5_paths:
        for date in dates16:
            _, _, era5 = read_tif(os.path.join(era5_path, date + '.tif'))
            era5s.append(era5)

    month = getMonth(t)
    _, _, emission = read_tif(os.path.join(emission_path, 'odiac2022_1km_excl_intl_{}.tif'.format(month)))

    # 获取年份
    yearIDX = getYear(t)
    # print(yearIDX + 2015)

    # 获取16id
    idx16 = get16index(t)
    # print(idx16)

    points = np.empty((0, 91), dtype=np.float32)
    for i in range(xco2.shape[0]):
        for j in range(xco2.shape[1]):
            if np.isnan(xco2[i, j]) or mask[i, j] < 0.5:
                continue
            point = np.empty((0,), dtype=np.float32)
            for era5 in era5s:
                point = np.concatenate((point, era5[i:i + 1, j]))

            # point = srtm[:, i, j]
            point = np.concatenate((point, srtm[:, i, j]))

            point = np.concatenate((point, population[yearIDX:yearIDX + 1, i, j]))
            point = np.concatenate((point, landuse[yearIDX:yearIDX + 1, i, j]))
            point = np.concatenate((point, ndvi[idx16:idx16 + 1, i, j]))

            code = np.array([emission[i, j], getDateCode(t), i, j, xco2[i, j]], dtype=np.float32)
            # code = np.array([emission[i, j], getDateCode(t), i, j, 0], dtype=np.float32)
            point = np.concatenate((point, code))

            points = np.concatenate((points, [point]), axis=0)
    return points


def extract(t, file):
    _, _, xco2 = read_tif(os.path.join(xco2_path, t + '.tif'))
    # xco2 = xco2[0]
    # print(t)
    # print(getDateCode(t))

    # _, _, maskXCO2 = read_tif(os.path.join(r'H:\greenhouse\XCO2\OCO2\tif', t + '.tif'))
    # maskXCO2= maskXCO2[0]
    era5s = []
    for era5_path in era5_paths:
        # print(t)
        _, _, era5 = read_tif(os.path.join(era5_path, t + '.tif'))
        era5s.append(era5)

    month = getMonth(t)
    _, _, emission = read_tif(os.path.join(emission_path, 'odiac2022_1km_excl_intl_{}.tif'.format(month)))

    # 获取年份
    yearIDX = getYear(t)
    # print(yearIDX + 2015)

    # 获取16id
    idx16 = get16index(t)
    # print(idx16)

    points = np.empty((0, 91), dtype=np.float32)
    for i in range(xco2.shape[0]):
        for j in range(xco2.shape[1]):
            if np.isnan(xco2[i, j]) or mask[i, j] < 0.5:
                continue

            point = srtm[:, i, j]

            point = np.concatenate((point, population[yearIDX:yearIDX + 1, i, j]))
            point = np.concatenate((point, landuse[yearIDX:yearIDX + 1, i, j]))
            point = np.concatenate((point, ndvi[idx16:idx16 + 1, i, j]))

            for era5 in era5s:
                point = np.concatenate((point, era5[i:i + 1, j]))
            code = np.array([emission[i, j], getDateCode(t), i, j, xco2[i, j]], dtype=np.float32)
            # code = np.array([emission[i, j], getDateCode(t), i, j, 0], dtype=np.float32)
            point = np.concatenate((point, code))

            points = np.concatenate((points, [point]), axis=0)
    return points


def site_extract(t, x, y):
    era5s = []
    for era5_path in era5_paths:
        _, _, era5 = read_tif(os.path.join(era5_path, t + '.tif'))
        era5s.append(era5)

    month = getMonth(t)
    _, _, emission = read_tif(os.path.join(emission_path, 'odiac2022_1km_excl_intl_{}.tif'.format(month)))

    points = np.empty((0, 91), dtype=np.float32)
    for i in range(y - 2, y + 3):
        for j in range(x - 2, x + 3):
            # if np.isnan(xco2[i, j]) or mask[i, j] < 0.5:
            if mask[i, j] < 0.5:
                continue

            point = srtm[:, i, j]
            # 获取年份
            yearIDX = getYear(t)
            point = np.concatenate((point, population[yearIDX:yearIDX + 1, i, j]))
            point = np.concatenate((point, landuse[yearIDX:yearIDX + 1, i, j]))

            # 获取16id
            idx16 = get16index(t)
            point = np.concatenate((point, ndvi[idx16:idx16 + 1, i, j]))

            for era5 in era5s:
                point = np.concatenate((point, era5[i:i + 1, j]))

            # code = np.array([getDateCode(t), i, j, xco2[i, j]], dtype=np.float32)
            code = np.array([emission[i, j], getDateCode(t), i, j, 0], dtype=np.float32)
            point = np.concatenate((point, code))

            points = np.concatenate((points, [point]), axis=0)
    return points


if __name__ == '__main__':

    # mask 陆地区域
    mask_path = r'H:\greenhouse\mask\MASK.tif'
    _, _, mask = read_tif(mask_path)
    # 常量数据
    srtm_path = r'H:\greenhouse\srtm\crop.tif'
    _, _, srtm = read_tif(srtm_path)

    # 年度数据
    population_path = r'H:\greenhouse\population\worldscan\crop.tif'
    landuse_path = r'H:\greenhouse\landuse\crop.tif'
    _, _, population = read_tif(population_path)
    _, _, landuse = read_tif(landuse_path)

    # 每月数据
    emission_path = r'H:\greenhouse\CO2emission\crop'

    # 16天数据
    ndvi_path = r'H:\greenhouse\NDVI\crop.tif'
    _, _, ndvi = read_tif(ndvi_path)

    # 每日数据
    era5_paths = [r'H:\greenhouse\ERA5\d2m\crop',
                  r'H:\greenhouse\ERA5\t2m\crop',
                  r'H:\greenhouse\ERA5\sp\crop',
                  r'H:\greenhouse\ERA5\u10\crop',
                  r'H:\greenhouse\ERA5\v10\crop']
    xco2_path = r'H:\greenhouse\assimilation'

    points = np.empty((0, 87, 5, 5), dtype=np.float32)
    codes = np.empty((0, 4), dtype=np.float32)
    positions = np.empty((0, 2), dtype=np.int8)

    year = 2021
    days = [f'{year}_{day}.tif' for day in range(1, 60)]
    print(days)
    for file in days:
        print(file)

        paras = file.split('_')
        year = paras[0]
        day = paras[1].replace('.tif', '')
        if year == '2015' and int(day) < 16:
            continue
        point, code = extract_patch(file.replace('.tif', ''))

        print(point.shape)
        print(code.shape)
        np.save(r'chutu/patch_{}.npy'.format(file.replace('.tif', '')), point)
        np.save(r'chutu/code_{}.npy'.format(file.replace('.tif', '')), code)
    exit()

    array = list(os.listdir(r'H:\greenhouse\ERA5\d2m'))[:-2]
    # 自定义排序函数
    def custom_sort(string):
        parts = string.split('_')
        return (int(parts[0]), int(parts[1].split('.')[0]))


    # 使用自定义排序函数对数组进行排序
    sorted_array = sorted(array, key=custom_sort)
    print(sorted_array)
    print(len(sorted_array))
    exit()
    sites = ['Beijing', 'Guangzhou', 'Wuhan', 'Nanjing', 'Harbin', 'Lhasa', 'Urumqi']
    for site in sites:
        # points = np.empty((0, 16), dtype=np.float32)
        points = np.empty((0, 87, 5, 5), dtype=np.float32)
        codes = np.empty((0, 4), dtype=np.float32)
        for file in tqdm(sorted_array):

            if not isTIF(file):
                continue

            paras = file.split('_')
            year = paras[0]
            day = paras[1].replace('.tif', '')
            if year == '2015' and int(day) < 16:
                continue

            if site == 'Beijing':
                point, code = extract_patch_site(file.replace('.tif', ''), 440, 150)  # XH
            if site == 'Guangzhou':
                point, code = extract_patch_site(file.replace('.tif', ''), 409, 317)  # HF
            if site == 'Wuhan':
                point, code = extract_patch_site(file.replace('.tif', ''), 420, 243)  # WLG
            if site == 'Nanjing':
                point, code = extract_patch_site(file.replace('.tif', ''), 465, 228)  # LLN
            if site == 'Harbin':
                point, code = extract_patch_site(file.replace('.tif', ''), 542, 91)  # LLN
            if site == 'Lhasa':
                point, code = extract_patch_site(file.replace('.tif', ''), 188, 252)  # LLN
            if site == 'Urumqi':
                point, code = extract_patch_site(file.replace('.tif', ''), 153, 111)  # LLN

            codes = np.concatenate((codes, code), axis=0)
            points = np.concatenate((points, point), axis=0)

        print(site)
        print(points.shape)
        print(codes.shape)
        np.save(r'npy/patch_city_{}.npy'.format(site), points)
        np.save(r'npy/code_city_{}.npy'.format(site), codes)

    exit()
    # 提取站点数据
    array = list(os.listdir(r'H:\greenhouse\ERA5\d2m'))[:-2]


    # 自定义排序函数
    def custom_sort(string):
        parts = string.split('_')
        return (int(parts[0]), int(parts[1].split('.')[0]))


    # 使用自定义排序函数对数组进行排序
    sorted_array = sorted(array, key=custom_sort)
    print(sorted_array)

    sites = ['XH', 'HF', 'WLG', 'LLN']
    for site in sites:
        # points = np.empty((0, 16), dtype=np.float32)
        points = np.empty((0, 87, 5, 5), dtype=np.float32)
        codes = np.empty((0, 4), dtype=np.float32)
        for file in tqdm(sorted_array):

            if not isTIF(file):
                continue

            paras = file.split('_')
            year = paras[0]
            day = paras[1].replace('.tif', '')
            if year == '2015' and int(day) < 16:
                continue

            if site == 'XH':
                point, code = extract_patch_site(file.replace('.tif', ''), 447, 151)  # XH
            if site == 'HF':
                point, code = extract_patch_site(file.replace('.tif', ''), 469, 230)  # HF
            if site == 'WLG':
                point, code = extract_patch_site(file.replace('.tif', ''), 286, 187)  # WLG
            if site == 'LLN':
                point, code = extract_patch_site(file.replace('.tif', ''), 486, 315)  # LLN

            codes = np.concatenate((codes, code), axis=0)
            points = np.concatenate((points, point), axis=0)

        print(points.shape)
        print(codes.shape)
        np.save(r'npy/patch_{}.npy'.format(site), points)
        np.save(r'npy/code_{}.npy'.format(site), codes)
