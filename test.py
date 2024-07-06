import pyodbc
import pandas as pd
import random
from datetime import datetime, timedelta
import numpy as np
import re

# # Kết nối với cơ sở dữ liệu SQL Server
server = 'LAPTOP-DOUILK3I'
database = 'CUA_HANG_DIEN_THOAI'
username = 'sa'
password = '123456'
conn = pyodbc.connect(
    f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}')

# Truy vấn dữ liệu từ bảng "devices"
# query = 'SELECT * FROM user_history'
# df = pd.read_sql(query, conn)
# df.to_csv('temp3.csv', index=False)

print(pd.__version__)
# # Truy vấn dữ liệu từ bảng "devices" để lấy danh sách sản phẩm
# query = 'SELECT id FROM devices'
# products = [row.id for row in conn.execute(query)]
# end_date = datetime.today()
# start_date = end_date - timedelta(days=2)
# date_range = [start_date + timedelta(days=i) for i in range(2)]
# # Tạo dữ liệu lịch sử xem cho 2000 người dùng
# user_history = []
# for user_id in range(1, 2001):
#     user = f'{user_id}'  # Tạo tên người dùng

#     viewed_products = random.sample(products, 6)

#     for i in range(10):
#         product = random.sample(viewed_products, 1)
#         timestamp = random.choice(
#             date_range) + timedelta(minutes=random.randint(0, 24 * 60 - 1))
#         user_history.append([user, product[0], timestamp])

# # Tạo DataFrame từ user_history
# df = pd.DataFrame(user_history, columns=['user_id', 'item_id', 'timestamp'])

# # Lưu DataFrame vào tệp CSV
# df.to_csv('user_history.csv', index=False)

# # Đóng kết nối


# data = pd.read_csv('user_history.csv')

# data['timestamp'] = pd.to_datetime(data['timestamp'])
# data = data.sort_values(['user_id', 'item_id', 'timestamp'])

# # Tính thời gian giữa các lần xem cho cùng một sản phẩm của mỗi người dùng
# data['time_diff'] = data.groupby(['user_id', 'item_id'])['timestamp'].diff()

# # Thiết lập xếp hạng mặc định dựa trên số lần xem
# data['rating'] = 1

# # Điều chỉnh xếp hạng dựa vào thời gian giữa các lần xem
# data['rating'] += data.groupby(['user_id', 'item_id'])['time_diff'].apply(
#     lambda x: (x < timedelta(hours=1)).cumsum())

# # Điều chỉnh xếp hạng dựa trên số lần xem
# data['rating'] += data.groupby(['user_id', 'item_id'])['item_id'].cumcount()
# data = pd.read_csv('user_history.csv')

# for row in data.itertuples():
#     print(row.user_id, row.item_id, row.timestamp, row.rating)
#     query = 'INSERT INTO user_history(user_id, item_id, timestamp, rating) VALUES(?,?,?,?)'
#     conn.execute(query, row.user_id, row.item_id,
#                  row.timestamp, row.rating)
# conn.commit()
# conn.close()
# print('thành công')
# data.to_csv('user_history.csv', index=False)


# def process_storage(text):
#     if "GB" in text:
#         # Trường hợp có "GB" (gigabyte)
#         numbers = re.findall(r'\d+', text)
#         if numbers:
#             return int(numbers[0])  # Lấy giá trị số đầu tiên
#     elif "MB" in text:
#         # Trường hợp có "MB" (megabyte)
#         numbers = re.findall(r'\d+', text)
#         if numbers:
#             return int(numbers[0]) / 1024  # Chuyển MB thành GB
#     elif "TB" in text:
#         # Trường hợp có "TB" (terabyte)
#         numbers = re.findall(r'\d+', text)
#         if numbers:
#             return int(numbers[0]) * 1024  # Chuyển TB thành GB
#     elif "Unspecified" in text or "No card slot" in text:
#         # Trường hợp không xác định hoặc không có khe cắm thẻ
#         return 0
#     else:
#         # Trường hợp không xác định
#         return 0


# def extract_ram(text):
#     # Loại bỏ dấu &nbsp;
#     text = text.replace('&nbsp;', '0')

#     # Tìm các con số và dấu '/' trong chuỗi
#     matches = re.findall(
#         r'(\d+(?:\.\d+)?)\s?(?:\/)?\s?(\d+(?:\.\d+)?)?\s?(GB|MB)?', text)

#     # Tạo danh sách các giá trị RAM chuyển đổi thành số
#     ram_values = []
#     for match in matches:
#         val1 = float(match[0]) if match[0] else 0
#         val2 = float(match[1]) if match[1] else 0
#         unit = match[2]

#         if unit == 'GB':
#             val1 *= 1024  # Chuyển đổi GB thành MB

#         if val2 != 0:
#             # Nếu tồn tại phép chia
#             val1 = val1 / val2

#         ram_values.append(val1)

#     # Tìm giá trị RAM lớn nhất trong danh sách
#     max_ram = max(ram_values)

#     return max_ram


# def extract_battery(text):
#     # Loại bỏ dấu &nbsp;
#     text = text.replace('&nbsp;', '0')

#     # Tìm các con số và "mAh" trong chuỗi
#     match = re.search(r'(\d+)\s*mAh', text)

#     if match:
#         return int(match.group(1))
#     else:
#         return 0


# data = pd.read_csv('data.csv')
# data = data.drop(['url_hash', 'id', 'gia', 'cluster',
#                  'deleted_at', 'so_luong', 'picture', 'brand_id', 'released_at',], axis=1)
# # data.to_csv('temp.csv')
# # cột body
# t = data['body'].unique()
# for i in t:
#     if pd.notna(i):
#         numbers = [float(number) for number in re.findall(r'\d+\.\d+|\d+', i)]

# # cột os dùng w2vec
# # cot storage
# t = data['storage'].unique()
# encoded_data = [process_storage(i.split(',')[0]) for i in t]
# print(encoded_data)
# # cột display_size
# t = data['display_size'].unique()

# numbers = [float(re.search(r'\d+\.\d+', s).group())
#            for s in t if re.search(r'\d+\.\d+', s)]
# # cột display_resolution
# t = data['display_resolution'].unique()
# screen_sizes = []  # Danh sách lưu trữ kích thước màn hình

# # Biểu thức chính quy để tìm các con số trong chuỗi
# pattern = r'(\d+)x(\d+) pixels'

# for item in t:
#     match = re.search(pattern, item)
#     if match:
#         # Chuyển đổi con số thành số nguyên
#         width, height = map(int, match.groups())
#         screen_sizes.append([width, height])

# # cột camera_pixels

# t = data['camera_pixels'].unique()
# t = [0 if val.strip() == 'NO' else float(val.split()[0]) for val in t]

# # cột video_pixels
# t = data['video_pixels'].unique()
# mapping = {
#     '2160p': 2160,
#     '720p': 720,
#     '1080p': 1080,
#     'No video recorder': 0,
#     'Video recorder': 0,
#     '288p': 288,
#     '480p': 480,
#     '4320p': 4320,
#     '3240p': 3240,
#     '1440p': 1440,
# }
# t = [mapping[val] if val in mapping else val for val in t]
# t = [0 if pd.isna(val) else val for val in t]
# # cột ram
# t = data['ram'].unique()
# t = [extract_ram(text) for text in t]

# # chip set dung vec

# # cột battery_size
# t = data['battery_size'].unique()
# t = [extract_battery(text) for text in t]


# # battery_type mã hóa LabelEncoder
