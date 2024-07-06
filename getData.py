import pyodbc
from collections import Counter
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from yellowbrick.cluster import KElbowVisualizer
import random
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
import re


def process_storage(text):
    if "GB" in text:
        # Trường hợp có "GB" (gigabyte)
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[0])  # Lấy giá trị số đầu tiên
    elif "MB" in text:
        # Trường hợp có "MB" (megabyte)
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[0]) / 1024  # Chuyển MB thành GB
    elif "TB" in text:
        # Trường hợp có "TB" (terabyte)
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[0]) * 1024  # Chuyển TB thành GB
    elif "Unspecified" in text or "No card slot" in text:
        # Trường hợp không xác định hoặc không có khe cắm thẻ
        return 0
    else:
        # Trường hợp không xác định
        return 0


def extract_ram(text):
    # Loại bỏ dấu &nbsp;
    text = text.replace('&nbsp;', '0')

    # Tìm các con số và dấu '/' trong chuỗi
    matches = re.findall(
        r'(\d+(?:\.\d+)?)\s?(?:\/)?\s?(\d+(?:\.\d+)?)?\s?(GB|MB)?', text)

    # Tạo danh sách các giá trị RAM chuyển đổi thành số
    ram_values = []
    for match in matches:
        val1 = float(match[0]) if match[0] else 0
        val2 = float(match[1]) if match[1] else 0
        unit = match[2]

        if unit == 'GB':
            val1 *= 1024  # Chuyển đổi GB thành MB

        if val2 != 0:
            # Nếu tồn tại phép chia
            val1 = val1 / val2

        ram_values.append(val1)

    # Tìm giá trị RAM lớn nhất trong danh sách
    max_ram = max(ram_values)

    return max_ram


def extract_battery(text):
    # Loại bỏ dấu &nbsp;
    text = text.replace('&nbsp;', '0')

    # Tìm các con số và "mAh" trong chuỗi
    match = re.search(r'(\d+)\s*mAh', text)

    if match:
        return int(match.group(1))
    else:
        return 0


server = 'LAPTOP-DOUILK3I'
database = 'CUA_HANG_DIEN_THOAI'
username = 'sa'
password = '123456'
conn = pyodbc.connect(
    f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}')


# Truy vấn dữ liệu từ bảng "devices"
# query = 'SELECT * FROM devices'
# df = pd.read_sql(query, conn)
# df.to_csv('data.csv', index=False)
# print("thành công")
# cursor = conn.cursor()
# cursor.execute(query)
# user_history = {}

# # Danh sách người dùng mẫu và sản phẩm
# users = ['user1', 'user2', 'user3', 'user4', 'user5']
# products = [row for row in cursor.fetchall()]

# # Tạo dữ liệu lịch sử xem ngẫu nhiên
# for user_id in range(1, 2001):
#     # Tạo tên người dùng ví dụ: user1, user2, ..., user2000
#     user = f'user{user_id}'

#     # Lấy 5 sản phẩm xem ngẫu nhiên cho mỗi người dùng
#     viewed_products = random.sample(products, 5)
#     test = []
#     for i in viewed_products:
#         for j in i:
#             test.append(j)
#     user_history[user] = test

# # Đóng kết nối
# conn.close()
# df = pd.DataFrame.from_dict(user_history, orient='index')
# df = df.reset_index()
# df = df.rename(columns={'index': 'User'})
# df.to_csv('user_history.csv', index=False)
# # Hiển thị từ điển lịch sử xem
# print(user_history)

# df = pd.read_sql(query, conn)
# df.to_csv('devices_data.csv', index=False)
# conn.close()


dataall = pd.read_csv('data.csv')
data = pd.read_csv('data.csv')
data = data[['device_name', 'os', 'specifications', 'chipset']]
dataall = dataall.drop(['url_hash'], axis=1)
data = data.fillna("0")  # Thay thế các giá trị NaN bằng 0 hoặc giá trị khác
data = data.drop_duplicates()  # Loại bỏ các dòng trùng lặp


all_words = set()
for col in data.columns:
    if data[col].dtype == 'object':
        for row in data[col]:
            if isinstance(row, str):
                all_words.update(row.split())
    else:
        all_words.update(data[col].astype(str))
model = Word2Vec(sentences=list(all_words), vector_size=100,
                 window=5, min_count=1, sg=0)
encoded_data = pd.DataFrame()
for col in data.columns:
    if data[col].dtype == 'object':
        col_vectors = data[col].str.split().apply(lambda words: [model.wv[word] for word in words if word in model.wv]).apply(
            lambda vectors: sum(vectors) if vectors else [0.0] * 100)
        col_vectors = col_vectors.apply(lambda vector: pd.Series(vector))
        encoded_data = pd.concat([encoded_data, col_vectors], axis=1)
t = dataall['body']
body = []
for i in t:
    if pd.notna(i):
        numbers = [float(number) for number in re.findall(r'\d+\.\d+|\d+', i)]
        body.append(numbers)
    else:
        body.append([0])
rs = []
for sublist in body:
    if len(sublist) > 1:
        rs.append(sublist[0]*sublist[1])
    else:
        rs.append(sublist[0])
df_body = pd.DataFrame(rs, columns=['body'])

t = dataall['storage']
storage = []
for i in t:
    storage.append(process_storage(i.split(',')[0]))
df_storage = pd.DataFrame(storage, columns=['storage'])

t = dataall['display_size']
display_size = [float(re.search(r'\d+\.\d+', s).group())
                for s in t if re.search(r'\d+\.\d+', s)]
df_display_size = pd.DataFrame(display_size, columns=['display_size'])

t = dataall['display_resolution']
display_resolution = []
pattern = r'(\d+)x(\d+) pixels'
for item in t:
    match = re.search(pattern, item)
    if match:
        # Chuyển đổi con số thành số nguyên
        width, height = map(int, match.groups())
        display_resolution.append(width*height)
df_display_resolution = pd.DataFrame(
    display_resolution, columns=['display_resolution'])

t = dataall['camera_pixels']
camera_pixels = [0 if val.strip() == 'NO' else float(val.split()[0])
                 for val in t]
df_camera_pixels = pd.DataFrame(
    camera_pixels, columns=['camera_pixels'])

t = dataall['video_pixels']
video_pixels = []
mapping = {
    '2160p': 2160,
    '720p': 720,
    '1080p': 1080,
    'No video recorder': 0,
    'Video recorder': 0,
    '288p': 288,
    '480p': 480,
    '4320p': 4320,
    '3240p': 3240,
    '1440p': 1440,
    '240p': 240,
}
t = [mapping[val] if val in mapping else val for val in t]
video_pixels = [0 if pd.isna(val) else val for val in t]
df_video_pixels = pd.DataFrame(
    video_pixels, columns=['video_pixels'])

t = dataall['ram']
ram = [extract_ram(text) for text in t]

df_ram = pd.DataFrame(
    ram, columns=['ram'])

t = dataall['battery_size']
battery_size = [extract_battery(text) for text in t]
df_battery_size = pd.DataFrame(
    battery_size, columns=['battery_size'])
t = dataall['battery_type']
label_encoder = LabelEncoder()
battery_type = label_encoder.fit_transform(t)
df_battery_type = pd.DataFrame(
    battery_type, columns=['battery_type'])

encoded_data = pd.concat([encoded_data, df_body], axis=1)
encoded_data = pd.concat([encoded_data, df_battery_type], axis=1)
encoded_data = pd.concat([encoded_data, df_battery_size], axis=1)
encoded_data = pd.concat([encoded_data, df_camera_pixels], axis=1)
encoded_data = pd.concat([encoded_data, df_display_resolution], axis=1)
encoded_data = pd.concat([encoded_data, df_ram], axis=1)
encoded_data = pd.concat([encoded_data, df_video_pixels], axis=1)
encoded_data = pd.concat([encoded_data, df_storage], axis=1)
encoded_data = pd.concat([encoded_data, df_display_size], axis=1)
encoded_data = pd.concat([encoded_data, dataall['gia']], axis=1)


# scaler = MinMaxScaler(feature_range=(0, 20))
# column_to_normalize = encoded_data['body']
# standardized_column = scaler.fit_transform(
#     column_to_normalize.values.reshape(-1, 1))
# encoded_data['body'] = standardized_column

# column_to_normalize = encoded_data['battery_type']
# standardized_column = scaler.fit_transform(
#     column_to_normalize.values.reshape(-1, 1))
# encoded_data['battery_type'] = standardized_column

# column_to_normalize = encoded_data['battery_size']
# standardized_column = scaler.fit_transform(
#     column_to_normalize.values.reshape(-1, 1))
# encoded_data['battery_size'] = standardized_column

# column_to_normalize = encoded_data['camera_pixels']
# standardized_column = scaler.fit_transform(
#     column_to_normalize.values.reshape(-1, 1))
# encoded_data['camera_pixels'] = standardized_column


# column_to_normalize = encoded_data['display_resolution']
# standardized_column = scaler.fit_transform(
#     column_to_normalize.values.reshape(-1, 1))
# encoded_data['display_resolution'] = standardized_column


# column_to_normalize = encoded_data['ram']
# standardized_column = scaler.fit_transform(
#     column_to_normalize.values.reshape(-1, 1))
# encoded_data['ram'] = standardized_column


# column_to_normalize = encoded_data['video_pixels']
# standardized_column = scaler.fit_transform(
#     column_to_normalize.values.reshape(-1, 1))
# encoded_data['video_pixels'] = standardized_column


# column_to_normalize = encoded_data['storage']
# standardized_column = scaler.fit_transform(
#     column_to_normalize.values.reshape(-1, 1))
# encoded_data['storage'] = standardized_column

# column_to_normalize = encoded_data['display_size']
# standardized_column = scaler.fit_transform(
#     column_to_normalize.values.reshape(-1, 1))
# encoded_data['display_size'] = standardized_column


# column_to_normalize = encoded_data['gia']
# standardized_column = scaler.fit_transform(
#     column_to_normalize.values.reshape(-1, 1))
# encoded_data['gia'] = standardized_column

# encoded_data.to_csv('temp2.csv')

# encoded_data = pd.concat([encoded_data, df], axis=1)
# encoded_data.to_csv("temp1.csv")
# print(encoded_data)
# encoded_data = pd.concat([encoded_data, dataall['cluster']], axis=1)


# data_array = encoded_data.to_numpy()
# reference_product = data_array[0]
# cluster_products = encoded_data[encoded_data['cluster']
#                                 == reference_product[-1]]
# cluster_products = cluster_products[cluster_products.index != 0]
# cluster_products_2d = cluster_products.iloc[:, :-1].values.copy()
# reference_product_2d = reference_product[:-1]
# reference_product_2d = reference_product_2d.reshape(1, -1).copy()
# similarities = pairwise_distances(
#     reference_product_2d, cluster_products_2d, metric='euclidean')
# sorted_indices = np.argsort(similarities.flatten())
# top_n_indices = sorted_indices[:5]
# top_n_products = cluster_products.iloc[top_n_indices]
# related_products = top_n_products.index
# related_products_id = []
# for i in related_products:
#     related_products_id.append(dataall.iloc[i]['id'])


# final_data = pd.concat([encoded_data, dataall['gia']], axis=1)
# # final_data = encoded_data

model = KMeans()
visualizer = KElbowVisualizer(model, k=(2, 10))
visualizer.fit(encoded_data)
visualizer.show()


# for n_clusters in num_clusters:
#     kmeans = KMeans(n_clusters=n_clusters)
#     kmeans.fit(final_data)
#     distortions.append(kmeans.inertia_)
# print(distortions)

# plt.plot(num_clusters, distortions, marker='o')
# plt.xlabel('Số lượng cụm')
# plt.ylabel('Distortion')
# plt.title('Biểu đồ Elbow')
# plt.show()


k = visualizer.elbow_value_
kmeans = KMeans(n_clusters=k, random_state=0, init='k-means++')
encoded_data['cluster'] = kmeans.fit_predict(encoded_data)
final_data = pd.concat([encoded_data, dataall['id']], axis=1)
final_data.to_csv('test.csv')
# final_data = pd.read_csv('phancum.csv')
# query = 'UPDATE devices SET devices.cluster = ? WHERE devices.id = ?'
# for row in final_data.itertuples():
#     print(row.cluster, row.id)
#     conn.execute(query, row.cluster, row.id)
# conn.commit()
# conn.close()
# print('thành công')

# data['cluster'] = kmeans.fit_predict(encoded_data)
# reference_product = data.iloc[0]
# similar_products = data[data['cluster'] == reference_product['cluster']]
# data_array = encoded_data.to_numpy()
# reference_product = data_array[0]
# distances = pairwise_distances(
#     data_array, reference_product.reshape((1, -1)), metric='euclidean')
# similar_products = data.iloc[np.argsort(
#     distances.flatten())].head(10).index

kmeans.fit(encoded_data)
labels = kmeans.labels_

# Đếm số lượng mẫu trong từng cụm
cluster_counts = Counter(labels)

# Hiển thị số lượng mẫu trong từng cụm
for cluster, count in cluster_counts.items():
    print(f"Cụm {cluster}: {count} mẫu")


# print(dataall.loc[dataall['id'] == 8522])

app = Flask(__name__)


def load_dulieu():
    query = 'SELECT * FROM devices'
    df = pd.read_sql(query, conn)
    df.to_csv('data.csv', index=False)


def ma_hoa(dataall):
    data = pd.read_csv('data.csv')
    data = data[['device_name', 'os', 'specifications', 'chipset']]
    # Thay thế các giá trị NaN bằng 0 hoặc giá trị khác
    data = data.fillna("0")
    data = data.drop_duplicates()  # Loại bỏ các dòng trùng lặp
    all_words = set()
    for col in data.columns:
        if data[col].dtype == 'object':
            for row in data[col]:
                if isinstance(row, str):
                    all_words.update(row.split())
        else:
            all_words.update(data[col].astype(str))
    model = Word2Vec(sentences=list(all_words), vector_size=100,
                     window=5, min_count=1, sg=0)
    encoded_data = pd.DataFrame()
    for col in data.columns:
        if data[col].dtype == 'object':
            col_vectors = data[col].str.split().apply(lambda words: [model.wv[word] for word in words if word in model.wv]).apply(
                lambda vectors: sum(vectors) if vectors else [0.0] * 100)
            col_vectors = col_vectors.apply(lambda vector: pd.Series(vector))
            encoded_data = pd.concat([encoded_data, col_vectors], axis=1)
    t = dataall['body']
    body = []
    for i in t:
        if pd.notna(i):
            numbers = [float(number)
                       for number in re.findall(r'\d+\.\d+|\d+', i)]
            body.append(numbers)
        else:
            body.append([0])
    rs = []
    for sublist in body:
        if len(sublist) > 1:
            rs.append(sublist[0]*sublist[1])
        else:
            rs.append(sublist[0])
    df_body = pd.DataFrame(rs, columns=['body'])

    t = dataall['storage']
    storage = []
    for i in t:
        storage.append(process_storage(i.split(',')[0]))
    df_storage = pd.DataFrame(storage, columns=['storage'])

    t = dataall['display_size']
    display_size = [float(re.search(r'\d+\.\d+', s).group())
                    for s in t if re.search(r'\d+\.\d+', s)]
    df_display_size = pd.DataFrame(display_size, columns=['display_size'])

    t = dataall['display_resolution']
    display_resolution = []
    pattern = r'(\d+)x(\d+) pixels'
    for item in t:
        match = re.search(pattern, item)
        if match:
            # Chuyển đổi con số thành số nguyên
            width, height = map(int, match.groups())
            display_resolution.append(width*height)
    df_display_resolution = pd.DataFrame(
        display_resolution, columns=['display_resolution'])

    t = dataall['camera_pixels']
    camera_pixels = [0 if val.strip() == 'NO' else float(val.split()[0])
                     for val in t]
    df_camera_pixels = pd.DataFrame(
        camera_pixels, columns=['camera_pixels'])

    t = dataall['video_pixels']
    video_pixels = []
    mapping = {
        '2160p': 2160,
        '720p': 720,
        '1080p': 1080,
        'No video recorder': 0,
        'Video recorder': 0,
        '288p': 288,
        '480p': 480,
        '4320p': 4320,
        '3240p': 3240,
        '1440p': 1440,
        '240p': 240,
    }
    t = [mapping[val] if val in mapping else val for val in t]
    video_pixels = [0 if pd.isna(val) else val for val in t]
    df_video_pixels = pd.DataFrame(
        video_pixels, columns=['video_pixels'])

    t = dataall['ram']
    ram = [extract_ram(text) for text in t]

    df_ram = pd.DataFrame(
        ram, columns=['ram'])

    t = dataall['battery_size']
    battery_size = [extract_battery(text) for text in t]
    df_battery_size = pd.DataFrame(
        battery_size, columns=['battery_size'])
    t = dataall['battery_type']
    label_encoder = LabelEncoder()
    battery_type = label_encoder.fit_transform(t)
    df_battery_type = pd.DataFrame(
        battery_type, columns=['battery_type'])

    encoded_data = pd.concat([encoded_data, df_body], axis=1)
    encoded_data = pd.concat([encoded_data, df_battery_type], axis=1)
    encoded_data = pd.concat([encoded_data, df_battery_size], axis=1)
    encoded_data = pd.concat([encoded_data, df_camera_pixels], axis=1)
    encoded_data = pd.concat([encoded_data, df_display_resolution], axis=1)
    encoded_data = pd.concat([encoded_data, df_ram], axis=1)
    encoded_data = pd.concat([encoded_data, df_video_pixels], axis=1)
    encoded_data = pd.concat([encoded_data, df_storage], axis=1)
    encoded_data = pd.concat([encoded_data, df_display_size], axis=1)
    encoded_data = pd.concat([encoded_data, dataall['gia']], axis=1)
    encoded_data = pd.concat([encoded_data, dataall['cluster']], axis=1)
    return encoded_data


@app.route('/sanphamcolienquan', methods=['GET'])
def recommendations():
    product_id = request.args.get('product_id')
    print(product_id)
    if product_id is None:
        return
    load_dulieu()
    dataall = pd.read_csv('data.csv')
    product_index = dataall.loc[dataall['id'] == int(product_id)].index[0]
    print(product_index)
    encoded_data = ma_hoa(dataall)
    data_array = encoded_data.to_numpy()
    reference_product = data_array[product_index]
    cluster_products = encoded_data[encoded_data['cluster']
                                    == reference_product[-1]]
    cluster_products = cluster_products[cluster_products.index != product_index]
    cluster_products_2d = cluster_products.iloc[:, :-1].values.copy()
    reference_product_2d = reference_product[:-1]
    reference_product_2d = reference_product_2d.reshape(1, -1).copy()
    similarities = pairwise_distances(
        reference_product_2d, cluster_products_2d, metric='euclidean')
    sorted_indices = np.argsort(similarities.flatten())
    top_n_indices = sorted_indices[:5]
    top_n_products = cluster_products.iloc[top_n_indices]
    related_products = top_n_products.index
    related_products_id = []
    for i in related_products:
        related_products_id.append(int(dataall.iloc[i]['id']))
    print(related_products_id)
    return jsonify({"related_products_id": related_products_id})


if __name__ == '__main__':
    app.run(debug=True)
