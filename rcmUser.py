from flask import Flask, request, jsonify
from collections import Counter
import pyodbc
import pandas as pd
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from yellowbrick.cluster import KElbowVisualizer
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
from datetime import timedelta


server = 'LAPTOP-DOUILK3I'
database = 'CUA_HANG_DIEN_THOAI'
username = 'sa'
password = '123456'
print("bắt đầu")
conn = pyodbc.connect(
    f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}')

# data = pd.read_csv('user_history.csv')
# unique_user_ids = data['user_id'].unique()
# user_to_indexdata = {user_id: index for index,
#                      user_id in enumerate(unique_user_ids)}
# item_idsdata = data['item_id'].unique()
# index_to_itemdata = {i: item_id for i, item_id in enumerate(item_idsdata)}
# train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
# user_ids = train_data['user_id'].unique()
# item_ids = train_data['item_id'].unique()
# num_users = len(user_ids)
# num_items = len(item_ids)
# user_to_index = {user_id: i for i, user_id in enumerate(user_ids)}
# index_to_user = {i: user_id for i, user_id in enumerate(user_ids)}
# item_to_index = {item_id: i for i, item_id in enumerate(item_ids)}
# index_to_item = {i: item_id for i, item_id in enumerate(item_ids)}
# train_user_ids = train_data['user_id'].map(user_to_index)
# train_item_ids = train_data['item_id'].map(item_to_index)
# trainRatings = np.array(train_data['rating'])
# test_user_ids = test_data['user_id'].map(user_to_index)
# test_item_ids = test_data['item_id'].map(item_to_index)
# testRatings = np.array(test_data['rating'])
# user_input = tf.keras.layers.Input(shape=(1,), name='user_input')
# item_input = tf.keras.layers.Input(shape=(1,), name='item_input')
# user_embedding = tf.keras.layers.Embedding(
#     input_dim=len(train_user_ids), output_dim=64)(user_input)
# item_embedding = tf.keras.layers.Embedding(
#     input_dim=len(train_item_ids), output_dim=64)(item_input)

# user_flatten = tf.keras.layers.Flatten()(user_embedding)
# item_flatten = tf.keras.layers.Flatten()(item_embedding)

# concat = tf.keras.layers.Concatenate()([user_flatten, item_flatten])

# hidden1 = tf.keras.layers.Dense(64, activation='relu')(concat)
# hidden2 = tf.keras.layers.Dense(32, activation='relu')(hidden1)
# output = tf.keras.layers.Dense(1, activation='sigmoid')(hidden2)
# print(train_user_ids.shape[0])  # In ra số lượng mẫu trong train_user_ids
# print(train_item_ids.shape[0])  # In ra số lượng mẫu trong train_item_ids
# print(trainRatings.shape[0])
# model = tf.keras.models.Model(inputs=[user_input, item_input], outputs=output)
# model.compile(optimizer='adam', loss='mean_squared_error')

# model.fit([train_user_ids, train_item_ids], trainRatings, epochs=10,
#           batch_size=64, validation_data=([test_user_ids, test_item_ids], testRatings))

# model.save('mymodel.h5')
# # Xác định người dùng cụ thể và danh sách sản phẩm đã tương tác
# user_id = 'User1'
# # Lấy chỉ mục của người dùng cụ thể từ tập huấn luyện
# user_index_to_recommend = user_to_indexdata.get(user_id)


# if user_index_to_recommend is not None:
#     # Số lượng sản phẩm cần đề xuất
#     k = 5  # Điều chỉnh theo số lượng sản phẩm bạn muốn đề xuất

#     # Tạo danh sách sản phẩm chưa tương tác
#     all_items_to_recommend = np.setdiff1d(np.arange(
#         len(item_ids)), data[data['user_id'] == user_id]['item_id'].unique())

#     if len(all_items_to_recommend) > 0:
#         # Sử dụng mô hình NCF để dự đoán xác suất tương tác cho các sản phẩm chưa tương tác
#         predicted_probabilities = model.predict([np.array(
#             [user_index_to_recommend] * len(all_items_to_recommend)), all_items_to_recommend])

#         # Sắp xếp các sản phẩm theo xác suất giảm dần
#         sorted_indices = np.argsort(predicted_probabilities, axis=0)[::-1]
#         top_k_recommendations = all_items_to_recommend[sorted_indices][:k]
#         print(top_k_recommendations)
#         recommended_item_ids = []
#         for i in top_k_recommendations:
#             recommended_item_ids.append(index_to_itemdata[i[0]])
#         print(recommended_item_ids)
#     else:
#         print("Không còn sản phẩm để đề xuất cho người dùng này.")
# else:
#     print(f"Người dùng với ID {user_id} không tồn tại trong tập huấn luyện.")


app = Flask(__name__)


def load_data():
    query = 'SELECT * FROM user_history'
    df = pd.read_sql(query, conn)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(
        ['user_id', 'item_id', 'timestamp'])

    # Tính thời gian giữa các lần xem cho cùng một sản phẩm của mỗi người dùng
    df['time_diff'] = df.groupby(
        ['user_id', 'item_id'])['timestamp'].diff()
    df['rating'] = 1
    # Điều chỉnh xếp hạng nếu sản phẩm được mua
    df.loc[df['buy'] == 1, 'rating'] += 1
    # Điều chỉnh xếp hạng dựa vào thời gian giữa các lần xem
    df['rating'] += df.groupby(['user_id', 'item_id'])['time_diff'].apply(
        lambda x: (x < timedelta(hours=1)).cumsum())

    # Điều chỉnh xếp hạng dựa trên số lần xem
    df['rating'] += df.groupby(
        ['user_id', 'item_id'])['item_id'].cumcount()
    df.to_csv('user_history.csv')


def get_user_history():
    query = 'SELECT * FROM user_history'
    df = pd.read_sql(query, conn)
    return df


def len_devices():
    cursor = conn.cursor()
    query = 'SELECT COUNT(*) FROM devices'
    cursor.execute(query)
    row_count = cursor.fetchone()[0]
    cursor.close()
    return int(row_count)


def get_product_interact(user_id):
    query = 'SELECT item_id FROM user_history where id = ?'
    cursor = conn.cursor()
    cursor.execute(query, user_id)
    result = cursor.fetchall()
    conn.close()
    item_ids_user_inter = [row[0] for row in result]
    return item_ids_user_inter


@app.route('/trainmodel', methods=['GET'])
def train_model():
    load_data()
    data = pd.read_csv('user_history.csv')
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42)
    user_ids = train_data['user_id'].unique()
    item_ids = train_data['item_id'].unique()
    user_to_index = {user_id: i for i, user_id in enumerate(user_ids)}
    item_to_index = {item_id: i for i, item_id in enumerate(item_ids)}
    train_user_ids = train_data['user_id'].map(user_to_index)
    train_item_ids = train_data['item_id'].map(item_to_index)
    trainRatings = np.array(train_data['rating'])
    test_user_ids = test_data['user_id'].map(user_to_index)
    test_item_ids = test_data['item_id'].map(item_to_index)
    testRatings = np.array(test_data['rating'])
    user_input = tf.keras.layers.Input(shape=(1,), name='user_input')
    item_input = tf.keras.layers.Input(shape=(1,), name='item_input')
    user_embedding = tf.keras.layers.Embedding(
        input_dim=len(train_user_ids), output_dim=64)(user_input)
    item_embedding = tf.keras.layers.Embedding(
        input_dim=len(train_item_ids), output_dim=64)(item_input)

    user_flatten = tf.keras.layers.Flatten()(user_embedding)
    item_flatten = tf.keras.layers.Flatten()(item_embedding)

    concat = tf.keras.layers.Concatenate()([user_flatten, item_flatten])

    hidden1 = tf.keras.layers.Dense(64, activation='relu')(concat)
    hidden2 = tf.keras.layers.Dense(32, activation='relu')(hidden1)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(hidden2)
    model = tf.keras.models.Model(
        inputs=[user_input, item_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit([train_user_ids, train_item_ids], trainRatings, epochs=10,
              batch_size=64, validation_data=([test_user_ids, test_item_ids], testRatings))
    model.save('mymodel.h5')
    return jsonify({"train": "thanhcong"})


@app.route('/cothebanquantam', methods=['GET'])
def recommendations():
    user_id = int(request.args.get('user_id'))
    product_id = int(request.args.get('product_id'))
    print(product_id)
    cluster = pd.read_csv('data.csv')
    product = cluster[cluster['id'] == product_id]
    cluster_product = product['cluster']
    print(cluster_product)
    print(cluster_product.iloc[0])
    data = get_user_history()
    dataitem = data['item_id'].unique()
    item_to_index = {item_id: i for i, item_id in enumerate(dataitem)}
    unique_user_ids = data['user_id'].unique()
    user_to_indexdata = {user_id: index for index,
                         user_id in enumerate(unique_user_ids)}
    user_index_to_recommend = user_to_indexdata.get(user_id)
    index_to_itemdata = {i: item_id for i, item_id in enumerate(dataitem)}
    model = tf.keras.models.load_model("mymodel.h5")
    if user_index_to_recommend is not None:
        # Số lượng sản phẩm cần đề xuất
        k = 5  # Điều chỉnh theo số lượng sản phẩm bạn muốn đề xuất
        filtered_data = cluster[cluster['cluster']
                                == cluster_product.iloc[0]]
        items_with_cluster = filtered_data['id'].unique()
        filtered_data = data[data['item_id'].isin(items_with_cluster)]
        items_with_cluster = filtered_data['item_id'].unique()
        index_of_cluster = []
        for i in items_with_cluster:
            index_of_cluster.append(item_to_index[i])
        all_items_to_recommend = np.intersect1d(
            np.arange(len(dataitem)), np.array(index_of_cluster))
        if len(all_items_to_recommend) > 0:
            # Sử dụng mô hình NCF để dự đoán xác suất tương tác cho các sản phẩm chưa tương tác
            predicted_probabilities = model.predict([np.array(
                [user_index_to_recommend] * len(all_items_to_recommend)), all_items_to_recommend])

            # Sắp xếp các sản phẩm theo xác suất giảm dần
            sorted_indices = np.argsort(predicted_probabilities, axis=0)[::-1]
            top_k_recommendations = all_items_to_recommend[sorted_indices][:k]
            recommended_item_ids = []
            for i in top_k_recommendations:
                recommended_item_ids.append(int(index_to_itemdata[i[0]]))
            print(recommended_item_ids)
            return jsonify({"related_products_id": recommended_item_ids})
        else:
            return jsonify({"related_products_id": "khong co san pham de xuat cho nguoi dung nay"})
    else:
        return jsonify({"related_products_id": "nguoi dung khong ton tai trong tap huan luyen"})


if __name__ == '__main__':
    app.run(debug=True)
