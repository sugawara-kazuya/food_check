# rとしているため読み込んでいる
with open('labels.txt', 'r') as f:
    labels_list = f.read().splitlines()
# mapはイテレータとして取得する
labels_list = list(map(lamdba x :.lower(), labels_list))

import glob
import cv2

def load_images(dir_path, label, resieze_shape, max_n = None):
    #globとは引数に指定されたパターンにマッチするファイルパス名を取得することができる
    image_path_list = glob.glob(dir_path + label + "/*jpg")
    #もしmax_nがNoneとmax_path_list>max_nでない時
    if max_n is not None and len(image_path_list) > max_n:
    # image_path_listにmax_nまで入れる
        image_path_list = image_path_list[:max_n]
    # 空のimage_listを作成
    image_list = []
    # 画像を読み取る
    for image_path in image_path_list:
        # 画像の読み込み
        image = cv2.imread(image_path)
        # imageを取得して画像サイズを変更
        image = cv2.resize(image, resize_shape)
        # BGR画像をRGBに変更
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image_listにimageを追加している
        iamge_list.append(image)
    return image_list, [labels_list.index(label)] * len(image_lsit)

# ディレクトリを指定
dir_path = 'image_food'
# 高里幅を定義
width = 200
hegiht = 200
# resize_shapeにsizeを入れる
resize_shape = tuple((witdh, hegith))
# 空のリストを指定している
image_list = []
label_lsit = []
for label in label_list:
    image_list_temp, label_temp = load_images(dir_path, label, resize_shape)
    image_list.extend(image_list_temp)

# np.stackで１つ上の次元を作成。つまり4次元に変更した
X = np.stack(image_list, axis=0)
Y = np.array(label_list)
# to_categoricalを使用してone-hot化した
Y = to_categorycal(Y)

print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)