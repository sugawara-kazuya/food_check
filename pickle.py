import pickle
# pickle = 複数のオブジェクトを１つにまとめることができる
with open('dataset.pkl', 'wb') as f:
    pickle.dump((X, Y), f)

with open('/dataset.pkl', 'rb') as f:
    X, Y = pickle.load(f)

print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)