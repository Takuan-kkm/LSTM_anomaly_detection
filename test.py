import pickle

with open("train.sav", "rb") as f:
    train = pickle.load(f)

print(train[111])
