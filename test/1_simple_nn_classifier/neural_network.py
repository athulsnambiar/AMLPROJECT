import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense

x_train = np.random.rand(100000, 2)
side = 1


def square_circle(side: float) -> float:
    area_sq: float = side * side
    area_cr: float = area_sq / 2
    radius = math.sqrt(area_cr / math.pi)
    return radius


def inside_circle(row):
    length_sq = row[0] * row[0] + row[1] * row[1]
    if length_sq <= square_circle(1):
        return np.array([1,0])
    return np.array([0,1])


y_train = np.apply_along_axis(inside_circle, axis=1, arr=x_train)
print(y_train.shape)

model = Sequential()
model.add(Dense(units=10, activation='relu', input_dim=2))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=100)

x_test = np.random.rand(200, 2)
y_test = np.apply_along_axis(inside_circle, axis=1, arr=x_test)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
# classes = model.predict(x_test, batch_size=128)
print(loss_and_metrics)
