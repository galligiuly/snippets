## Keras Sequential model 


#### Linear function to predict

y = 3*x + 2

```python
import numpy as np
import matplotlib.pyplot as plt


f= lambda x: 3*x+2
x_train = np.linspace(0, 100, 1000)
y_train = f(x_train)

x_test = np.linspace(100.01, 110, 100)
y_test = f(x_test)

plt.plot(x_train, y_train, 'r')
plt.plot(x_test, y_test, 'g')

import keras

from keras import layers 
from keras import models

m = models.Sequential()
m.add(layers.Dense(64, input_shape=(1,)))
m.add(layers.Dense(1))

m.summary()

from keras import optimizers
from keras import losses
from keras import metrics

m.compile(optimizer=optimizers.rmsprop(), loss=losses.mean_squared_error)
h = m.fit(x_train, y_train, batch_size=8, epochs=5, validation_split=0.2)

m.evaluate(x_test, y_test)

plt.plot(x_train, y_train, '-g')
plt.plot(x_test, m.predict(x_test), 'r-')
plt.plot(x_test, y_test, 'k-')

plot_metric(h, 'loss')

```

#### Parabollic function to predict

```python
f = lambda x: 2*x**2 + x +1
x_train = linspace(-100,100,1000)
y_train = f(x_train)

x_test = linspace(-110,-100.01,50)
y_test = f(x_test)

plt.plot(x_train, y_train,'r')
plt.plot(x_test, y_test, 'g')

m = models.Sequential()

m.add(layers.Dense(64, input_shape=(1,), activation='relu'))
m.add(layers.Dense(32, activation='relu'))
#m.add(layers.Dense(16, activation='relu'))
m.add(layers.Dense(1))

m.summary()

m.compile(optimizer=optimizers.rmsprop(), loss=losses.mean_squared_error)

# Without normalization, it is very difficult to train the model
x_train_norm = (x_train-x_train.mean())/(x_train.std())
y_train_norm = (y_train-y_train.mean())/(y_train.std())

h = m.fit(x_train_norm, y_train_norm, batch_size=8, epochs=10, validation_split=.2)

plot_metric(h, 'loss')

plt.plot(x_train, y_train, 'g')
#plt.plot(x_test, m.predict(x_test), 'r')
plt.plot(x_test, m.predict((x_test-x_train.mean())/x_train.std())*y_train.std()+y_train.mean(), 'r')

```

#### Trigonometric functions to predict 

𝑦=cos(𝑥)

```python

f = lambda x: cos(x)
x_train = linspace(-2*np.pi, 1.5*np.pi,10000)
y_train = f(x_train)

x_test = linspace(1.51*pi,2.5*pi,50)
y_test = f(x_test)

plt.plot(x_train, y_train, 'r')
plt.plot(x_test, y_test, 'g')

m = models.Sequential()

m.add(layers.Dense(100, input_shape=(1,), activation='tanh'))
m.add(layers.Dense(90, activation='tanh'))
m.add(layers.BatchNormalization())
m.add(layers.Dense(80, activation='tanh'))
m.add(layers.BatchNormalization())
m.add(layers.Dense(70, activation='tanh'))
m.add(layers.Dropout(0.2))
m.add(layers.Dense(60, activation='tanh'))
m.add(layers.BatchNormalization())
m.add(layers.Dense(50, activation='tanh'))
m.add(layers.Dropout(0.2))
m.add(layers.Dense(40, activation='tanh'))
m.add(layers.BatchNormalization())
m.add(layers.Dense(30, activation='tanh'))
m.add(layers.Dense(20, activation='tanh'))
m.add(layers.BatchNormalization())
m.add(layers.Dense(10, activation='tanh'))
m.add(layers.BatchNormalization())
m.add(layers.Dense(5, activation='tanh'))
m.add(layers.Dense(1, activation='tanh'))

m.summary()

m.compile(optimizer=optimizers.rmsprop(), loss=losses.mean_squared_error)

# Without normalization, it is very difficult to train the model
x_train_norm = (x_train-x_train.mean())/(x_train.std())
y_train_norm = (y_train-y_train.mean())/(y_train.std())

h = m.fit(x_train_norm, y_train_norm, batch_size=128, epochs=50, validation_split=.2)

plot_metric(h, 'loss')

m.evaluate((x_test-x_train.mean())/x_train.std(), (y_test-y_train.mean())/y_train.std())

plt.plot(x_train, y_train, 'r')
plt.plot(x_train, m.predict(x_train_norm)*y_train.std()+y_train.mean(), '--')
plt.plot(x_test, m.predict((x_test-x_train.mean())/x_train.std())*y_train.std()+y_train.mean(), 'g')

```



