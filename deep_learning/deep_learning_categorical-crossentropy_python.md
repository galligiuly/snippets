## Categorical crossentropy

We use this function because we are trying to predict discrete classes

```python

from keras import models
from keras import layers

def build_model():
    m = models.Sequential()
    # The first layer will transform our matrices into vectors
    m.add(layers.Flatten(input_shape=(28,28)))
    # The Dense layer needs vectors as inputs
    m.add(layers.Dense(512, activation='relu'))
    m.add(layers.Dense(256, activation='relu'))
    m.add(layers.Dense(10, activation='softmax'))
    return m

from keras import optimizers
from keras import losses
from keras import metrics

m = build_model()

m.compile(
    optimizer=optimizers.rmsprop(),
    loss=losses.categorical_crossentropy,
    metrics=[metrics.categorical_accuracy]      
    )
    
m.summary()

h = m.fit(train_imgs_t, train_labels_t, epochs=80, batch_size=4096, validation_split=.2)

plot_metric(h, 'loss')

plot_metric(h, 'categorical_accuracy')

loss, acc = m.evaluate(test_imgs_t, test_labels_t)

test_imgs.shape[0]*(1-acc)  # number of images that will be missclassified

plot_mnist_image(2543, test_imgs, test_labels)

m.predict(test_imgs_t[2543:2544,])

np.argmax(m.predict(test_imgs_t[2543:2544,]))

````

