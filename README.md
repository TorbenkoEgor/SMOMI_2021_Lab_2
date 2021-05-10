## Лабораторная работа №2

---

## train_enb0_wo_weights.py

Модель использует EfficientNetB0 с случайной инициализацией весов. Learning rate = 0.001.

build_model:

    inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
    outputs = tf.keras.applications.EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

Цвета

![Image alt](https://github.com/TorbenkoEgor/SMOMI_2021_Lab_2/blob/main/graphs/train1_color.png)

Метрика точности
 
![Image alt](https://github.com/TorbenkoEgor/SMOMI_2021_Lab_2/blob/main/graphs/train1_acc.png)

Метрика функции потерь

![Image alt](https://github.com/TorbenkoEgor/SMOMI_2021_Lab_2/blob/main/graphs/train1_loss.png)

### Вывод


Эта реализация нейронной сети не обучилась. Это связано в первую очередь с тем, что тут используется EfficientNetB0 с большим количеством случайно инициализированных весов.

---

## train_enb0_imagenet_1.py

Модель использует EfficientNetB0 с весами предобученной сети на ImageNet. Learning rate = 0.001.

build_model

    inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
    x = tf.keras.applications.EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation = tf.keras.activations.softmax)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


Цвета

![Image alt](https://github.com/TorbenkoEgor/SMOMI_2021_Lab_2/blob/main/graphs/train2_color.png)

Метрики точности

![Image alt](https://github.com/TorbenkoEgor/SMOMI_2021_Lab_2/blob/main/graphs/train2_acc.png)

Метрика функции потерь

![Image alt](https://github.com/TorbenkoEgor/SMOMI_2021_Lab_2/blob/main/graphs/train2_loss.png)

### Вывод 

Эта реализация нейронной сети также не обучилась, однако имела более предсказуемое поведение на графиках. Причиной плохого обучения является высокий темп обучения.

---

# Вывод


Первая реализация нейронной сети не обучилась по причине того, что в ней используется EfficientNetB0 с большим количеством случайно инициализированных весов. Вторая реализация нейронной сети также не обучилась, однако поведение графиков стало более предсказуемое. Если подобрать более подходящий темп обучения, то сеть должна обучаться лучше.

---

# Дополнительные результаты

train_enb0_imagenet_2.py

Модель использует EfficientNetB0 с весами предобученной сети на ImageNet. Learning rate = 0.00001.

    inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
    x = tf.keras.applications.EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation = tf.keras.activations.softmax)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

Цвета

![Image alt](https://github.com/TorbenkoEgor/SMOMI_2021_Lab_2/blob/main/graphs/train3_color.png)

Метрики точности

![Image alt](https://github.com/TorbenkoEgor/SMOMI_2021_Lab_2/blob/main/graphs/train3_acc.png)

Метрика функции потерь

![Image alt](https://github.com/TorbenkoEgor/SMOMI_2021_Lab_2/blob/main/graphs/train3_loss.png)

Вывод

За счёт меньшего темпа обучения эта реализация нейронной сети обучилась лучше, чем train_enb0_imagenet_1.

---

train_enb0_imagenet_3.py

Модель использует EfficientNetB0 с весами предобученной сети на ImageNet. Learning rate = 0.000001.

build_model

    inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
    x = tf.keras.applications.EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation = tf.keras.activations.softmax)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

Цвета

![Image alt](https://github.com/TorbenkoEgor/SMOMI_2021_Lab_2/blob/main/graphs/train4_color.png)

Метрики точности

![Image alt](https://github.com/TorbenkoEgor/SMOMI_2021_Lab_2/blob/main/graphs/train4_acc.png)

Метрика функции потерь

![Image alt](https://github.com/TorbenkoEgor/SMOMI_2021_Lab_2/blob/main/graphs/train4_loss.png)

Вывод

Сеть обучилась медленнее, но точнее чем train_enb0_imagenet_2. К сожалению, из-за технических проблем, не получилось обучить сеть до конца.
