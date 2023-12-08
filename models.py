from tensorflow import keras as keras
from keras.layers import Conv2D, BatchNormalization, ReLU, Add, Input, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam
from keras import HyperParameters

class ResNet18Model:
    def __init__(self, input_shape=(224, 224, 3), classes=1000):
        self.input_shape = input_shape
        self.classes = classes
        self.hp = None

    def set_hyperparameters(self, hp):
        self.hp = hp

    def identity_block(self, x, filters, kernel_size=3, l2_reg=0.01):
        fx = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2(l2_reg))(x)
        fx = BatchNormalization()(fx)
        fx = ReLU()(fx)

        fx = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2(l2_reg))(fx)
        fx = BatchNormalization()(fx)

        x = Add()([x, fx])
        x = ReLU()(x)
        return x

    def convolutional_block(self, x, filters, kernel_size=3, stride=2, l2_reg=0.01):
        shortcut = x

        fx = Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_regularizer=l2(l2_reg))(x)
        fx = BatchNormalization()(fx)
        fx = ReLU()(fx)

        fx = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2(l2_reg))(fx)
        fx = BatchNormalization()(fx)

        shortcut_channels = shortcut.shape[-1]
        if stride != 1 or shortcut_channels != filters:
            shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same', kernel_regularizer=l2(l2_reg))(shortcut)
            shortcut = BatchNormalization()(shortcut)

        x = Add()([fx, shortcut])
        x = ReLU()(x)
        return x

    def build(self):
        if self.hp is None:
            self.hp = HyperParameters()

        l2_reg = self.hp.Float('l2_reg', min_value=1e-5, max_value=1e-2, sampling='LOG')
        inputs = Input(self.input_shape)

        x = Conv2D(64, 7, strides=2, padding='same', kernel_regularizer=l2(l2_reg))(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D(3, strides=2, padding='same')(x)

        num_filters = 64
        for _ in range(2):
            x = self.identity_block(x, num_filters, l2_reg=l2_reg)

        for _ in range(3):
            num_filters *= 2
            x = self.convolutional_block(x, num_filters, l2_reg=l2_reg)
            for _ in range(1):
                x = self.identity_block(x, num_filters, l2_reg=l2_reg)

        x = GlobalAveragePooling2D()(x)

        dropout_rate = self.hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        x = Dropout(dropout_rate)(x)

        outputs = Dense(self.classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)

        lr = self.hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model