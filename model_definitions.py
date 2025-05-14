import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet201, InceptionResNetV2, Xception, InceptionV3, VGG19, ResNet101V2

# ---------------- CBAM Attention ----------------

def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    shared_layer_one = layers.Dense(channel // ratio,
                                    activation='relu',
                                    kernel_initializer='he_normal',
                                    use_bias=True,
                                    bias_initializer='zeros')
    shared_layer_two = layers.Dense(channel,
                                   kernel_initializer='he_normal',
                                   use_bias=True,
                                   bias_initializer='zeros')

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)

    return layers.Multiply()([input_feature, cbam_feature])

def spatial_attention(input_feature, kernel_size=7):
    def get_output_shape(input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], 1)

    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True),
                            output_shape=get_output_shape)(input_feature)
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True),
                            output_shape=get_output_shape)(input_feature)
    concat = layers.Concatenate(axis=3)([avg_pool, max_pool])

    cbam_feature = layers.Conv2D(filters=1,
                                kernel_size=kernel_size,
                                strides=1,
                                padding='same',
                                activation='sigmoid',
                                kernel_initializer='he_normal',
                                use_bias=False)(concat)

    return layers.Multiply()([input_feature, cbam_feature])

def cbam_block(input_feature, ratio=8, kernel_size=7):
    cbam_feature = channel_attention(input_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature, kernel_size)
    return cbam_feature

# ---------------- Deep Attention ----------------

def deep_channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    shared_layer_one = layers.Dense(channel // ratio,
                                   activation='relu',
                                   kernel_initializer='he_normal',
                                   use_bias=True,
                                   bias_initializer='zeros')
    shared_layer_two = layers.Dense(channel // ratio // 2,
                                   activation='relu',
                                   kernel_initializer='he_normal',
                                   use_bias=True,
                                   bias_initializer='zeros')
    shared_layer_three = layers.Dense(channel,
                                     kernel_initializer='he_normal',
                                     use_bias=True,
                                     bias_initializer='zeros')

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    avg_pool = shared_layer_three(avg_pool)

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    max_pool = shared_layer_three(max_pool)

    da_feature = layers.Add()([avg_pool, max_pool])
    da_feature = layers.Activation('sigmoid')(da_feature)

    return layers.Multiply()([input_feature, da_feature])

def deep_spatial_attention(input_feature, kernel_size=7):
    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(input_feature)
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True))(input_feature)
    concat = layers.Concatenate(axis=3)([avg_pool, max_pool])

    da_feature = layers.Conv2D(filters=16,
                              kernel_size=kernel_size,
                              strides=1,
                              padding='same',
                              activation='relu',
                              kernel_initializer='he_normal',
                              use_bias=False)(concat)

    da_feature = layers.Conv2D(filters=1,
                              kernel_size=kernel_size,
                              strides=1,
                              padding='same',
                              activation='sigmoid',
                              kernel_initializer='he_normal',
                              use_bias=False)(da_feature)

    return layers.Multiply()([input_feature, da_feature])

def deep_attention_block(input_feature, ratio=8, kernel_size=7):
    da_feature = deep_channel_attention(input_feature, ratio)
    da_feature = deep_spatial_attention(da_feature, kernel_size)
    return da_feature

# ---------------- Model Architectures ----------------

def build_inceptionresnetv2_with_cbam(num_classes, input_shape=(224,224,3)):
    base_model = InceptionResNetV2(include_top=False, weights=None, input_shape=input_shape)
    x = base_model.output
    x = cbam_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    model = keras.Model(inputs=base_model.input, outputs=outputs)
    return model

def build_custom_cnn_with_deep_attention(num_classes, input_shape=(224,224,3)):
    activation = 'relu'
    dropout_rate = 0.3
    l2_reg = 1e-4
    num_filters = 64

    def act_layer(x):
        if activation == 'leaky_relu':
            return layers.LeakyReLU()(x)
        else:
            return layers.Activation(activation)(x)

    inputs = keras.Input(shape=input_shape)

    # First block
    x = layers.Conv2D(num_filters, (3,3), padding='same', use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = act_layer(x)
    x = layers.Conv2D(num_filters, (3,3), padding='same', use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = act_layer(x)
    x = deep_attention_block(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Second block
    x = layers.Conv2D(num_filters*2, (3,3), padding='same', use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = act_layer(x)
    x = layers.Conv2D(num_filters*2, (3,3), padding='same', use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = act_layer(x)
    x = deep_attention_block(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Third block
    x = layers.Conv2D(num_filters*4, (3,3), padding='same', use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = act_layer(x)
    x = layers.Conv2D(num_filters*4, (3,3), padding='same', use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = act_layer(x)
    x = deep_attention_block(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation=activation,
                     kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = keras.Model(inputs, outputs)
    return model

def build_xception_with_cbam(num_classes, input_shape=(224,224,3)):
    base_model = Xception(include_top=False, weights=None, input_shape=input_shape)
    x = base_model.output
    x = cbam_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    model = keras.Model(inputs=base_model.input, outputs=outputs)
    return model
