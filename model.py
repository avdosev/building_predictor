from tensorflow import keras

def conv_layer(input):
    cnv1 = keras.layers.Conv2D(40, 3)(input)
    batch1 = keras.layers.BatchNormalization()(cnv1)
    cnv2 = keras.layers.Conv2D(10, 3)(batch1)
    batch2 = keras.layers.BatchNormalization()(cnv2)
    cnv3 = keras.layers.Conv2D(20, 3)(batch2)
    batch3 = keras.layers.BatchNormalization()(cnv3)
    cnv4 = keras.layers.Conv2D(10, 3)(batch3)
    cnv_out = keras.layers.GlobalMaxPool2D()(cnv4)
    return cnv_out

def get_model(class_count, embedeing_size=20):
    input = keras.layers.Input(shape=(11, 11, 1))
    cnv_out = conv_layer(input)
    embeding = keras.layers.Dense(20, activation='relu')(cnv_out)
    batch_emb = keras.layers.BatchNormalization()(embeding)
    out = keras.layers.Dense(class_count, activation='softmax')(batch_emb)

    return keras.Model(inputs=input, outputs=out)

if __name__ == "__main__":
    model = get_model(4)
    model.summary()
