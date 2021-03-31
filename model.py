from tensorflow import keras

def conv_layer(input):
    cnv1 = keras.layers.Conv2D(64, 3)(input)
    # batch1 = keras.layers.BatchNormalization()(maxpool1)
    cnv2 = keras.layers.Conv2D(32, 3)(cnv1)
    # batch2 = keras.layers.BatchNormalization()(maxpool2)
    cnv3 = keras.layers.Conv2D(64, 3)(cnv2)
    # batch3 = keras.layers.BatchNormalization()(maxpool3)
    cnv4 = keras.layers.Conv2D(32, 3)(cnv3)
    cnv_out = keras.layers.GlobalMaxPool2D()(cnv4)
    return cnv_out

def get_model(class_count, embeding_size=20):
    input = keras.layers.Input(shape=(11, 11, 1))
    cnv_out = conv_layer(input)
    embeding = keras.layers.Dense(embeding_size, activation='relu')(cnv_out)
    batch_emb = keras.layers.BatchNormalization()(embeding)
    out = keras.layers.Dense(class_count, activation='softmax')(batch_emb)

    return keras.Model(inputs=input, outputs=out)

if __name__ == "__main__":
    model = get_model(4)
    model.summary()
