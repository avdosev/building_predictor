from tensorflow import keras

def conv_layer(input):
    cnv1 = keras.layers.Conv2D(27, 2, activation='sigmoid')(input)
    cnv2 = keras.layers.Conv2D(6, 2, activation='sigmoid')(cnv1)
    cnv3 = keras.layers.Conv2D(9, 2, activation='sigmoid')(cnv2)
    cnv4 = keras.layers.Conv2D(27, 2, activation='sigmoid')(cnv3)
    cnv_out = keras.layers.GlobalMaxPool2D()(cnv4)
    return cnv_out

def get_model(class_count, embeding_size=20, conv_size=5):
    input = keras.layers.Input(shape=(conv_size, conv_size, 1))
    input_info = keras.layers.Input(4 + 3)
    cnv_out = conv_layer(input)
    embeding_input = keras.layers.concatenate([cnv_out, input_info])
    embeding = keras.layers.Dense(embeding_size, activation='sigmoid')(embeding_input)
    batch_emb = keras.layers.BatchNormalization()(embeding)
    embeding2 = keras.layers.Dense(embeding_size, activation='sigmoid')(batch_emb)
    batch_emb2 = keras.layers.BatchNormalization()(embeding2)
    out = keras.layers.Dense(class_count, activation='softmax')(batch_emb2)

    return keras.Model(inputs=[input, input_info], outputs=out)

if __name__ == "__main__":
    model = get_model(4)
    model.summary()
