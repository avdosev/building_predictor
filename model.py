from tensorflow import keras

def conv_layer(input):
    cnv1 = keras.layers.Conv2D(10, 3)(input)
    cnv2 = keras.layers.Conv2D(20, 3)(cnv1)
    cnv3 = keras.layers.Conv2D(30, 3)(cnv2)
    cnv4 = keras.layers.Conv2D(40, 3)(cnv3)
    cnv_out = keras.layers.GlobalAvgPool2D()(cnv4)
    return cnv_out

def get_model(class_count):
    input = keras.layers.Input(shape=(10, 10, 1))
    cnv_out = conv_layer(input)
    embeding = keras.layers.Dense(20, activation='relu')(cnv_out)
    out = keras.layers.Dense(class_count, activation='softmax')(embeding)


    return keras.Model(inputs=input, outputs=out)

if __name__ == "__main__":
    model = get_model(4)
    model.summary()
