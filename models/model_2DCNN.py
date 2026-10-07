import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input

def cnn_2D(X, num_classes):    
    inputs = Input(shape=(X.shape[1:]))
    
    # Shallow Path
    c1 = Conv2D(16, activation='relu', kernel_size=(3,3), padding="same")(inputs)
    #p1 = AveragePooling2D(pool_size=2)(c1)
    c2 = Conv2D(32, activation='relu', kernel_size=(3,3), padding="same")(c1)
    
    flat = Flatten()(c2)
    
    out = Dense(108, activation='relu')(flat)

    predict = Dense(num_classes,activation="softmax")(out)


    model = tf.keras.Model(inputs=[inputs], outputs=predict)
    
    model.compile(optimizer='adam',#SGD(learning_rate=0.001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy']) 
    
    return model

