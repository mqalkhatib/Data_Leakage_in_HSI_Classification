
import tensorflow as tf

from tensorflow.keras.layers import Flatten, Dense, Input
def MLP(X, num_classes):
    inputs = Input(shape=(X.shape[1:]))

    flatenned = Flatten()(inputs)
    
    l1 = Dense(64, activation='relu')(flatenned)
    l2 = Dense(128, activation='relu')(l1)
    l3 = Dense(256, activation='relu')(l2)
    predict = Dense(num_classes,activation="softmax")(l3)
    model = tf.keras.Model(inputs=[inputs], outputs=predict)
    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])   
    return model
