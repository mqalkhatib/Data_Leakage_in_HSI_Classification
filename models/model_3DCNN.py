
from tensorflow.keras.layers import Activation, Conv3D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential


def cnn_3D(img_list, num_class, lr=1e-3):
            shapeinput = img_list.shape[1:] + (1,) 
            
            clf = Sequential()

            clf.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=shapeinput))
            clf.add(Activation('relu'))
            clf.add(Conv3D(32, kernel_size=(3, 3, 3)))
            clf.add(Activation('relu'))
            clf.add(Conv3D(32, kernel_size=(3, 3, 3)))
            clf.add(Activation('relu'))


            clf.add(Flatten())
            clf.add(Dense(128))
            clf.add(Activation('relu'))
            clf.add(Dropout(0.3))
            clf.add(Dense(64))
            clf.add(Activation('relu'))
            clf.add(Dropout(0.2))
            clf.add(Dense(num_class, activation='softmax'))
            clf.compile(loss = "categorical_crossentropy", 
                        optimizer = 'adam', 
                        metrics = 'accuracy')
            return clf


