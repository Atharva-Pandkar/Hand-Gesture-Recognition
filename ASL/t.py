def train_model():
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import ReduceLROnPlateau
    import pandas as pd

    train_df = pd.read_csv("C:\\Users\\athar\\PycharmProjects\\Final_Project\\ASL\\sign_mnist_train.csv")
    test_df = pd.read_csv("C:\\Users\\athar\\PycharmProjects\\Final_Project\\ASL\\sign_mnist_test.csv")

    y_train = train_df['label']
    y_test = test_df['label']
    del train_df['label']
    del test_df['label']

    from sklearn.preprocessing import LabelBinarizer
    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)

    x_train = train_df.values
    x_test = test_df.values

    x_train = x_train / 255
    x_test = x_test / 255

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(x_train)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5,
                                                min_lr=0.00001)

    model = Sequential()
    model.add(Conv2D(75, (3, 3), strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(Conv2D(50, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(Conv2D(25, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=24, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(datagen.flow(x_train, y_train, batch_size=128), epochs=20, validation_data=(x_test, y_test),
                        callbacks=[learning_rate_reduction])

    model.save('smnist.h5')

def  display():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import cv2
    import mediapipe as mp
    from keras.models import load_model
    import numpy as np
    import time
    import pandas as pd

    model = load_model("C:\\Users\\athar\\PycharmProjects\\Final_Project\\smnist.h5")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    _, frame = cap.read()

    h, w, c = frame.shape

    img_counter = 0
    analysis_frame = ''
    letter_pred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                  'V', 'W', 'X', 'Y']
    while True:
        _, frame = cap.read()

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            analysis_frame = frame
            showframe = analysis_frame

            cv2.imshow("Frame", showframe)

            frame_rgb_analysis = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
            result_analysis = hands.process(frame_rgb_analysis)
            hand_landmarks_analysis = result_analysis.multi_hand_landmarks
            if hand_landmarks_analysis:
                for hand_LMs_analysis in hand_landmarks_analysis:
                    x_max = 0
                    y_max = 0
                    x_min = w
                    y_min = h
                    for lm_analysis in hand_LMs_analysis.landmark:
                        x, y = int(lm_analysis.x * w), int(lm_analysis.y * h)
                        if x > x_max:
                            x_max = x
                        if x < x_min:
                            x_min = x
                        if y > y_max:
                            y_max = y
                        if y < y_min:
                            y_min = y
                    y_min -= 20
                    y_max += 20
                    x_min -= 20
                    x_max += 20

            analysis_frame = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
            analysis_frame = analysis_frame[y_min:y_max, x_min:x_max]
            analysis_frame = cv2.resize(analysis_frame, (28, 28))

            n_list = []
            rows, cols = analysis_frame.shape
            for i in range(rows):
                for j in range(cols):
                    k = analysis_frame[i, j]
                    n_list.append(k)

            datan = pd.DataFrame(n_list).T

            colname = []
            for val in range(784):
                colname.append(val)
            datan.columns = colname

            pixel_data = datan.values
            pixel_data = pixel_data / 255
            pixel_data = pixel_data.reshape(-1, 28, 28, 1)
            prediction = model.predict(pixel_data)
            pred_array = np.array(prediction[0])
            letter_prediction_dict = {letter_pred[i]: pred_array[i] for i in range(len(letter_pred))}
            pred_array_ordered = sorted(pred_array, reverse=True)
            high_1 = pred_array_ordered[0]
            high_2 = pred_array_ordered[1]
            high_3 = pred_array_ordered[2]
            
            for key, value in letter_prediction_dict.items():
                if value == high_1:
                    print("Predicted Character 1: ", key)
                    print('Confidence 1: ', 100 * value)
                elif value == high_2:
                    print("Predicted Character 2: ", key)
                    print('Confidence 2: ', 100 * value)
                elif value == high_3:
                    print("Predicted Character 3: ", key)
                    print('Confidence 3: ', 100 * value)
            time.sleep(5)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)
        hand_landmarks = result.multi_hand_landmarks
        if hand_landmarks:
            for hand_LMs in hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in hand_LMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                y_min -= 20
                y_max += 20
                x_min -= 20
                x_max += 20
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.imshow("Frame", frame)

    cap.release()
    cv2.destroyAllWindows()
