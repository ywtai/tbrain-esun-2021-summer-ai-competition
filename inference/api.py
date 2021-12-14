from argparse import ArgumentParser
import base64
import datetime
import hashlib

import cv2
from flask import Flask
from flask import request
from flask import jsonify
import numpy as np
from PIL import Image

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications import EfficientNetB7

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

####### PUT YOUR INFORMATION HERE #########
CAPTAIN_EMAIL = 'will.nien@quantatw.com'  #
SALT = 'my_quanta_salt'                   #
###########################################

#Ensemble method 1
VGG16_model_enabled = 0
OCR_LSTM_model_enabled = 0

#Ensemble method 2 (maximum count of enabled model is 3)
DenseNet_model_enabled = 1
Xception_model_enabled = 1
GoogleNet_model_enabled = 1
EfficientNet_model_enabled = 1


def count_area_pixel(img, start_pixel, img_high):
    pixel_cnt = 0
    for i in range(start_pixel, start_pixel + 64):
        for j in range(img_high):
            if img[j][i] == 0:
                pixel_cnt += 1
    return pixel_cnt


def search_text_position(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    H, W = thresh.shape
    max_pixel_area = 0
    max_pixel_pos = 0

    for i in range(W - 64):
        tmp = count_area_pixel(thresh, i, H)
        if tmp >= max_pixel_area:
            max_pixel_area = tmp
            max_pixel_pos = i

    img = img[0:H, max_pixel_pos:max_pixel_pos + 64]

    c_H, c_W, channel = img.shape
    top = round(c_H / 7)
    bot = round(c_H / 7 * 6)
    img_top = img[0:top, 0:c_W]
    img_bot = img[bot:c_H, 0:c_W]
    is_line_threshold = 10
    minLineLength = 20
    maxLineGap = 10

    edges = cv2.Canny(img_top, 50, 250)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, is_line_threshold, minLineLength, maxLineGap)

    try:
        lines = lines[:, 0, :]
        #print(len(lines))
        for x1, y1, x2, y2 in lines:
            cv2.line(img_top, (x1, y1), (x2, y2), (255, 255, 255), 3)
    except Exception as e:
        pass
        #print("This top img didn't detect any line")

    edges = cv2.Canny(img_bot, 50, 250)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, is_line_threshold, minLineLength, maxLineGap)

    try:
        lines = lines[:, 0, :]
        #print(len(lines))
        for x1, y1, x2, y2 in lines:
            cv2.line(img_bot, (x1, y1), (x2, y2), (255, 255, 255), 3)
    except Exception as e:
        pass
        #print("This bottom img didn't detect any line")

    return img


def ocr_image_preprocess(image):
    image = cv2.resize(image, (64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image.astype(np.float32)
    image = np.expand_dims(image, 2)
    image = np.expand_dims(image, 0)

    return image


def VGG16_image_preprocess(image):
    image = cv2.resize(image, (64, 64))
    image = image.astype(np.float32)
    image = np.array([image]).reshape(1, 64, 64, 3)
    image = image / 255

    return image


def DenseNet_image_preprocess(image):
    image = cv2.resize(image, (64, 64))

    #gray
    image = Image.fromarray(image).convert('L')
    image = np.expand_dims(image, 2)
    image = np.concatenate((image, image, image),-1)
    image = image.astype(np.float32)

    image = np.array([image]).reshape(1, 64, 64, 3)
    image = image / 255

    return image


def Xception_image_preprocess(image):
    image = cv2.resize(image, (71, 71))

    #gray
    image = Image.fromarray(image).convert('L')
    image = np.expand_dims(image, 2)
    image = np.concatenate((image, image, image),-1)
    image = image.astype(np.float32)

    image = np.array([image]).reshape(1, 71, 71, 3)
    image = image / 255

    return image


def GoogleNet_image_preprocess(image):
    image = cv2.resize(image, (75, 75))

    #gray
    image = Image.fromarray(image).convert('L')
    image = np.expand_dims(image, 2)
    image = np.concatenate((image, image, image),-1)
    image = image.astype(np.float32)

    image = np.array([image]).reshape(1, 75, 75, 3)
    image = image / 255

    return image


def EfficientNet_image_preprocess(image):
    image = cv2.resize(image, (64, 64))

    #gray
    image = Image.fromarray(image).convert('L')
    image = np.expand_dims(image, 2)
    image = np.concatenate((image, image, image),-1)
    image = image.astype(np.float32)

    image = np.array([image]).reshape(1, 64, 64, 3)
    image = image / 255

    return image


def generate_server_uuid(input_string):
    """ Create your own server_uuid.

    @param:
        input_string (str): information to be encoded as server_uuid
    @returns:
        server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string + SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid


def base64_to_binary_for_cv2(image_64_encoded):
    """ Convert base64 to numpy.ndarray for cv2.

    @param:
        image_64_encode(str): image that encoded in base64 string format.
    @returns:
        image(numpy.ndarray): an image.
    """
    img_base64_binary = image_64_encoded.encode("utf-8")
    img_binary = base64.b64decode(img_base64_binary)
    image = cv2.imdecode(np.frombuffer(img_binary, np.uint8), cv2.IMREAD_COLOR)
    return image


def predict(image, model_name):
    """ Predict your model result.

    @param:
        image (numpy.ndarray): an image.
    @returns:
        prediction (str): a word.
    """

    ####### PUT YOUR MODEL INFERENCING CODE HERE #######
    if model_name is 'ocr':
        logits = ocr_model(image)

        # prediction = '陳'
        decoded, log_prob = \
            tf.nn.ctc_beam_search_decoder(inputs=logits,
                                          sequence_length=np.asarray([(image.shape[1]) // 16], dtype=np.int32))
        dense_decoded = tf.sparse.to_dense(decoded[0], default_value=-1)

        pred = [words[j.numpy()] for j in dense_decoded[0] if j.numpy() != -1]
        prediction = pred[0]

    elif model_name is 'VGG16':
        VGG16_predict = VGG16_model.predict_classes(image)
        pred = words[VGG16_predict[0]]
        prediction = pred

    elif model_name is 'DenseNet':
        DenseNet_predict = DenseNet_model.predict(image)
        DenseNet_predict = DenseNet_predict.argmax(axis=-1)
        pred = words[DenseNet_predict[0]]
        prediction = pred

    elif model_name is 'Xception':
        Xception_predict = Xception_model.predict(image)
        Xception_predict = Xception_predict.argmax(axis=-1)
        pred = words[Xception_predict[0]]
        prediction = pred

    elif model_name is 'GoogleNet':
        GoogleNet_predict = GoogleNet_model.predict(image)
        GoogleNet_predict = GoogleNet_predict.argmax(axis=-1)
        pred = words[GoogleNet_predict[0]]
        prediction = pred

    elif model_name is 'EfficientNet':
        EfficientNet_predict = EfficientNet_model.predict(image)
        EfficientNet_predict = EfficientNet_predict.argmax(axis=-1)
        pred = words[EfficientNet_predict[0]]
        prediction = pred


    # print(prediction)

    ####################################################
    if _check_datatype_to_string(prediction):
        return prediction


def _check_datatype_to_string(prediction):
    """ Check if your prediction is in str type or not.
        If not, then raise error.

    @param:
        prediction: your prediction
    @returns:
        True or raise TypeError.
    """
    if isinstance(prediction, str):
        return True
    raise TypeError('Prediction is not in string type.')


def voting_4(ans_1, ans_2, ans_3, ans_4):
    if ans_1 != ans_2 and ans_1 != ans_3 and ans_1 != ans_4 and ans_2 != ans_3 and ans_2 != ans_4 and ans_3 != ans_4:
        answer = "isnull"
    elif ans_1 == ans_2 and ans_2 == ans_3 and ans_3 == ans_4:
        answer = ans_1
    elif ans_1 == ans_2 and ans_1 == ans_3:
        answer = ans_1
    elif ans_1 == ans_2 and ans_1 == ans_4 :
        answer = ans_1
    elif ans_2 == ans_3 and ans_2 == ans_4:
        answer = ans_2
    elif ans_3 == ans_4 and ans_3 == ans_1:
        answer = ans_3
    else:
        answer = "isnull"

    return answer


def voting_3(ans_1, ans_2, ans_3):
    if ans_1 != ans_2 and ans_1 != ans_3 and ans_2 != ans_3:
        answer = "isnull"
    elif ans_1 == ans_2 and ans_1 == ans_3 and ans_2 == ans_3:
        answer = ans_1
    elif ans_1 == ans_2:
        answer = ans_1
    elif ans_1 == ans_3:
        answer = ans_1
    elif ans_2 == ans_3:
        answer = ans_2
    else:
        answer = "isnull"

    return answer


def voting_2(ans_1, ans_2):
    if ans_1 != ans_2:
        answer = "isnull"
    elif ans_1 == ans_2:
        answer = ans_1
    return answer

@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API. """
    data = request.get_json(force=True)

    # 自行取用，可紀錄玉山呼叫的 timestamp
    esun_timestamp = data['esun_timestamp']

    # 取 image(base64 encoded) 並轉成 cv2 可用格式
    image_64_encoded = data['image']
    image = base64_to_binary_for_cv2(image_64_encoded)

    # to save ori image
    #imagekeep = image.copy()

    # text focus
    image = search_text_position(image)
    # cv2.imwrite('text_focus.jpg', image)

    if OCR_LSTM_model_enabled == 1:
        ocr_image = ocr_image_preprocess(image)

    if VGG16_model_enabled == 1:
        VGG16_image = VGG16_image_preprocess(image)

    if DenseNet_model_enabled == 1:
        DenseNet_image = DenseNet_image_preprocess(image)

    if Xception_model_enabled == 1:
        Xception_image = Xception_image_preprocess(image)

    if GoogleNet_model_enabled == 1:
        GoogleNet_image = GoogleNet_image_preprocess(image)

    if EfficientNet_model_enabled == 1:
        EfficientNet_image = EfficientNet_image_preprocess(image)

    t = datetime.datetime.now()
    ts = str(int(t.utcnow().timestamp()))
    # print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL + ts)

    try:
        if OCR_LSTM_model_enabled == 1:
            ocr_answer = predict(ocr_image, 'ocr')
            #if no ensemble, assign the answer first
            answer = ocr_answer

        if VGG16_model_enabled == 1:
            VGG16_answer = predict(VGG16_image, 'VGG16')
            # if no ensemble, assign the answer first
            answer = VGG16_answer

        if DenseNet_model_enabled == 1:
            DenseNet_answer = predict(DenseNet_image, 'DenseNet')
            # if no ensemble, assign the answer first
            answer = DenseNet_answer

        if Xception_model_enabled == 1:
            Xception_answer = predict(Xception_image, 'Xception')
            # if no ensemble, assign the answer first
            answer = Xception_answer

        if GoogleNet_model_enabled == 1:
            GoogleNet_answer = predict(GoogleNet_image, 'GoogleNet')
            # if no ensemble, assign the answer first
            answer = GoogleNet_answer

        if EfficientNet_model_enabled == 1:
            EfficientNet_answer = predict(EfficientNet_image, 'EfficientNet')
            # if no ensemble, assign the answer first
            answer = EfficientNet_answer


        #ensemble method 2
        if DenseNet_model_enabled == 1 and Xception_model_enabled == 1 and GoogleNet_model_enabled == 1 and EfficientNet_model_enabled == 1:
            answer = voting_4(DenseNet_answer, Xception_answer, GoogleNet_answer, EfficientNet_answer)
            print(DenseNet_answer + ' : ' + Xception_answer + ' : ' + GoogleNet_answer +  ' : ' + EfficientNet_answer + ' = ' + answer)
        elif DenseNet_model_enabled == 1 and Xception_model_enabled == 1 and GoogleNet_model_enabled == 1 and EfficientNet_model_enabled == 0:
            answer = voting_3(DenseNet_answer, Xception_answer, GoogleNet_answer)
            print(DenseNet_answer + ' : ' + Xception_answer + ' : ' + GoogleNet_answer + ' = ' + answer)
        elif DenseNet_model_enabled == 0 and Xception_model_enabled == 1 and GoogleNet_model_enabled == 1 and EfficientNet_model_enabled == 1:
            answer = voting_3(EfficientNet_answer, Xception_answer, GoogleNet_answer)
            print(EfficientNet_answer + ' : ' + Xception_answer + ' : ' + GoogleNet_answer + ' = ' + answer)
        elif DenseNet_model_enabled == 1 and Xception_model_enabled == 1 and GoogleNet_model_enabled == 0 and EfficientNet_model_enabled == 1:
            answer = voting_3(DenseNet_answer, Xception_answer, EfficientNet_answer)
            print(DenseNet_answer + ' : ' + Xception_answer + ' : ' + EfficientNet_answer + ' = ' + answer)
        elif DenseNet_model_enabled == 1 and Xception_model_enabled == 0 and GoogleNet_model_enabled == 1 and EfficientNet_model_enabled == 1:
            answer = voting_3(DenseNet_answer, GoogleNet_answer, EfficientNet_answer)
            print(DenseNet_answer + ' : ' + GoogleNet_answer + ' : ' + EfficientNet_answer + ' = ' + answer)

        elif DenseNet_model_enabled == 1 and Xception_model_enabled == 1 and GoogleNet_model_enabled == 0 and EfficientNet_model_enabled == 0:
            answer = voting_2(DenseNet_answer, Xception_answer)
            print(DenseNet_answer + ' : ' + Xception_answer + ' = ' + answer)
        elif DenseNet_model_enabled == 1 and Xception_model_enabled == 0 and GoogleNet_model_enabled == 0 and EfficientNet_model_enabled == 1:
            answer = voting_2(DenseNet_answer, EfficientNet_answer)
            print(DenseNet_answer + ' : ' + EfficientNet_answer + ' = ' + answer)
        elif DenseNet_model_enabled == 1 and Xception_model_enabled == 0 and GoogleNet_model_enabled == 1 and EfficientNet_model_enabled == 0:
            answer = voting_2(DenseNet_answer, GoogleNet_answer)
            print(DenseNet_answer + ' : ' + GoogleNet_answer + ' = ' + answer)
        elif DenseNet_model_enabled == 0 and Xception_model_enabled == 1 and GoogleNet_model_enabled == 1 and EfficientNet_model_enabled == 0:
            answer = voting_2(Xception_answer, GoogleNet_answer)
            print(Xception_answer + ' : ' + GoogleNet_answer + ' = ' + answer)
        elif DenseNet_model_enabled == 0 and Xception_model_enabled == 1 and GoogleNet_model_enabled == 0 and EfficientNet_model_enabled == 1:
            answer = voting_2(Xception_answer, EfficientNet_answer)
            print(Xception_answer + ' : ' + EfficientNet_answer + ' = ' + answer)
        elif DenseNet_model_enabled == 0 and Xception_model_enabled == 0 and GoogleNet_model_enabled == 1 and EfficientNet_model_enabled == 1:
            answer = voting_2(GoogleNet_answer, EfficientNet_answer)
            print(Xception_answer + ' : ' + EfficientNet_answer + ' = ' + answer)

        #ensemble method 1
        if OCR_LSTM_model_enabled == 1 and VGG16_model_enabled == 1:
            if ocr_answer == VGG16_answer:
                answer = ocr_answer
            else:
                answer = "isnull"
            print(ocr_answer + ' : ' + VGG16_answer + ' = ' + answer)


    except TypeError as type_error:
        # You can write some log...
        raise type_error
    except Exception as e:
        # You can write some log...
        raise e
    # server_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    server_timestamp = int(datetime.datetime.now().timestamp())

    # to save ori image into imagekeep folder
    #cv2.imwrite('./imagekeep/' + str(server_timestamp) + '_' + answer + '.jpg', imagekeep)

    return jsonify({'esun_uuid': data['esun_uuid'],
                    'server_uuid': server_uuid,
                    'answer': answer,
                    'server_timestamp': server_timestamp})


if __name__ == "__main__":
    class_num = 800

    if OCR_LSTM_model_enabled == 1:
        ocr_model = tf.saved_model.load('models/ocr_model')

    if VGG16_model_enabled == 1:
        # create graph and load weight
        vgg16 = VGG16(include_top=False, input_shape=(64, 64, 3))
        VGG16_model = Sequential(vgg16.layers)

        for layer in VGG16_model.layers[:15]:
            layer.trainable = False

        VGG16_model.add(Flatten())
        VGG16_model.add(Dense(256, activation='relu'))
        VGG16_model.add(Dropout(0.5))
        VGG16_model.add(Dense(class_num, activation='softmax'))

        VGG16_model.compile(
            loss='categorical_crossentropy',
            optimizer=SGD(lr=1e-4, momentum=0.9),
            metrics=['accuracy']
        )
        VGG16_model.load_weights('models/VGG16_dataset_acc66.h5')

    if DenseNet_model_enabled == 1:
        # create graph and load weight
        base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

        x = base_model.output
        x = GlobalAveragePooling2D(data_format='channels_last')(x)
        # x = Dropout(0.5)(x)
        predictions = Dense(class_num, activation='softmax')(x)
        DenseNet_model = Model(inputs=base_model.input, outputs=predictions)
        DenseNet_model.load_weights('models/DENSENET_TRANSFER.h5')

    if Xception_model_enabled == 1:
        # create graph and load weight
        base_model = Xception(weights='imagenet', include_top=False, input_shape=(71, 71, 3))

        x = base_model.output
        x = GlobalAveragePooling2D(data_format='channels_last')(x)
        # x = Dropout(0.5)(x)
        predictions = Dense(class_num, activation='softmax')(x)
        Xception_model = Model(inputs=base_model.input, outputs=predictions)
        Xception_model.load_weights('models/XCEPTION_TRANSFER.h5')

    if GoogleNet_model_enabled == 1:
        # create graph and load weight
        base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(75, 75, 3))

        x = base_model.output
        x = GlobalAveragePooling2D(data_format='channels_last')(x)
        # x = Dropout(0.5)(x)
        predictions = Dense(class_num, activation='softmax')(x)
        GoogleNet_model = Model(inputs=base_model.input, outputs=predictions)
        GoogleNet_model.load_weights('models/GOOGLENET_TRANSFER.h5')

    if EfficientNet_model_enabled == 1:
        # create graph and load weight
        base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

        x = base_model.output
        x = GlobalAveragePooling2D(data_format='channels_last')(x)
        # x = Dropout(0.5)(x)
        predictions = Dense(class_num, activation='softmax')(x)
        EfficientNet_model = Model(inputs=base_model.input, outputs=predictions)
        EfficientNet_model.load_weights('models/EFFICIENTNET_TRANSFER.h5')

    f = open('training data dic.txt', encoding='utf_8_sig')
    words = f.read().splitlines()

    arg_parser = ArgumentParser(
        usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    )
    arg_parser.add_argument('-p', '--port', default=8080, help='port')
    arg_parser.add_argument('-d', '--debug', default=True, help='debug')
    options = arg_parser.parse_args()

    app.run(host='0.0.0.0', debug=options.debug, port=options.port)
