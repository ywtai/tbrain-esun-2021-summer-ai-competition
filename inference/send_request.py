import json
import requests
import time
import os
import cv2
import base64
import sys

#url = "http://34.87.3.253:8080/inference"
url = "http://127.0.0.1:8080/inference"

try:
    file_name = sys.argv[1]
except IndexError:
    file_name = '5.jpg'


if file_name == '-h':
    print("usage: python send_request.py [file or folder] \t\t\t>>>looking for the images in the path "
              "of datasets/raw_data/test_images/")
    print("example:\ntime python send_request.py 5.jpg")
    print("time python send_request.py imagekeep")
    print("time python send_request.py imagekeep_rotate")
    exit()


if file_name.find('.jpg') == -1:
    image_path = '../train/datasets/raw_data/test_images/' + file_name
    image_list = os.listdir(image_path)

    image_dict = []

    correct_count = 0
    total_count = 0
    ok_flag = 'OK'

    for file in image_list:
        if file.find('.jpg') > 0:
            total_count += 1
            label = file[-5]
            img = cv2.imread(image_path + '/' + file)
            string = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
            datas = {"esun_uuid": "123",
                     "esun_timestamp": time.time(),
                     "image": string,
                     "retry": 2
                     }
            files = json.dumps(datas)
            r = requests.post(url, data=files)
            #print(str(r.content, 'utf_8_sig'))
            y = json.loads(str(r.content, 'utf_8_sig'))

            if y['answer'] == label:
                correct_count += 1
                ok_flag = 'OK'
            else:
                ok_flag = 'NG'

            print(ok_flag + ' ' + label + ' : ' + y['answer'] + ' [ ' + str(correct_count) + ' / ' + str(total_count) + ']')

    print('Accuracy : ' + str(float(correct_count/total_count)))

else:
    img = cv2.imread('../train/datasets/raw_data/test_images/' + file_name)
    string = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
    datas = { "esun_uuid": "123",
              "esun_timestamp": time.time(),
              "image": string,
              "retry": 2
            }
    files = json.dumps(datas)
    r = requests.post(url, data=files)
    print(str(r.content, 'utf_8_sig'))




