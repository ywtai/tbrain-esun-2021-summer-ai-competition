Install Conda
==============
- curl -O https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
- bash ./Anaconda3-2020.11-Linux-x86_64.sh
- export PATH="/home/[USERNAME]/anaconda3/bin:$PATH"


Setup
=============
- conda create --name tbrain python=3.7
- conda activate tbrain
- conda deactivate


Install Packages
=============
- pip3 install -r requirements.txt


Gitlab
=============
- git clone https://github.com/ywtai/tbrain-esun-2021-summer-ai-competition.git
- git push -u origin master


Test Inference
=============
- cd tbrain-esun-2021-summer-ai-competition/inference
- python api.py
- time python send_request.py [-h | a.jpg | imagekeep | imagekeep_rotate]
    1. resize to 64 x 64
    2. color to gray.
    3. normalization by dividing 255.
    4. text focus by search_text_position


Data Preprocessing
=============
- cd tbrain-esun-2021-summer-ai-competition/train
- jupyter notebook preprocessing.ipynb or python preprocessing.py [-h | esun.tar | esun_less.tar] #tar including original images
    1. data augmentation by random rotation
    2. text focus by search_text_position
    3. resize to 64 x 64
- Output : train_data_clean64_64.tfrecords

Train Models
=============
- cd tbrain-esun-2021-summer-ai-competition/train
- time python VGG16.py (397m, Epoch 686/1000,  loss: 1.8077 - accuracy: 0.5843 - val_loss: 2.3044 - val_accuracy: 0.5341)
    1. requires train_64_64.tfrecords
    2. normalization by dividing 255.
    3. color to gray.
- time python ocr_biLSTM.py (64m, Epoch 48/1000, train_accuracy = 0.994, train_loss = 0.029, val_accuracy = 0.496, val_loss = 3.525)
    1. requires train_64_64.tfrecords and test_64_64.tfrecords
    2. normalization by dividing 255.
    3. color to gray.
- time python DenseNet_gray_3.py (927m, Epoch 143/1000, loss: 0.0228 - accuracy: 0.9930 - val_loss: 0.0019 - val_accuracy: 0.9993)
    1. requires train_data_clean64_64.tfrecords
    2. normalization by dividing 255.
    3. color to gray.
    4. rotate 90.
    5. trained by esun_0603_51380_ok.tar
- time python Xception_gray_3.py (459m, Epoch 107/1000, loss: 0.0164 - accuracy: 0.9953 - val_loss: 0.0031 - val_accuracy: 0.9991)
    1. requires train_data_clean64_64.tfrecords
    2. normalization by dividing 255.
    3. color to gray.
- time python GoogleNet_gray_3.py (1429m, Epoch 174/1000, loss: 0.0528 - accuracy: 0.9844 - val_loss: 0.0105 - val_accuracy: 0.9986)
    1. requires train_data_clean64_64.tfrecords
    2. normalization by dividing 255.
    3. color to gray.
    4. rotate 90.
- time python EfficientNet_gray_3.py (1250m, Epoch 93/1000, loss: 0.0188 - accuracy: 0.9949 - val_loss: 9.3117e-04 - val_accuracy: 0.9998)
    1. requires train_data_clean64_64.tfrecords
    2. normalization by dividing 255.
    3. color to gray.
    4. rotate 90.

Deploy Models
=============
- cd tbrain-esun-2021-summer-ai-competition/train
- cp [*.h5 or ocr_model] ../inference/models

Websites
=============
- https://tbrain.trendmicro.com.tw/Competitions/Details/14
- https://aidea-web.tw/topic/8b1a4c64-c875-407a-8af9-292a8802d4d0
