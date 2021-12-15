# TBrain E-SUN 2021 Summer AI Competition
# Handwriting Chinese Characters Recognition
![E-SUN Competition](https://github.com/ywtai/tbrain-esun-2021-summer-ai-competition/blob/main/image/tbrain.png?raw=true)

## Description
TBrain E-SUN 2021 Summer AI Competition is an AI competition for handwriting Chinese characters recognition. E-SUN provide dataset with 800 categories of single traditional Chinese character image. Besides, during the competition, we need to upload our model into server and use API to receive test images and return results.

***

## Result 
- **Accuracy: 89.4%**
- **Rank 43 in 468 teams**

***

## Instruction

### Install Conda
```bash
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash ./Anaconda3-2020.11-Linux-x86_64.sh
export PATH="/home/[USERNAME]/anaconda3/bin:$PATH"
```


### Setup
```bash
conda create --name tbrain python=3.7
conda activate tbrain
conda deactivate
```

### Install Packages
```bash
pip3 install -r requirements.txt
```

### Test Inference
```bash
cd tbrain-esun-2021-summer-ai-competition/inference
python api.py
time python send_request.py [-h | a.jpg | imagekeep | imagekeep_rotate]
```
- resize to 64 x 64
- color to gray.
- normalization by dividing 255.
- text focus by search_text_position


### Data Preprocessing
```bash
cd tbrain-esun-2021-summer-ai-competition/train
python preprocessing.py [-h | esun.tar | esun_less.tar] #tar including original images
```
- data augmentation by random rotation
- text focus by search_text_position
- resize to 64 x 64
- Output : train_data_clean64_64.tfrecords

### Train Models
```bash
cd tbrain-esun-2021-summer-ai-competition/train
time python VGG16.py
```
**Result:** 397m, Epoch 686/1000,  loss: 1.8077, accuracy: 0.5843,  val_loss: 2.3044, val_accuracy: 0.5341
**Requirement:** train_64_64.tfrecords (normalization by dividing 255, color to gray)

```bash
time python ocr_biLSTM.py
```
**Result:** 64m, Epoch 48/1000, train_accuracy = 0.994, train_loss = 0.029, val_accuracy = 0.496, val_loss = 3.525
**Requirement:** train_64_64.tfrecords and test_64_64.tfrecords (normalization by dividing 255, color to gray)

```bash
time python DenseNet_gray_3.py
```
**Result:** 927m, Epoch 143/1000, loss: 0.0228 - accuracy: 0.9930 - val_loss: 0.0019 - val_accuracy: 0.9993
**Requirement:** train_data_clean64_64.tfrecords (normalization by dividing 255, color to gray, rotate 90, trained by esun_0603_51380_ok.tar)

```bash
time python Xception_gray_3.py
```
**Result:** 459m, Epoch 107/1000, loss: 0.0164 - accuracy: 0.9953 - val_loss: 0.0031 - val_accuracy: 0.9991
**Requirement:** train_data_clean64_64.tfrecords (normalization by dividing 255, color to gray)

```bash
- time python GoogleNet_gray_3.py
```
**Result:** 1429m, Epoch 174/1000, loss: 0.0528 - accuracy: 0.9844 - val_loss: 0.0105 - val_accuracy: 0.9986
**Requirement:** train_data_clean64_64.tfrecords (normalization by dividing 255, color to gray, rotate 90)
```bash
time python EfficientNet_gray_3.py
```
**Result:** 1250m, Epoch 93/1000, loss: 0.0188 - accuracy: 0.9949 - val_loss: 9.3117e-04 - val_accuracy: 0.9998
**Requirement:** train_data_clean64_64.tfrecords (normalization by dividing 255, color to gray, rotate 90)

### Deploy Models
```bash
cd tbrain-esun-2021-summer-ai-competition/train
cp [*.h5 or ocr_model] ../inference/models
```

# Websites
- https://tbrain.trendmicro.com.tw/Competitions/Details/14
- https://aidea-web.tw/topic/8b1a4c64-c875-407a-8af9-292a8802d4d0
