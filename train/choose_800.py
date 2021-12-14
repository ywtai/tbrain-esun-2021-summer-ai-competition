import os
import random
import shutil
import tarfile

def tar_dir(path):
    cur_path = os.getcwd()
    os.chdir(path)
    tar = tarfile.open( path + '.tar', 'w:')
       
    for root, dirs, files in os.walk(path):
        for file_name in files:
            tar.add('train/' + file_name)
    
    tar.close()
    os.chdir(cur_path)

def choose_800():
    f = open('training data dic.txt', encoding='utf_8_sig')
    words = f.read().splitlines()
    
    ori_data_path = '/home/quanta/Desktop/project/quanta-ai-brain/esun_ocr/train/datasets/raw_data/esun_images/esun/train/'
    new_data_path = '/home/quanta/Desktop/project/quanta-ai-brain/esun_ocr/train/datasets/raw_data/esun_images/esun_800'
    
    if not os.path.exists(new_data_path):
        os.makedirs(new_data_path)
    else:
        shutil.rmtree(new_data_path)
    if not os.path.exists(new_data_path + '/train'):
        os.makedirs(new_data_path + '/train')
        
    
    
        
    data_list = os.listdir(ori_data_path)
    
    data_dict = {}
    
    for word in words:
        data_dict[word] = []
    
    for file in data_list:
        try:
            data_dict[file[-5]].append(int(file[0:-6]))
        except Exception as e:
            print(file)
            pass
    
    for word in words:
        max = len(data_dict[word])
        rand = random.randint(0, max-1)
        source = ori_data_path + str(data_dict[word][rand]) + '_' + word + '.jpg'
        destination = new_data_path + '/train/' + str(data_dict[word][rand]) + '_' + word + '.jpg'
        shutil.copyfile(source, destination)
        
    tar_dir(new_data_path)
    
        
if __name__ == '__main__':
    file_name = 'esun.tar'
    ori_data_path = '/home/quanta/Desktop/project/quanta-ai-brain/esun_ocr/train/datasets/raw_data/esun_images/esun/train/'
    if not os.path.exists(ori_data_path):
        tar = tarfile.open('datasets/raw_data/esun_images/' + file_name, "r:")
        tar.extractall()
        tar.close()
            
    choose_800()
        
        
        
        
        
        
