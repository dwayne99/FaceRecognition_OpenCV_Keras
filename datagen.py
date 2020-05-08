import os 
import shutil

images_path = 'images'
train_size = 0.8

if not os.path.isdir('data'):
    os.makedirs('data')
    os.makedirs('data/train')
    os.makedirs('data/test')

dest_train = 'data/train'
dest_test = 'data/test'

for name in os.listdir(images_path):
    if name not in os.listdir(dest_train) and name not in os.listdir(dest_test):
        os.makedirs('data/train/' + name.upper())
        os.makedirs('data/test/' + name.upper())

    name_path = os.path.join(images_path,name)
    num_files = len(os.listdir(name_path))
    train_count = int(num_files*train_size)
    for pic in os.listdir(name_path)[:train_count]:
        pic_path = os.path.join(name_path,pic)
        dest_path = dest_train + '/'+ name + '/'
        shutil.move(pic_path,dest_path)
    for pic in os.listdir(name_path):
        pic_path = os.path.join(name_path,pic)
        dest_path = dest_test + '/'+ name + '/'
        shutil.move(pic_path,dest_path)


