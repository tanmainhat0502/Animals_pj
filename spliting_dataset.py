import os  
import shutil
from sklearn.model_selection import train_test_split

img_folder = os.path.join(os.getcwd(), "animals_dataset")
output = os.path.join(os.getcwd(), "splited_dataset")
os.makedirs(output, exist_ok=True)
train_folder = os.path.join(output, "train")
os.makedirs(train_folder, exist_ok=True)
test_folder = os.path.join(output, "test")
os.makedirs(test_folder, exist_ok=True)


for class_path in os.listdir(img_folder):
    class_foler = os.path.join(img_folder, class_path)
    img = []
    for img_path in os.listdir(class_foler):
        item = os.path.join(class_foler, img_path)
        img.append(item)
    train, test = train_test_split(img, test_size=0.2)
    train_class_ouput_folder = os.path.join(train_folder, class_path)
    os.makedirs(train_class_ouput_folder, exist_ok=True)
    test_class_out_folder = os.path.join(test_folder, class_path)
    os.makedirs(test_class_out_folder, exist_ok=True)
    for item in train:
        shutil.copy(item, train_class_ouput_folder)
    for item in test:
        shutil.copy(item, test_class_out_folder)
    print(len(train))
    print(len(test))
