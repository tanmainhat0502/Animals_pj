import os
import shutil
import yaml
from sklearn.model_selection import train_test_split
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_dataset(config_path='configs/dataset.yaml'):
    """
    Chia dataset thành train/val/test dựa trên cấu hình trong config_path.
    
    Args:
        config_path (str): Đường dẫn tới file config dataset
    
    Raises:
        FileNotFoundError: Nếu file config hoặc thư mục raw_path không tồn tại
        ValueError: Nếu tỷ lệ split không hợp lệ
    """

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Không tìm thấy file config: {config_path}")
        raise
    

    raw_path = config['data']['raw_path']
    processed_path = config['data']['processed_path']
    test_size = config['data']['test_size']
    val_size = config['data']['val_size']
    
 
    if not (0 < test_size < 1 and 0 < val_size < 1 and test_size + val_size < 1):
        raise ValueError(f"Tỷ lệ split không hợp lệ: test_size={test_size}, val_size={val_size}")
    
  
    if not os.path.exists(raw_path):
        logger.error(f"Thư mục dữ liệu gốc không tồn tại: {raw_path}")
        raise FileNotFoundError(f"Thư mục dữ liệu gốc không tồn tại: {raw_path}")
    
   
    os.makedirs(processed_path, exist_ok=True)
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(processed_path, split), exist_ok=True)
    
  
    for class_name in os.listdir(raw_path):
        class_path = os.path.join(raw_path, class_name)
        if not os.path.isdir(class_path):
            logger.warning(f"Bỏ qua {class_path} vì không phải thư mục")
            continue
        
        img_list = [
            os.path.join(class_path, img_name)
            for img_name in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, img_name))
        ]
        
        if not img_list:
            logger.warning(f"Không tìm thấy ảnh trong {class_path}")
            continue
        
        train_val, test = train_test_split(img_list, test_size=test_size, random_state=42)
        train_size = 1 - test_size - val_size
        val_relative_size = val_size / (train_size + val_size)
        train, val = train_test_split(train_val, test_size=val_relative_size, random_state=42)
        
        for split, data in [('train', train), ('val', val), ('test', test)]:
            split_folder = os.path.join(processed_path, split, class_name)
            os.makedirs(split_folder, exist_ok=True)
            for img_path in data:
                try:
                    shutil.copy(img_path, split_folder)
                except Exception as e:
                    logger.error(f"Lỗi khi sao chép {img_path}: {str(e)}")

        logger.info(f"Class {class_name}: {len(train)} train, {len(val)} val, {len(test)} test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuẩn bị và chia dataset thành train/val/test")
    parser.add_argument('--config', type=str, default='configs/dataset.yaml', 
                        help='Đường dẫn tới file config dataset')
    args = parser.parse_args()
    
    split_dataset(args.config)