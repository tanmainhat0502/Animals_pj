# utils/data_utils.py
import os
from torch.utils.data import Dataset
from PIL import Image
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnimalsDataset(Dataset):
    """
    Custom Dataset class cho bài toán phân loại động vật.
    
    Args:
        root_dir (str): Thư mục chứa dữ liệu (train/val/test)
        transform (callable, optional): Các biến đổi áp dụng lên ảnh
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = []
        
        # Kiểm tra thư mục gốc
        if not os.path.exists(root_dir):
            logger.error(f"Thư mục dữ liệu không tồn tại: {root_dir}")
            raise FileNotFoundError(f"Thư mục dữ liệu không tồn tại: {root_dir}")
        
        # Load dữ liệu
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if not self.classes:
            logger.error(f"Không tìm thấy class nào trong {root_dir}")
            raise ValueError(f"Không tìm thấy class nào trong {root_dir}")
        
        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if os.path.isfile(img_path):
                    self.images.append(img_path)
                    self.labels.append(label)
                else:
                    logger.warning(f"Bỏ qua {img_path} vì không phải file")
        
        if not self.images:
            logger.error(f"Không tìm thấy ảnh nào trong {root_dir}")
            raise ValueError(f"Không tìm thấy ảnh nào trong {root_dir}")
        
        logger.info(f"Loaded {len(self.images)} images from {len(self.classes)} classes in {root_dir}")
    
    def __len__(self):
        """Trả về số lượng mẫu trong dataset."""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Lấy mẫu tại index idx.
        
        Args:
            idx (int): Index của mẫu
        
        Returns:
            tuple: (image, label) - ảnh sau khi transform và nhãn tương ứng
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} ngoài phạm vi dataset (size: {len(self)})")
        
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Lỗi khi mở ảnh {img_path}: {str(e)}")
            raise
        
        if self.transform is not None:
            try:
                image = self.transform(image)
            except Exception as e:
                logger.error(f"Lỗi khi áp dụng transform cho {img_path}: {str(e)}")
                raise
        
        return image, label
    
    def get_class_name(self, label):
        """Trả về tên class từ label."""
        if label < 0 or label >= len(self.classes):
            raise ValueError(f"Label {label} không hợp lệ")
        return self.classes[label]