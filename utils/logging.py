# utils/logging.py
import logging
from torch.utils.tensorboard import SummaryWriter
import os

def setup_logging(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    
    # File logger
    logging.basicConfig(
        filename=os.path.join(log_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(log_dir, 'tensorboard'))
    
    return logging.getLogger(__name__), writer

def close_logging(writer):
    writer.close()