import os
import json
import logging
import datetime


def get_logger(log_dir, log_file, level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(log_file)
    logger.setLevel(level)
    
    if logger.handlers:
        logger.handlers = []
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.propagate = False
    
    return logger

def get_logger_downstream(log_dir, task_name, mode, percent):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(f"{task_name}_{mode}_{percent}")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers = []

    log_file = os.path.join(log_dir, f"{task_name}_{mode}_{percent}.log")
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.propagate = False
    
    return logger

def log_hyperparameters(logger, args):
    logger.info("=== Running with these settings ===")
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")
    logger.info("=================================")

def log_epoch_metrics(logger, epoch, train_loss, val_loss, val_metric, metric_name):
    logger.info(f"Epoch {epoch + 1} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, {metric_name}: {val_metric:.4f}")

def log_evaluation_results(logger, test_metric, metric_name, best_val_metric, best_epoch):
    logger.info("=== Evaluation result ===")
    logger.info(f"Best validation {metric_name}: {best_val_metric:.4f} at epoch {best_epoch}")
    logger.info(f"Test {metric_name}: {test_metric:.4f}")
    logger.info("============================")

def log_confusion_matrix(logger, cm, class_names):
    logger.info("=== Confusion Matrix ===")
    header = "".join([f"{name:^10}" for name in class_names])
    logger.info(f"{'':^10}{header}")
    
    for i, row in enumerate(cm):
        row_str = "".join([f"{val:^10.2f}" if isinstance(val, float) else f"{val:^10d}" for val in row])
        logger.info(f"{class_names[i]:^10}{row_str}")
    logger.info("=========================")

def save_metrics_to_json(log_dir, task_name, mode, percent, metrics_dict):
    json_path = os.path.join(log_dir, f"metrics_{task_name}_{mode}_{percent}.json")

    metrics_dict['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
