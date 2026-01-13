import logging 
from pathlib import Path

def setup_logging(
        full_path = 'logs/project_logs.log', 
        root_level = logging.DEBUG, 
        file_level = logging.DEBUG, 
        stream_level = logging.WARNING, 
        filemode = 'w'
        ):

    '''Function for set logging. If you need common log file, change full path argument'''
    
    log_path = Path(full_path)
    root_logger = logging.getLogger()
    root_logger.setLevel(root_level)

    formatter = logging.Formatter(
        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S'
    )

    # For file
    file_handler = logging.FileHandler(
        log_path,
        encoding = 'utf-8',
        mode = filemode
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    # For console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(stream_level)
    console_handler.setFormatter(formatter)

    if not root_logger.handlers:
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    