import logging
import time
from functools import wraps
from pyspark.sql import SparkSession
from pyspark import SparkConf

def get_logger(name: str) -> logging.Logger:
    """
    配置并获取标准化日志对象。
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def timeit(logger=None):
    """
    计算函数执行时间的装饰器，支持传入日志对象。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            log = logger if logger else get_logger(func.__name__)
            log.info(f"Method '{func.__name__}' executed in {elapsed_time:.4f} seconds.")
            
            return result
        return wrapper
    return decorator

def init_spark(app_name: str = "DistributedClustering", local_cores: str = "*", executor_memory: str = "4g") -> SparkSession:
    """
    初始化 SparkSession，包含常用性能调优配置。
    """
    conf = SparkConf() \
        .setAppName(app_name) \
        .setMaster(f"local[{local_cores}]") \
        .set("spark.executor.memory", executor_memory) \
        .set("spark.driver.memory", executor_memory) \
        .set("spark.sql.shuffle.partitions", "200") \
        .set("spark.default.parallelism", "200") \
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    get_logger("SparkUtils").info(f"SparkSession '{app_name}' initialized with local[{local_cores}].")
    return spark
