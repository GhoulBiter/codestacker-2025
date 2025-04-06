import os
import sys
import re
from datetime import datetime
from loguru import logger
import pymongo
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from the .env file.
load_dotenv()

# Global configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# MongoDB config from environment
MONGO_USERNAME = os.getenv("MONGO_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGO_HOST = os.getenv("MONGO_HOST", "8vjx8lp.mongodb.net")
MONGO_DATABASE = os.getenv("MONGO_DATABASE", "crime_classification")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION_MLOPS", "mlops_logs")
DISABLE_SSL = True

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Remove default Loguru sinks
logger.remove()

# Add a console sink with color formatting
logger.add(
    sys.stdout,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    level=LOG_LEVEL,
    colorize=True,
)

# Add a file sink for local log files
file_format = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{name}:{function}:{line} - {message}"
)
logger.add(
    "logs/train_{time}.log",
    rotation="500 MB",
    retention="30 days",
    level=LOG_LEVEL,
    compression="zip",
    format=file_format,
)

# Create a global ThreadPoolExecutor for asynchronous MongoDB writes.
executor = ThreadPoolExecutor(max_workers=2)


class SimpleMongoDBSink:
    """
    A custom Loguru sink that parses the log string using a regex,
    then inserts a document into MongoDB with the parsed fields asynchronously.
    """

    def __init__(self, collection_name):
        self.collection = None
        if all([MONGO_USERNAME, MONGO_PASSWORD, MONGO_HOST]):
            try:
                base_uri = (
                    f"mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}"
                    f"@{MONGO_DATABASE}.{MONGO_HOST}/"
                    "?retryWrites=true&w=majority&appName=Cluster0"
                )

                if DISABLE_SSL:
                    base_uri += "&ssl=false"

                self.client = pymongo.MongoClient(base_uri)
                self.db = self.client[MONGO_DATABASE]
                self.collection = self.db[collection_name]
                logger.debug(f"Connected to MongoDB collection: {collection_name}")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB Atlas: {e}")
                self.collection = None
        else:
            logger.warning("MongoDB credentials not provided; skipping MongoDB logging")
            self.collection = None

        # Regex pattern based on the file sink format.
        self.pattern = re.compile(
            r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| "
            r"(?P<level>\S+)\s*\| "
            r"(?P<module>[^:]+):(?P<function>[^:]+):(?P<line>\d+) - "
            r"(?P<message>.*)"
        )

    def __call__(self, formatted_message):
        if self.collection is None:
            return

        try:
            match = self.pattern.match(formatted_message)
            if match:
                log_data = match.groupdict()
                log_data["timestamp"] = datetime.strptime(
                    log_data["timestamp"], "%Y-%m-%d %H:%M:%S.%f"
                )
                log_data["environment"] = ENVIRONMENT

                # Offload the insertion to a background thread.
                executor.submit(self.collection.insert_one, log_data)
            else:
                print("Failed to parse log message:", formatted_message)
        except Exception as e:
            print("Error writing log to MongoDB:", e)


# Add the MongoDB sink if credentials are present.
if all([MONGO_USERNAME, MONGO_PASSWORD, MONGO_HOST]):
    logger.add(SimpleMongoDBSink(MONGO_COLLECTION), level=LOG_LEVEL, format=file_format)
else:
    logger.warning(
        "MongoDB credentials not provided; logs will not be stored in MongoDB"
    )


def get_logger(name=None):
    """
    Optional helper to retrieve a logger bound with a specific name.
    Usage:
        from your_module import get_logger
        my_logger = get_logger("my_module")
        my_logger.info("Hello from my_module!")
    """
    if name:
        return logger.bind(name=name)
    return logger
