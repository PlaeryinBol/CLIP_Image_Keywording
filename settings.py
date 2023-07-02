# Base
IMAGES_DIR = './data/images'  # folder with images for image keywording/search
VOCABULARY = './data/stock_keywords_vocabulary.json'  # json-file with keywords vocabulary
OUTPUT_DIR = './data/target_images'  # folder for saving images (result of image search)
VOCAB_FEATURES = './data/vocab_features.pt'  # pytorch features file for keywords
IMAGE_FEATURES = './data/image_features.pt'  # pytorch features file for images

# Image Keywording ---------------------------------------------------------------------
TEXT_MODEL = 'clip-ViT-B-32-multilingual-v1'  # text encoder
IMAGE_MODEL = 'clip-ViT-B-32'  # image encoder
DEVICE = 'cuda'  # cpu/cuda
CACHE_FOLDER = './data'  # folder for data files and models
BATCH_SIZE = 32  # batch size for image features extraction
TOP_K = 10  # number of keywords per image
PROMPT_TEMPLATE = 'a photo of {TARGET}'  # prompt template for embedding process
CREATE_KEYWORDING_DF = True  # create keywords df or not

# Image Search -------------------------------------------------------------------------
TARGET_KEYWORDS = ['flowers', 'trees', 'autumn', 'food']  # query list
VOCAB_SYNONYMS = True  # find synonyms in vocabulary or take from SYNONYMS_DICT
VOCAB_SYNONYMS_COUNT = 10
SYNONYMS_DICT = {
    'flowers': ["blossoms", "blooms", "flora", "plants"],
    'trees': ["forest", "woods", "groves"],
    'autumn': ["fall", "harvest", "season"],
    'food': ["cuisine", "meals", "dishes", "fare"]
}
