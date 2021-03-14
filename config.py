import json

__json = json.load(open('config.json', 'r'))

DATASET_DIR = __json.get('dataset_dir')
PREPROCESSING_DATA_DIR = __json.get('preprocessing_data_dir')
SUBMISSION_CSV = __json.get('submission_csv')

TEST_SIZE = __json.get('test_size')
INPUT_SHAPE = __json.get('input_shape')
BATCH_SIZE = __json.get('batch_size')

EPOCHS = __json.get('epochs')
GPU_CONF = __json.get('gpu')

REMOVE_PATH_VARS = __json.get('remove_path_vars')

REACTION_TYPES = [['ER_1keV', 'ER_3keV', 'ER_6keV', 'ER_10keV', 'ER_20keV', 'ER_30keV'],
                  ['NR_1keV', 'NR_3keV', 'NR_6keV', 'NR_10keV', 'NR_20keV', 'NR_30keV']]

if bool(REMOVE_PATH_VARS):
    import sys
    try:
        sys.path.remove('D:\\Projects\\TFModels\\models\\research')
        sys.path.remove('D:\\Projects\\TFModels\\models\\research\\slim')
        sys.path.remove('D:\\Projects\\TFModels\\models\\research\\object_detection')
    except:
        pass

DIST_MATRIX = None
SEED = 42
