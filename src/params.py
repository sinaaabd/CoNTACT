from yacs.config import CfgNode as CN

_cfg = CN()
''' Some parameters related to different input features '''
#Add them Here...

''' Settings for Modular Co-Attention '''
_cfg.CORE = CN()
#HIDDEN_SIZE must be dividable to MULTI_HEAD
_cfg.CORE.HIDDEN_SIZE = 128 #512
_cfg.CORE.DROPOUT_R = 0.1
_cfg.CORE.MULTI_HEAD = 8
#WARNING: HIDDEN_SIZE_HEAD = HIDDEN_SIZE / MULTI_HEAD
_cfg.CORE.HIDDEN_SIZE_HEAD = 16 #64
#WARNING: FF_SIZE = HIDDEN_SIZE * 4
_cfg.CORE.FF_SIZE = 512 #2048
_cfg.CORE.LAYER = 8
_cfg.CORE.FLAT_MLP_SIZE = 512
_cfg.CORE.FLAT_GLIMPSES = 1
_cfg.CORE.FLAT_OUT_SIZE = 1024



''' Settings for Drugs '''
_cfg.DRUG = CN()
_cfg.DRUG.MAX_NUM_ATOMS = 60 #290
_cfg.DRUG.ATOM_INPUT_FEATURES_DIM = 75
_cfg.DRUG.ATOM_EMBEDDING_DIM = 128
_cfg.DRUG.PADDING = True
_cfg.DRUG.HIDDEN_LAYERS_DIMS = [128, 128, 128]


''' Settings for Targets '''
_cfg.TARGET = CN()
_cfg.TARGET.PADDING = True
_cfg.TARGET.EMBEDDING_DIMS = 128
_cfg.TARGET.CNN_KERNEL_SIZES = [10, 25, 28, 35]
_cfg.TARGET.CNN_OUTPUT_CHANNEL_DIM = 128


''' Settings of the model and its optimizer '''
_cfg.SETTINGS = CN()
_cfg.SETTINGS.GPU_ID = 0
_cfg.SETTINGS.MAX_EPOCH = 100 #100
_cfg.SETTINGS.BATCH_SIZE = 64 #64
_cfg.SETTINGS.NUM_WORKERS = 0
_cfg.SETTINGS.NUM_CLASS = 1
_cfg.SETTINGS.LR = 1e-4 
_cfg.SETTINGS.USE_LD = True
_cfg.SETTINGS.LR_DECAY = 0.9
_cfg.SETTINGS.DECAY_INTERVAL = 10
_cfg.SETTINGS.WEIGHT_DECAY = 1e-4
_cfg.SETTINGS.SEED = 42
_cfg.SETTINGS.DOTATT_HIDDEN_DIM = 32


''' Result '''
_cfg.RESULT = CN()
_cfg.RESULT.OUTPUT_DIR = r"../output/result/"
_cfg.RESULT.SAVE_MODEL = True

def get_cfg():
    return _cfg.clone()
