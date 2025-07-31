# scripts/experiments_lstm.py
experiments = [
    {
        "name": "baseline",
        "SEQ_LEN": 60,
        "HIDDEN_SIZE": 64,
        "NUM_LAYERS": 2,
        "LR": 0.001,
        "EPOCHS": 50,
        "COUNT": 300,
        "INPUT_SIZE": 1,
        "OUTPUT_SIZE": 1
    },
    {
        "name": "long_seq_high_hidden",
        "SEQ_LEN": 90,
        "HIDDEN_SIZE": 128,
        "NUM_LAYERS": 2,
        "LR": 0.001,
        "EPOCHS": 50,
        "COUNT": 300,
        "INPUT_SIZE": 1,
        "OUTPUT_SIZE": 1
    },
    {
        "name": "drop_lr",
        "SEQ_LEN": 60,
        "HIDDEN_SIZE": 64,
        "NUM_LAYERS": 2,
        "LR": 0.0005,
        "EPOCHS": 50,
        "COUNT": 300,
        "INPUT_SIZE": 1,
        "OUTPUT_SIZE": 1
    },
    {
        "name": "multi_feature_test",
        "SEQ_LEN": 60,
        "HIDDEN_SIZE": 64,
        "NUM_LAYERS": 2,
        "LR": 0.001,
        "EPOCHS": 50,
        "COUNT": 300,
        "INPUT_SIZE": 4,
        "OUTPUT_SIZE": 1
    },
    {
        "name": "deep_lstm",
        "SEQ_LEN": 60,
        "HIDDEN_SIZE": 64,
        "NUM_LAYERS": 4,
        "LR": 0.001,
        "EPOCHS": 50,
        "COUNT": 300,
        "INPUT_SIZE": 1,
        "OUTPUT_SIZE": 1
    },

]
