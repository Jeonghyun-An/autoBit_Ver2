# LSTM 모델 학습용 설정값 (한 곳에서 관리)

SEQ_LEN = 60
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LR = 0.001
EPOCHS = 50
COUNT = 300  # fetch_upbit_data 개수
INPUT_SIZE = 1
OUTPUT_SIZE = 1
MODEL_PATH = "model_lstm.pt"
SCALER_PATH = "scaler_lstm.pkl"