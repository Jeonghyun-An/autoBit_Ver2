# LSTM 모델 학습용 설정값 (한 곳에서 관리)
# long_seq_high_hidden 실험 결과 적용(채택함)

SEQ_LEN = 90        # 시퀀스 길이
HIDDEN_SIZE = 128    # LSTM 은닉층 크기
NUM_LAYERS = 2      # LSTM 층 수
LR = 0.001          # 학습률
EPOCHS = 50         # 학습 에폭 수
COUNT = 300         # fetch_upbit_data 개수
INPUT_SIZE = 1      # 입력 크기 (종가 1개)
OUTPUT_SIZE = 1     # 출력 크기 (종가 1개)
MODEL_PATH = "model_lstm.pt"
SCALER_PATH = "scaler_lstm.pkl"