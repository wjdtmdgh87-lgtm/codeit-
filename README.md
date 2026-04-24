데이터 폴더에 이미지(train,test의 png파일)와 라벨 파일(txt) 넣어주시고 main.py 실행하면 됩니다.

# 데이터 분할 (fold txt 생성)
python main.py --mode data

# 학습
python main.py --mode train

# 예측
python main.py --mode predict --source data/images/test/

# 데이터 분할 + 학습 한 번에(예측은 별도입니다.)
python main.py --mode all
