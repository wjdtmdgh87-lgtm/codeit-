"""
tests/utils/callbacks.py
테스트 콜백함수(Callback Function) 유틸리티
대상 모델: "yolov8m.pt"

* 손실함수 - torch.nn.BCEWithLogitsLoss
참고: https://wikidocs.net/194978
"""

import SafeCustomFocalLoss  # FocalLoss

def on_train_start(trainer):
    """AI 모델 학습 시작될 때(on_train_start) 자동으로 실행되는 콜백 함수"""
    if getattr(trainer.model, 'criterion', None) is None:  # 손실함수(criterion) 없는 경우
        trainer.model.criterion = trainer.model.init_criterion()  # 기본 손실함수(criterion) 초기화
    # trainer.model.criterion.bce는 torch.nn.BCEWithLogitsLoss과 같다.
    # 이진 분류(Binary Classification) 손실함수(criterion) FocalLoss 커스텀 클래스 SafeCustomFocalLoss 변경 
    trainer.model.criterion.bce = SafeCustomFocalLoss(trainer.model.criterion.bce, gamma=1.5, alpha=0.25)