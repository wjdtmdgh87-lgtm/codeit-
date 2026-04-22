import torch
import torch.nn as nn

class SafeCustomFocalLoss(nn.Module):
    """Focal Loss(초점 손실 함수) 커스텀 클래스"""
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """초기 세팅"""
        super().__init__()
        self.loss_fcn = loss_fcn  # 기존 채점 방식(보통 BCE Loss 사용)
        self.gamma = gamma  # '쉬운 예제(Easy Example)'에 대한 손실(Loss) 가중치를 얼마나 줄일지(조절할지) 결정하는 하이퍼파라미터
        self.alpha = alpha  # 희귀한 데이터(알약)에 가중치를 더 주는 비율
        self.reduction = getattr(loss_fcn, 'reduction', 'none')  # AI 모델 점수를 '평균' 또는 '총합' 두 가지 방식 중 하나 기억해둠. 
        self.loss_fcn.reduction = 'none'  # '자동 합산 기능' 임시로 꺼두기('none')
        
    def forward(self, pred, target):
        """
        순전파: 최종 초점 손실(focal_loss) 반환

        Args:
            pred: AI 모델 예측값
            target: 정답(target)
            
        Returns: 
            focal_loss: 최종 초점 손실
        """
        
        loss = self.loss_fcn(pred, target)  # 기본 채점표(loss_fcn) 틀린 만큼 원래 벌점(loss) 계산
        pred_prob = torch.sigmoid(pred)  # 시그모이드 함수(값 0~1) - AI 모델 예측값(pred) 0% ~ 100% 사이 확률값 변경
        # p_t: '정답 맞출 확률'.
        # 실제 정답이 1(알약)이면 예측 확률 그대로 쓰고, 정답이 0(배경)이면 (1 - 예측 확률)을 씁니다.
        p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
        
        # alpha_factor: 데이터 개수 불균형을 잡아주는 보너스 점수.
        # 진짜 알약(true=1)에는 alpha(0.25)를 곱하고, 배경(true=0)에는 1-alpha(0.75)를 곱해 밸런스 맞춘다.
        alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
        
        # (100% - 맞출 확률)의 gamma 제곱.
        # 만약 AI 모델이 이미 99% 확신하는 쉬운 문제(배경)라면, 이 값이 0에 가까워져서 벌점이 아예 없어진다.
        modulating_factor = (1.0 - p_t) ** self.gamma
        focal_loss = alpha_factor * modulating_factor * loss  # 최종 초점 손실 계산
        
        # 처음에 기억해둔 방식대로 AI 모델 최종 초점 손실 리턴.
        if self.reduction == 'mean': return focal_loss.mean()  # 평균
        elif self.reduction == 'sum': return focal_loss.sum()  # 총합
        return focal_loss  # 합치지 않고 각각의 초점 손실 그대로 내보내기