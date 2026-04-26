<h1 align="center">🔍 PILL SIGHT 알약 탐지 프로그램</h1>

<p align="center">
  <b>  ▎ YOLO 기반 실시간 알약 탐지 및 분류 시스템       
  카메라 또는 이미지 입력으로 알약의 종류와 위치를
  자동으로 감지하고, 실험별 성능 지표를          
  체계적으로 추적·비교합니다. </b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B6B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

---

## 👥 팀원 구성

<table align="center">
  <tr>
    <td align="center" width="150">
      <img src="docs/profiles/Adam/profile.png" width="100" height="100" style="border-radius:50%"/><br/>
      <b>Adam</b><br/>
      <a href="https://github.com/Adam-1228">@Adam</a>
    </td>
    <td align="center" width="150">
      <img src="docs/profiles/Eastar0102/profile.png" width="100" height="100" style="border-radius:50%"/><br/>
      <b>Eastar0102</b><br/>
      <a href="https://github.com/Eastar0102">@Eastar0102</a>
    </td>
    <td align="center" width="150">
      <img src="docs/profiles/heewon02/profile.png" width="100" height="100" style="border-radius:50%"/><br/>
      <b>heewon02</b><br/>
      <a href="https://github.com/heewon02">@heewon02</a>
    </td>
    <td align="center" width="150">
      <img src="docs/profiles/minjaejeon/profile.png" width="100" height="100" style="border-radius:50%"/><br/>
      <b>minjaejeon</b><br/>
      <a href="https://github.com/minjaejeon0827">@minjaejeon</a>
    </td>
    <td align="center" width="150">
      <img src="docs/profiles/%EA%B8%B0%ED%95%98/profile.png" width="100" height="100" style="border-radius:50%"/><br/>
      <b>기하</b><br/>
      <a href="https://github.com/wenttoofar">@기하</a>
    </td>
    <td align="center" width="150">
      <img src="docs/profiles/%EB%AC%B4%EC%8B%AC/profile.png" width="100" height="100" style="border-radius:50%"/><br/>
      <b>무심</b><br/>
      <a href="https://github.com/wjdtmdgh87-lgtm">@무심</a>
    </td>
  </tr>
</table>

---

## 🛠 기술 스택

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLO-FF6B6B?style=flat-square&logo=yolo&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square"/>
  <img src="https://img.shields.io/badge/Ultralytics-00B4D8?style=flat-square"/>
</p>

---

## 📁 프로젝트 구조

```
📦 project-root
├── 📂 src/
│   ├── 📂 eval/               # 평가 모듈
│   │   ├── metrics.py         # YOLO 메트릭 정규화
│   │   ├── visualize.py       # 결과 시각화
│   │   ├── compare_yolo_runs.py  # 실험 비교
│   │   └── report_template.py    # 리포트 생성
│   └── 📂 utils/              # 유틸리티
│       ├── common.py
│       └── io_utils.py
├── 📂 config/
│   └── settings.py            # 전역 설정
├── 📂 outputs/
│   └── yolo/                  # YOLO 출력 결과
├── 📂 docs/
│   └── profiles/              # 팀원 프로필
└── main.py
```

---

## 🚀 시작하기

### 설치

```bash
git clone https://github.com/wjdtmdgh87-lgtm/codeit-.git
cd codeit-
pip install -r requirements.txt
```

### 실행

```bash
python main.py
```

---

## 📊 주요 기능

| 기능 | 설명 |
|------|------|
| **메트릭 정규화** | mAP50, mAP50-95, Precision, Recall 등 YOLO 지표 표준화 |
| **실험 비교** | 여러 YOLO 실험 결과를 나란히 비교 |
| **시각화** | 학습 곡선 및 성능 지표 시각화 |
| **리포트 생성** | 실험 결과 자동 리포트 생성 |

---

<p align="center">
  Made with ❤️ HAPPY 6 TEAM
</p>
