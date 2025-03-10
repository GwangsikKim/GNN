## GNN을 활용한 P&ID 도면 관계성 학습 모델

### 📌 개요

이 프로젝트는 **Graph Neural Network (GNN)**을 활용하여 P&ID (Piping and Instrumentation Diagram) 도면에서 객체 간의 관계성을 분석하는 학습 모델을 구현하는 것을 목표로 합니다. 특히, Link Prediction 기법을 사용하여 도면 내 요소들 간의 숨겨진 관계를 예측합니다.

### 📁 프로젝트 구조

├── data/                 # 데이터셋 저장 폴더

├── models/               # 학습된 모델 저장 폴더

├── src/                  # 소스 코드 폴더

│   ├── dataset.py        # 데이터 로드 및 전처리 코드

│   ├── model.py          # GNN 모델 정의

│   ├── train.py          # 모델 학습 코드

│   ├── evaluate.py       # 모델 평가 코드

│   ├── utils.py          # 보조 함수 모음

├── notebooks/            # Jupyter Notebook 샘플 코드

├── README.md             # 프로젝트 설명 파일

### 📊 데이터셋

P&ID 도면의 객체 및 관계성을 그래프 데이터 형태로 변환하여 사용합니다.

🟢 노드(Node): 기기, 배관, 계측기 등의 개별 객체
🔵 엣지(Edge): 객체 간의 관계 (예: 배관이 연결된 기기, 센서와 모니터링 시스템 간의 관계 등)
🔴 속성(Feature): 객체의 속성 (예: 유형, 크기, 유량 등)

### 🛠️ 주요 기술

PyTorch Geometric (PyG): GNN 모델을 구현하고 학습하는 데 사용
DGL (Deep Graph Library): 대규모 그래프 데이터 처리 지원
Scikit-learn & Pandas: 데이터 전처리 및 분석
Matplotlib & NetworkX: 그래프 데이터 시각화

### 🏗️ 모델 아키텍처

이 프로젝트에서는 Link Prediction을 수행하기 위해 다음과 같은 GNN 모델을 사용합니다.
GraphSAGE: 노드 이웃 정보를 집계하여 학습하는 모델
GCN (Graph Convolutional Network): 그래프 데이터를 효율적으로 학습하는 컨볼루션 기반 모델
GAT (Graph Attention Network): 노드 간의 중요도를 다르게 반영하는 모델
모델은 주어진 P&ID 도면의 그래프 데이터를 학습하여, 새로운 노드 쌍 간의 관계가 존재할 확률을 예측합니다.

### 🚀 학습 과정

데이터 전처리: P&ID 도면을 그래프 구조로 변환하고, 학습/검증/테스트 데이터로 분할
모델 학습: GNN을 활용하여 관계 예측 모델 학습
평가 및 테스트: Precision, Recall, F1-score 등을 활용하여 모델 성능 평가

### ⚡ 실행 방법

1️⃣ 환경 설정
pip install -r requirements.txt

2️⃣ 데이터 전처리 및 그래프 생성
python src/dataset.py

3️⃣ 모델 학습
python src/train.py --model GCN --epochs 100

4️⃣ 모델 평가
python src/evaluate.py --model GCN

### 📌 결과 예시

P&ID 도면의 그래프 구조 예시:
[Pump] --- [Pipe] --- [Valve] --- [Sensor]
Link Prediction 결과 예시:
(노드 A, 노드 B) → 연결 가능성: 87%
(노드 C, 노드 D) → 연결 가능성: 45%

### 🎯 기대 효과

✅ P&ID 도면의 자동 분석을 통한 설계 검증 자동화
✅ 엔지니어링 작업의 효율성 향상 및 오류 감소
✅ 그래프 기반 AI 모델을 활용한 다양한 산업 응용 가능
