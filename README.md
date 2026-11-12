# 청소년 우울감 경험률 분석 및 미래 예측

## 실행
```bash
pip install -r requirements.txt
streamlit run main.py
```

## GitHub / Streamlit Cloud
1. 이 폴더의 `main.py`, `requirements.txt`, `youth_depression_data.csv`를 GitHub 저장소에 올립니다.
2. Streamlit Community Cloud에서 저장소를 연결합니다.
3. Main file path에 `main.py`를 입력하고 Deploy 합니다.

## CSV 필수 컬럼
Year, Gender, School_Level, Grade, Depression_Rate

- Year: 조사 연도(숫자)
- Gender: 남학생/여학생/전체
- School_Level: 중학교/고등학교/전체
- Grade: 중1~고3/전체
- Depression_Rate: 우울감 경험률(%)

기본 데이터는 질병관리청 2016~2025년 성별 우울감 경험률 공식 발표 수치입니다.
기본 데이터에는 학교급·학년별 상세값이 없으므로, 해당 비교를 하려면 공식 통계집에서 확인한 세부 수치를 같은 CSV 형식으로 추가해야 합니다.
