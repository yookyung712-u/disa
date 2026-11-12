import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="청소년 우울감 분석 및 예측", page_icon="📊", layout="wide")

st.title("📊 청소년 우울감 경험률 분석 및 미래 예측")
st.caption("질병관리청 청소년건강행태조사 자료를 활용한 통계 분석 및 선형회귀 실습")

st.info(
    "이 웹앱은 설문에서 보고된 '우울감 경험률'을 분석합니다. "
    "의학적 우울증을 진단하거나 개인의 정신건강을 판정하는 도구가 아닙니다."
)

# 앱에 기본으로 포함한 2016~2025년 성별 우울감 경험률
sample = pd.DataFrame([
    [2016, "남학생", "전체", "전체", 20.9],
    [2017, "남학생", "전체", "전체", 20.3],
    [2018, "남학생", "전체", "전체", 21.1],
    [2019, "남학생", "전체", "전체", 22.2],
    [2020, "남학생", "전체", "전체", 20.1],
    [2021, "남학생", "전체", "전체", 22.4],
    [2022, "남학생", "전체", "전체", 24.2],
    [2023, "남학생", "전체", "전체", 21.4],
    [2024, "남학생", "전체", "전체", 23.1],
    [2025, "남학생", "전체", "전체", 21.7],
    [2016, "여학생", "전체", "전체", 30.5],
    [2017, "여학생", "전체", "전체", 30.3],
    [2018, "여학생", "전체", "전체", 33.6],
    [2019, "여학생", "전체", "전체", 34.6],
    [2020, "여학생", "전체", "전체", 30.7],
    [2021, "여학생", "전체", "전체", 31.4],
    [2022, "여학생", "전체", "전체", 33.5],
    [2023, "여학생", "전체", "전체", 30.9],
    [2024, "여학생", "전체", "전체", 32.5],
    [2025, "여학생", "전체", "전체", 29.9],
], columns=["Year", "Gender", "School_Level", "Grade", "Depression_Rate"])

REQUIRED = ["Year", "Gender", "School_Level", "Grade", "Depression_Rate"]
DETAIL_REQUIRED = ["Year", "Indicator", "Gender", "School_Level", "Value",
                   "Unit", "Change_vs_2024", "Change_Unit", "Note"]

with st.sidebar:
    st.header("데이터 설정")
    uploaded = st.file_uploader(
        "① 연도별 우울감 데이터 업로드 (선택)",
        type=["csv"],
        help="Year, Gender, School_Level, Grade, Depression_Rate 열이 있는 CSV"
    )
    detail_uploaded = st.file_uploader(
        "② 2025년 상세 건강지표 CSV 업로드 (선택)",
        type=["csv"],
        help="youth_health_2025_detailed.csv 파일을 올리세요."
    )
    st.caption("두 파일은 용도가 다르므로 각각의 업로드 칸에 넣어 주세요.")

# 연도별 우울감 데이터 읽기
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            st.error("연도별 데이터 CSV에 다음 열이 없습니다: " + ", ".join(missing))
            st.stop()
        st.success("연도별 우울감 데이터를 불러왔습니다.")
    except Exception as e:
        st.error(f"연도별 CSV를 읽는 중 오류가 발생했습니다: {e}")
        st.stop()
else:
    df = sample.copy()
    st.caption("연도별 데이터 업로드가 없어 앱에 포함된 2016~2025년 성별 예시 데이터를 사용합니다.")

# 2025년 상세 건강지표 CSV 읽기
detail_df = None
if detail_uploaded is not None:
    try:
        detail_df = pd.read_csv(detail_uploaded)
        missing_detail = [c for c in DETAIL_REQUIRED if c not in detail_df.columns]
        if missing_detail:
            st.error("상세 건강지표 CSV에 다음 열이 없습니다: " + ", ".join(missing_detail))
            detail_df = None
        else:
            st.success("2025년 상세 건강지표 CSV를 불러왔습니다.")
    except Exception as e:
        st.error(f"상세 건강지표 CSV를 읽는 중 오류가 발생했습니다: {e}")
        detail_df = None

# 연도별 데이터 정리
df = df[REQUIRED].copy()
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Depression_Rate"] = pd.to_numeric(df["Depression_Rate"], errors="coerce")
df = df.dropna(subset=["Year", "Depression_Rate"])
df["Year"] = df["Year"].astype(int)
for col in ["Gender", "School_Level", "Grade"]:
    df[col] = df[col].fillna("전체").astype(str)
df = df[df["Depression_Rate"].between(0, 100)]

if df.empty:
    st.error("분석할 수 있는 연도별 데이터가 없습니다. 연도와 우울감 경험률을 확인하세요.")
    st.stop()

st.subheader("1. 데이터 살펴보기")
c1, c2, c3 = st.columns(3)
c1.metric("데이터 행 수", f"{len(df):,}개")
c2.metric("조사 연도 범위", f"{df['Year'].min()}~{df['Year'].max()}년")
c3.metric("평균 우울감 경험률", f"{df['Depression_Rate'].mean():.1f}%")

with st.expander("연도별 우울감 데이터 표 보기"):
    st.dataframe(
        df.sort_values(["Year", "Gender", "School_Level", "Grade"]),
        use_container_width=True
    )
    st.download_button(
        "연도별 데이터 CSV 다운로드",
        df.to_csv(index=False).encode("utf-8-sig"),
        "youth_depression_data.csv",
        "text/csv"
    )

st.subheader("2. 연도별 우울감 경험률 변화")
genders = sorted(df["Gender"].unique().tolist())
selected_genders = st.multiselect("성별 선택", genders, default=genders)
trend = df[df["Gender"].isin(selected_genders)].copy()

if not trend.empty:
    trend_summary = trend.groupby(
        ["Year", "Gender"], as_index=False
    )["Depression_Rate"].mean()
    fig = px.line(
        trend_summary, x="Year", y="Depression_Rate", color="Gender",
        markers=True, title="연도별 우울감 경험률",
        labels={
            "Year": "조사 연도",
            "Depression_Rate": "우울감 경험률(%)",
            "Gender": "성별"
        }
    )
    fig.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("성별을 하나 이상 선택하세요.")

# 상세 건강지표는 별도 CSV를 업로드했을 때 표시
st.subheader("3. 2025년 상세 건강지표")
if detail_df is None:
    st.info(
        "상세 통계를 보려면 사이드바의 '② 2025년 상세 건강지표 CSV 업로드' 칸에 "
        "youth_health_2025_detailed.csv 파일을 올려 주세요."
    )
else:
    detail_df = detail_df.copy()
    detail_df["Year"] = pd.to_numeric(detail_df["Year"], errors="coerce")
    detail_df["Value"] = pd.to_numeric(detail_df["Value"], errors="coerce")
    detail_df["Change_vs_2024"] = pd.to_numeric(
        detail_df["Change_vs_2024"], errors="coerce"
    )
    detail_df["Indicator"] = detail_df["Indicator"].astype(str)
    detail_df["Gender"] = detail_df["Gender"].fillna("전체").astype(str)
    detail_df["School_Level"] = detail_df["School_Level"].fillna("전체").astype(str)

    # 성별 상세 지표: 값과 전년 대비 변화를 표로 표시
    gender_detail = detail_df[
        (detail_df["Indicator"] != "우울감 경험률") |
        (detail_df["School_Level"] == "전체")
    ].copy()
    st.markdown("**성별 건강지표**")
    st.dataframe(
        gender_detail[
            ["Indicator", "Gender", "Value", "Unit",
             "Change_vs_2024", "Change_Unit", "Note"]
        ].rename(columns={
            "Indicator": "건강지표",
            "Gender": "성별",
            "Value": "2025년 값",
            "Unit": "단위",
            "Change_vs_2024": "전년 대비 변화",
            "Change_Unit": "변화 단위",
            "Note": "설명"
        }),
        use_container_width=True,
        hide_index=True
    )

    # 우울감 경험률의 학교급별 전년 대비 변화 폭
    school_change = detail_df[
        (detail_df["Indicator"] == "우울감 경험률") &
        (detail_df["School_Level"].isin(["중학교", "고등학교"]))
    ].dropna(subset=["Change_vs_2024"])

    st.markdown("**학교급별 우울감 경험률의 전년 대비 변화**")
    if not school_change.empty:
        school_change = school_change.copy()
        school_change["변화량(%p)"] = school_change["Change_vs_2024"]
        school_fig = px.bar(
            school_change,
            x="School_Level",
            y="변화량(%p)",
            text="변화량(%p)",
            title="2024년 대비 2025년 우울감 경험률 변화",
            labels={
                "School_Level": "학교급",
                "변화량(%p)": "전년 대비 변화(%p)"
            }
        )
        school_fig.update_traces(texttemplate="%{text:.1f}%p", textposition="outside")
        school_fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(school_fig, use_container_width=True)
        st.caption(
            "이 그래프는 2025년 실제 우울감 경험률이 아니라 전년 대비 변화 폭을 나타냅니다. "
            "제공된 자료에는 중학교 1.7%p 감소, 고등학교 2.4%p 감소만 제시되어 있습니다."
        )
    else:
        st.info("상세 CSV에 학교급별 우울감 경험률 변화 자료가 없습니다.")

    st.markdown("**학년별 비교**")
    st.info(
        "현재 업로드된 상세 CSV에는 학년별 우울감 경험률이 포함되어 있지 않습니다. "
        "학년별 실제 통계값을 확보하면 학년 비교 그래프를 추가할 수 있습니다."
    )

    with st.expander("업로드한 2025년 상세 통계 원본 보기"):
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

st.subheader("4. 학교급·학년별 실제 우울감 경험률 비교")
school_rate = df[df["School_Level"] != "전체"].groupby(
    "School_Level", as_index=False
)["Depression_Rate"].mean()

grade_rate = df[df["Grade"] != "전체"].groupby(
    "Grade", as_index=False
)["Depression_Rate"].mean()

col_a, col_b = st.columns(2)
with col_a:
    if not school_rate.empty:
        f = px.bar(
            school_rate, x="School_Level", y="Depression_Rate",
            title="학교급별 평균 우울감 경험률",
            labels={"School_Level": "학교급", "Depression_Rate": "우울감 경험률(%)"}
        )
        st.plotly_chart(f, use_container_width=True)
    else:
        st.info(
            "연도별 데이터에는 학교급별 실제 경험률이 없습니다. "
            "위의 2025년 상세 건강지표에서는 전년 대비 감소 폭을 별도로 확인할 수 있습니다."
        )

with col_b:
    if not grade_rate.empty:
        f = px.bar(
            grade_rate, x="Grade", y="Depression_Rate",
            title="학년별 평균 우울감 경험률",
            labels={"Grade": "학년", "Depression_Rate": "우울감 경험률(%)"}
        )
        st.plotly_chart(f, use_container_width=True)
    else:
        st.info("현재 데이터에는 학년별 실제 경험률이 없습니다. 학년별 통계값을 추가해야 합니다.")

st.subheader("5. 선형회귀 모델로 미래 경험률 예측")
st.write("독립 변수(X)는 조사 연도(Year), 종속 변수(y)는 우울감 경험률(Depression_Rate)입니다.")

gender_options = sorted(df["Gender"].unique().tolist())
school_options = sorted(df["School_Level"].unique().tolist())
grade_options = sorted(df["Grade"].unique().tolist())

p1, p2, p3 = st.columns(3)
with p1:
    model_gender = st.selectbox("모델에 사용할 성별", gender_options)
with p2:
    model_school = st.selectbox("모델에 사용할 학교급", school_options)
with p3:
    model_grade = st.selectbox("모델에 사용할 학년", grade_options)

model_df = df[
    (df["Gender"] == model_gender) &
    (df["School_Level"] == model_school) &
    (df["Grade"] == model_grade)
].groupby("Year", as_index=False)["Depression_Rate"].mean().sort_values("Year")

if len(model_df) < 5:
    st.error("선택한 조건의 데이터가 5개 연도 미만입니다. 최소 5개 연도 이상의 자료를 준비하세요.")
else:
    X = model_df[["Year"]]
    y = model_df["Depression_Rate"]
    split = max(3, int(len(model_df) * 0.7))
    if split >= len(model_df):
        split = len(model_df) - 1

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_pred)
    r2 = r2_score(y_test, test_pred) if len(y_test) >= 2 else np.nan

    m1, m2 = st.columns(2)
    m1.metric("MAE (평균 절대 오차)", f"{mae:.2f}%p")
    m2.metric("R² (결정계수)", "계산 불가" if np.isnan(r2) else f"{r2:.3f}")
    st.caption(
        f"시간 순서대로 앞의 {split}개 연도를 학습에 사용하고, "
        f"뒤의 {len(model_df)-split}개 연도를 평가에 사용했습니다."
    )

    future_year = st.slider(
        "예측할 연도",
        min_value=int(model_df["Year"].max()) + 1,
        max_value=int(model_df["Year"].max()) + 10,
        value=int(model_df["Year"].max()) + 5
    )
    future_value = float(model.predict(pd.DataFrame({"Year": [future_year]}))[0])
    st.metric(f"{future_year}년 선형회귀 예측값", f"{future_value:.1f}%")
    st.caption(
        "예측값은 과거 추세를 직선으로 연장한 통계적 추정치이며 "
        "실제 미래 결과를 보장하지 않습니다."
    )

    all_years = pd.DataFrame({
        "Year": np.arange(int(model_df["Year"].min()), future_year + 1)
    })
    all_years["Predicted"] = model.predict(all_years[["Year"]])

    chart = px.scatter(
        model_df, x="Year", y="Depression_Rate",
        title="실제 데이터와 선형회귀 예측선",
        labels={"Year": "조사 연도", "Depression_Rate": "우울감 경험률(%)"}
    )
    chart.add_scatter(
        x=all_years["Year"], y=all_years["Predicted"],
        mode="lines", name="선형회귀 예측선"
    )
    chart.add_scatter(
        x=X_test["Year"], y=test_pred,
        mode="markers", name="평가용 예측값"
    )
    chart.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(chart, use_container_width=True)

    comparison = pd.DataFrame({
        "Year": X_test["Year"].values,
        "실제 경험률(%)": y_test.values,
        "예측 경험률(%)": np.round(test_pred, 2)
    })
    st.write("평가용 데이터의 실제값과 예측값")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "자료 출처: 질병관리청 청소년건강행태조사(https://www.kdca.go.kr/yhs/). "
    "상세 CSV에 제공되지 않은 학교급·학년별 실제 경험률은 임의로 만들어 넣지 않았습니다."
)
