import streamlit as st
import pandas as pd
import plotly.express as px

# Sayfa ayarı
st.set_page_config(layout="wide", page_title="Araç Fiyat Tahmin Dashboard")

# Başlık ve tanıtım
st.title("🚗 Araç Fiyat Tahmin Dashboard")
st.markdown("""
Bu dashboard, kullanılmış araçların geçmiş verilerine dayanarak fiyat tahmini yapmayı amaçlar.
Veri seti, araçların marka, model, motor tipi, yakıt türü, hasar durumu gibi bilgilerini içermektedir.
Makine öğrenimi modelleriyle eğitilerek test seti üzerinde fiyat tahmini gerçekleştirilmiştir.
""")


# Verileri yükle
@st.cache_data
def load_data():
    train = pd.read_csv("train.csv")
    pred_df = pd.read_csv("model_predictions.csv")
    result_df = pd.read_csv("model_metrics.csv")
    test_pred = pd.read_csv("test_predictions.csv")
    return train, pred_df, result_df, test_pred


train_df, pred_df, result_df, test_pred = load_data()

# Sekmeler
eda_tab, clean_tab, model_tab, eval_tab, submission_tab = st.tabs(
    ["📊 Veri Keşfi", "🧼 Veri Temizleme", "🔎 Model Tahminleri", "📈 Performans", "📤 Test Sonuçları"])

with eda_tab:
    st.header("📊 Veri Keşfi")
    st.markdown("""
    Bu bölümde veri setini tanıyor ve temel istatistikleri inceliyoruz.
    Hangi markalar daha fazla satılmış, fiyat dağılımı nasıl, yakıt türleri nelerdir gibi sorulara yanıt arıyoruz.
    """)

    st.markdown("""
    **Veri seti bilgileri:**
    - Toplam kayıt sayısı: {}
    - Sütun sayısı: {}
    - Hedef değişken: `price`
    - Kategorik öznitelikler: `brand`, `model`, `fuel_type`, `transmission`, `accident`, `clean_title`
    """.format(train_df.shape[0], train_df.shape[1]))

    st.subheader("Marka Dağılımı")
    st.markdown("En çok ilanı verilen markaları aşağıdaki çubuk grafikte görebilirsiniz.")
    brand_counts = train_df["brand"].value_counts().reset_index()
    brand_counts.columns = ["brand", "count"]
    fig_brand_sorted = px.bar(
        brand_counts,
        x="brand",
        y="count",
        title="Marka Dağılımı",
        text="count"
    )
    fig_brand_sorted.update_traces(textposition="outside")
    fig_brand_sorted.update_layout(xaxis={'categoryorder': 'total descending'})
    st.plotly_chart(fig_brand_sorted)

    st.subheader("Yakıt Türü Dağılımı")
    st.markdown("Araçlarda kullanılan yakıt türlerinin oranları aşağıdaki pasta grafikte gösterilmiştir.")
    fig_fuel = px.pie(train_df, names="fuel_type", title="Yakıt Türü Dağılımı")
    st.plotly_chart(fig_fuel)

    st.subheader("Fiyat Dağılımı")
    st.markdown("İkinci el araç fiyatlarının dağılımı aşağıdaki histogramda verilmiştir.")
    fig_price = px.histogram(train_df, x="price", nbins=50, title="Araç Fiyatlarının Dağılımı")
    st.plotly_chart(fig_price)

with clean_tab:
    st.header("🧼 Veri Temizleme Açıklamaları")
    st.markdown("""
    Modelleme öncesi, eksik veriler doldurulmuş ve metin formatındaki veriler sayısallaştırılmıştır.
    Bu işlemler modelin daha doğru sonuçlar üretmesi için önemlidir.
    """)

    st.markdown("""
    Veriler modellemeye uygun hale getirildi:

    - **Eksik Verilerin Doldurulması:**
        - `fuel_type`: Bilinmeyenler → "Unknown"
        - `accident`: Eksik kayıtlar → "None reported"
        - `clean_title`: Eksik başlık → "Unknown"

    - **Kategorik Verilerin Dönüştürülmesi:**
        - `Label Encoding` yöntemi uygulandı. Bu yöntem, kategorik değerleri sayısal değerlere dönüştürür.
        - Ağaç tabanlı algoritmalarla uyumludur ve işlem süresi kısadır.

    🔄 Bu işlemler model tahmini dosyasının hazırlanma sürecinde önceden uygulanmıştır.
    """)

with model_tab:
    st.header("🔎 Doğrulama Kümesi Üzerinde Tahminler")
    st.markdown("""
    Aşağıda, farklı modeller tarafından yapılan fiyat tahminleri ile gerçek fiyatlar karşılaştırılmıştır.
    Noktalar ne kadar çizgiye yakınsa model o kadar başarılıdır.
    """)
    model_sel = st.selectbox("Model seçin", [col for col in pred_df.columns if col not in ["id", "actual"]])
    fig = px.scatter(pred_df, x="actual", y=model_sel, title=f"Gerçek vs Tahmin Fiyatları ({model_sel})",
                     labels={"actual": "Gerçek Fiyat", model_sel: "Tahmin Fiyat"}, opacity=0.6)
    fig.add_shape(type="line", x0=pred_df.actual.min(), x1=pred_df.actual.max(), y0=pred_df.actual.min(),
                  y1=pred_df.actual.max(), line=dict(dash="dash"))
    st.plotly_chart(fig)
    st.dataframe(pred_df[["id", "actual", model_sel]].head(20))

with eval_tab:
    st.header("📈 Model Performans Karşılaştırması")
    st.markdown("""
    Aşağıdaki tabloda ve grafikte modellerin başarı metrikleri sunulmaktadır.
    Düşük MAE ve RMSE ile yüksek R² skoru, iyi performansı gösterir.
    """)
    st.dataframe(result_df)
    st.plotly_chart(px.bar(result_df.melt(id_vars="Unnamed: 0", var_name="Metric", value_name="Value"),
                           x="Unnamed: 0", y="Value", color="Metric", barmode="group",
                           labels={"Unnamed: 0": "Model"}, title="Model Performans Metrikleri"))

with submission_tab:
    st.header("📤 Test Veri Seti Üzerinde Tahminler")
    st.markdown("""
    Aşağıda test veri seti için Random Forest modeliyle yapılan tahminleri görebilirsiniz.
    Bu tahminler, submission formatına uygun şekilde indirilip dış sistemlerde kullanılabilir.
    """)
    st.dataframe(test_pred.head(20))

    st.download_button(
        label="📥 Sample Submission Dosyasını İndir",
        data=test_pred.to_csv(index=False).encode('utf-8'),
        file_name="sample_submission.csv",
        mime="text/csv"
    )