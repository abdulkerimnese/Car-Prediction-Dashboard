import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Eğitim ve test verilerini yükle
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Eksik verilerin doldurulması
for df in [train_df, test_df]:
    df["fuel_type"].fillna("Unknown", inplace=True)
    df["accident"].fillna("None reported", inplace=True)
    df["clean_title"].fillna("Unknown", inplace=True)

# Ortak kategorik kolonlar
cat_cols = ["brand", "model", "fuel_type", "engine", "transmission", "ext_col", "int_col", "accident", "clean_title"]

# Tüm verilerdeki kategorileri eşleyebilmek için birleştir
combined = pd.concat([train_df[cat_cols], test_df[cat_cols]], axis=0)

# Label encoding (tüm veri üzerinden fit -> hem train hem test aynı etiketlerle encode edilir)
encoder = LabelEncoder()
for col in cat_cols:
    encoder.fit(combined[col].astype(str))
    train_df[col] = encoder.transform(train_df[col].astype(str))
    test_df[col] = encoder.transform(test_df[col].astype(str))

# Model eğitimi (Random Forest)
X_train = train_df.drop(["id", "price"], axis=1)
y_train = train_df["price"]
X_test = test_df.drop("id", axis=1)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# Submission formatına uygun çıktı
submission_df = pd.DataFrame({
    "id": test_df["id"],
    "price": preds.astype(int)
})

submission_df.to_csv("test_predictions.csv", index=False)
print("test_predictions.csv başarıyla oluşturuldu.")