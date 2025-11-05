from sklearn.datasets import fetch_openml

data, target = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)

df = data.copy()
df["income"] = target
df.dropna(inplace=True)
df["jovem"] = df["age"].apply(lambda x: "sim" if x < 30 else "não")

cols_to_drop = ["fnlwgt", "education-num", "native-country", "race"]
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

print(df)
