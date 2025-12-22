from sklearn.preprocessing import LabelEncoder

def create_features(df):
    df = df.copy()
    df['year_month'] = df['prev_sold_date'].dt.to_period('M').astype(str)

    for col in ['zip_code','season','metromicro']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    features = ['bed','bath','acre_lot','zip_code','season','metromicro']
    target = 'price'
    X = df[features]
    y = df[target]
    return X, y
