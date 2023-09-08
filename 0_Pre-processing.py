import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
# Loading
census_data = pd.read_csv('census.csv')
# Separating classifiers
X_census = census_data.iloc[:, 0:14].values
y_census = census_data.iloc[:, 14].values

# Label Encoder
label_encoder_work_class = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital_status = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_native_country = LabelEncoder()

X_census[:, 1] = label_encoder_work_class.fit_transform(X_census[:, 1])
X_census[:, 3] = label_encoder_education.fit_transform(X_census[:, 3])
X_census[:, 5] = label_encoder_marital_status.fit_transform(X_census[:, 5])
X_census[:, 6] = label_encoder_occupation.fit_transform(X_census[:, 6])
X_census[:, 7] = label_encoder_relationship.fit_transform(X_census[:, 7])
X_census[:, 8] = label_encoder_race.fit_transform(X_census[:, 8])
X_census[:, 9] = label_encoder_sex.fit_transform(X_census[:, 9])
X_census[:, 13] = label_encoder_native_country.fit_transform(X_census[:, 13])

scaler_census = StandardScaler()
X_census = scaler_census.fit_transform(X_census)

with open('census_data.pkl', mode='wb') as f:
    pickle.dump([X_census, y_census], f)






