import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv("car data.csv")

column_names = ["Selling_type", "Transmission", "Fuel_Type", "Car_Name" ]
encoders = {}
for col in column_names:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"Classes for {col}: {le.classes_}")

y = df["Selling_Price"]
X = df.drop(["Selling_Price"],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


reg = LinearRegression().fit(X_train, y_train)
pickle.dump(reg, open('model.pkl','wb'))



print(reg.score(X_train, y_train))

model = pickle.load(open('model.pkl','rb'))

# Example car details for prediction
car_details = ["ritz", 2014, 5.59, 27000, "Petrol", "Dealer", "Manual", 0]

manual_encoded_values = {
    "Car_Name": {"ritz": encoders["Car_Name"].transform(["ritz"])[0]},
    "Fuel_Type": {"Petrol": encoders["Fuel_Type"].transform(["Petrol"])[0]},
    "Selling_type": {"Dealer": encoders["Selling_type"].transform(["Dealer"])[0]},
    "Transmission": {"Manual": encoders["Transmission"].transform(["Manual"])[0]}
}

car_details_encoded = [
    manual_encoded_values["Car_Name"]["ritz"],  # Car_Name
    car_details[1],  # Year
    car_details[2],  # Present_Price
    car_details[3],  # Driven_kms
    manual_encoded_values["Fuel_Type"]["Petrol"],  # Fuel_Type
    manual_encoded_values["Selling_type"]["Dealer"],  # Selling_type
    manual_encoded_values["Transmission"]["Manual"],  # Transmission
    car_details[7]  # Owner
]


prediction = model.predict([car_details_encoded])
print("Predicted Selling Price:", prediction[0])

