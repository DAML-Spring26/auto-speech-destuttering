import joblib

model = joblib.load("models/dysfluency_rf.pkl")

print("Model loaded successfully")
# future access if you want the model:
# model = joblib.load("models/dysfluency_rf.pkl")
# prediction = model.predict(X)