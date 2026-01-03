scale=StandardScaler()
X_train_scaled=scale.fit_transform(X_train)
X_test_scaled=scale.transform(X_test)
with open('scale.pkl', 'wb') as f:
    pickle.dump(scale,f)


def load_model_component():
    model=tf.keras.models.load_model('best_model.h5')
# Loading scalar component
    with open('scale.pkl','rb') as f:
        scalar=pickle.load(f)
# Loading selected features:
    with open('selected_features','rb')as f:
        scaled_features=pickle.load(f)
    return model,scalar,scaled_features


def predict_churn():
    model,scalar,selected_features=load_model_component()
    sample_customer=pd.DataFrame([[3,33.45,3300,0,0]],columns=selected_features)
    sample_scaled=scalar.transform(sample_customer)
    prediction=model.predict(sample_scaled)[0][0]
    print(prediction)
    print("sample prediction for",selected_features)
    print('Prediction for input feature',sample_customer.values[0])
    print(f"churn probability : {prediction:.2f}")
    print(f"will churn :{'Yes' if prediction>0.5 else 'No'}")
