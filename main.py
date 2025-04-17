from src.predict import predict_obesity

def ask_questions():
    print("Please answer the following questions to predict obesity level:")

    questions = {
        "Gender": "Gender (Male/Female): ",
        "Age": "Age: ",
        "Height": "Height (in meters): ",
        "Weight": "Weight (in kg): ",
        "family_history_with_overweight": "Family history with overweight? (yes/no): ",
        "FAVC": "Do you consume high-caloric food frequently? (yes/no): ",
        "FCVC": "How often do you eat vegetables? (1-3): ",
        "NCP": "Number of main meals (1-4): ",
        "CAEC": "Do you eat between meals? (no/Sometimes/Frequently/Always): ",
        "SMOKE": "Do you smoke? (yes/no): ",
        "CH2O": "Daily water intake (1-3): ",
        "SCC": "Do you monitor your calorie consumption? (yes/no): ",
        "FAF": "Physical activity frequency (0-3): ",
        "TUE": "Time spent on technology devices daily (0-2): ",
        "CALC": "Frequency of alcohol consumption (no/Sometimes/Frequently/Always): ",
        "MTRANS": "Primary mode of transportation (Public_Transportation/Walking/Bike/Automobile/Motorbike): "
    }

    inputs = {}
    for key, question in questions.items():
        value = input(question)
        inputs[key] = value.strip()

    return inputs

if __name__ == "__main__":
    user_input = ask_questions()
    result = predict_obesity(user_input)
    print(f"\n✅ Predicted Obesity Level: {result}")
