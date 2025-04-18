from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train import train_model
from src.predict import predict_user_input

def run_training():
    df = load_data("dataSet\ObesityDataSet.csv")
    X, y = preprocess_data(df)
    train_model(X, y)

def ask_questions():
    questions = {
        "Age": "Enter your age: ",
        "Gender": "Gender (Male/Female): ",
        "Height": "Your height in meters (e.g., 1.75): ",
        "Weight": "Your weight in kg (e.g., 70): ",
        "CALC": "How often do you drink alcohol? (no/Sometimes/Frequently): ",
        "FAVC": "Do you eat high-calorie food frequently? (yes/no): ",
        "FCVC": "How often do you eat vegetables (1-3): ",
        "NCP": "How many meals do you have per day? (1-4): ",
        "SCC": "Do you monitor your calories? (yes/no): ",
        "SMOKE": "Do you smoke? (yes/no): ",
        "CH2O": "How much water do you drink daily (liters)? ",
        "family_history_with_overweight": "Do you have family history with overweight? (yes/no): ",
        "FAF": "Physical activity frequency per week (0-3): ",
        "TUE": "Time using tech devices daily (hours)? ",
        "CAEC": "When do you eat between meals? (no/Sometimes/Frequently/Always): ",
        "MTRANS": "Transportation method (Public_Transportation/Walking/Automobile): "
    }

    user_input = {}
    for key, q in questions.items():
        ans = input(q)
        if key not in ["Gender", "CALC", "FAVC", "SCC", "SMOKE", "family_history_with_overweight", "CAEC", "MTRANS"]:
            try:
                ans = float(ans)
            except:
                print("Invalid input. Please enter a number.")
                return
        user_input[key] = ans
    return user_input

if __name__ == "__main__":
    print("1. Train Model\n2. Predict Obesity Level")
    choice = input("Select an option (1/2): ")

    if choice == "1":
        run_training()
    elif choice == "2":
        inputs = ask_questions()
        prediction = predict_user_input(inputs)
        print(f"\n🎯 Predicted Obesity Level: {prediction}")
    else:
        print("Invalid choice.")
