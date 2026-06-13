from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("KNNmodel.pkl")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    try:
        gmat = float(request.form['gmat'])
        gpa = float(request.form['gpa'])
        exp = float(request.form['exp'])

        data = [[gmat, gpa, exp]]

        prediction = model.predict(data)

        if prediction[0] == 1:
            result = "🎉 High Chance of Admission"
            status = "success"
        else:
            result = "❌ Low Chance of Admission"
            status = "danger"

        return render_template(
            'index.html',
            prediction_text=result,
            status=status,
            gmat=gmat,
            gpa=gpa,
            exp=exp
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}",
            status="danger"
        )


if __name__ == '__main__':
    app.run(debug=True)