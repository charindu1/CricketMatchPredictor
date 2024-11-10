from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the dataset
data = pd.read_csv(r'dataset\cleaned_cricket.csv')

# Load the trained model
model = joblib.load('predictor.pickle')

# Mapping dictionaries (add all mappings you've used)
team_mapping = {
    'India': 1, 
    'Australia': 2, 
    'England': 3, 
    'South Africa': 4, 
    'Pakistan': 5,
    'Sri Lanka': 6,
    'New Zealand': 7, 
    'Zimbabwe': 8, 
    'West Indies': 9,
    'Bangladesh': 10, 
    'Kenya': 11, 
    'Netherlands': 12, 
    'Namibia': 13, 
    'Canada': 14,
    'Hong Kong': 15, 
    'U.A.E.': 16, 
    'U.S.A.': 17, 
    'Ireland': 18, 
    'Scotland': 19,
    'Afghanistan': 20, 
    'P.N.G.': 21, 
    'Nepal': 22, 
    'Oman': 23
}
type_mapping = {
    'Test': 1,
    'ODI': 2,
    'T20': 3
}
mapping_ground = {ground: i for i, ground in enumerate(data['Ground'].unique())}
toss_mapping = {'won': 1, 'lost': 2}
bat_mapping = {'1st': 1, '2nd': 2}

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    your_team = request.form['Your_team']
    other_team = request.form['Other_team']
    ground = request.form['ground']
    match_type = request.form['Type']
    toss = request.form['Toss']  
    bat = request.form['Bat']    

    # Convert inputs to model-appropriate formats
    user_data = pd.DataFrame({
        'Selected_team': [team_mapping.get(your_team)],
        'Other_team': [team_mapping.get(other_team)],
        'Ground': [mapping_ground.get(ground)],
        'Type': [type_mapping.get(match_type)],
        'Toss': [toss_mapping.get(toss, 0)],
        'Bat': [bat_mapping.get(bat, 0)]
    })

    # Predict
    probabilities = model.predict_proba(user_data)[0]
    results = {
        'win_probability': probabilities[0] * 100,
        'loss_probability': probabilities[1] * 100,
        'draw_probability': probabilities[2] * 100 if match_type == 'Test' else 0,
        'tie_probability': probabilities[3] * 100 if len(probabilities) > 3 else 0
    }

    return render_template('predict.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
