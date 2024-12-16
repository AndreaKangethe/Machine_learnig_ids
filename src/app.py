import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
import torch
from model import MyModel  # Corrected import to match the class name in model.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import subprocess

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Preprocessing function (you should modify this based on your data)
def preprocess_input(data):
    """
    Preprocess input data before passing it to the model.
    - Ensure data is numeric
    - Handle missing values, etc.
    """
    # Example of encoding categorical features and scaling
    label_encoder = LabelEncoder()
    data['protocol_type'] = label_encoder.fit_transform(data['protocol_type'])
    data['service'] = label_encoder.fit_transform(data['service'])
    data['flag'] = label_encoder.fit_transform(data['flag'])
    
    # Handle missing values by filling with column mean
    data = data.fillna(data.mean())
    
    # Return the processed data ready for prediction
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return torch.tensor(data_scaled, dtype=torch.float32)

# Function to get real-time network data (example using tcpdump)
def get_real_time_data():
    """
    Capture live network traffic using tcpdump and process it into the necessary format.
    This is a placeholder example; replace with actual data parsing logic as needed.
    """
    # Run tcpdump command to capture a few packets and parse them
    cmd = "tcpdump -c 10 -nn -t -q -e"
    result = subprocess.run(cmd, shell=True, capture_output=True)
    
    # Process the result into a DataFrame (this is an example; replace with actual feature extraction)
    network_data = pd.DataFrame({
        'protocol_type': ['tcp'],  # Example categorical data
        'service': ['http'],       # Example categorical data
        'flag': ['SF'],            # Example categorical data
        # Add additional network features as needed based on tcpdump output
    })
    
    return network_data

# Define the layout of the dashboard
app.layout = html.Div(
    children=[
        dbc.Row(
            dbc.Col(html.H1("Real-Time Network Intrusion Detection System"))
        ),
        dbc.Row(
            dbc.Col(html.Div("Here we will show network activity and detected alerts."))
        ),
        dbc.Row(
            dbc.Col(
                html.Div(
                    id='prediction-output',  # Placeholder for prediction output
                    children='Waiting for input data...'
                )
            )
        ),
        # Example: A simulated real-time input for network data
        dcc.Interval(
            id='interval-component',
            interval=10000,  # Update every 10 seconds (10000 ms)
            n_intervals=0
        )
    ]
)

# Load the model (ensure MyModel is loaded correctly)
model = MyModel(input_size=78, num_classes=2)  # Make sure input size matches your model's training configuration
model.load_state_dict(torch.load('path_to_model.pth'))
model.eval()

# Update function to make predictions
@app.callback(
    dash.dependencies.Output('prediction-output', 'children'),
    dash.dependencies.Input('interval-component', 'n_intervals')
)
def update_predictions(n):
    # Fetch real-time network data
    new_data = get_real_time_data()
    
    # Preprocess the input data
    processed_data = preprocess_input(new_data)

    # Make prediction with the model
    with torch.no_grad():
        prediction = model(processed_data)

    # Extract the class with the highest probability
    predicted_class = prediction.argmax(dim=1).item()

    # Return the predicted class
    return f"Prediction: {'Normal' if predicted_class == 0 else 'Attack'}"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
