import joblib

def load_models():
    path = "../models/rand_forest/"
    models = []
    for pos in ["goalkeepers", "defenders", "midfielders", "forwards"]:
         # Load the model for each position
        print(f"Loading model for {pos}...")
        model = joblib.load(f"{path}{pos}.joblib")
        models.append(model)
        return models

def run_model(position, stats, models):
    model = models[position]
    
        
