import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier

class EcoRouter:
    def __init__(self):
        # Lightweight NLP model for semantic embeddings (runs efficiently on CPU)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.is_trained = False

    def _extract_features(self, texts):
        """Extracts semantic embeddings and syntactic features (like length)."""
        # Generate 384-dimensional embeddings
        embeddings = self.encoder.encode(texts, show_progress_bar=False)
        # Add word count as an additional complexity feature
        lengths = np.array([len(str(t).split()) for t in texts]).reshape(-1, 1)
        return np.hstack((embeddings, lengths))

    def train(self, df):
        print("Extracting semantic features (this takes a moment)...")
        X_final = self._extract_features(df['prompt'].tolist())
        
        print("Training Random Forest Classifier...")
        self.classifier.fit(X_final, df['label'])
        self.is_trained = True

    def predict(self, text):
        if not self.is_trained:
            raise Exception("Router is not trained or loaded yet.")
        
        X_new = self._extract_features([text])
        return self.classifier.predict(X_new)[0]

    def save(self, filepath="router_model.pkl"):
        """Saves the trained model weights."""
        joblib.dump(self.classifier, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath="router_model.pkl"):
        """Loads pre-trained model weights for instant inference."""
        self.classifier = joblib.load(filepath)
        self.is_trained = True