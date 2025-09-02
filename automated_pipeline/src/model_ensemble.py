"""
WESAD Model Ensemble - Advanced ML Model Loading & Inference
============================================================

Course: Big Data Analytics (BDA) - IIIT Allahabad  
Assignment: HDA-3 Multimodal Sleep EEG and Wearable Data Analysis

This module loads the trained TabPFN and Cross-Modal Attention models
and provides inference capabilities with ensemble predictions.

Key Features:
- Loads TabPFN model (100% accuracy) for primary predictions
- Reconstructs Cross-Modal Attention model (84.1% accuracy) for interpretability
- Handles feature preprocessing with saved scalers
- Provides ensemble predictions with confidence scores
- Returns attention patterns for visualization
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention model for physiological sensor fusion.
    
    Exact reproduction of the architecture from training notebook.
    """

    def __init__(self, chest_dim: int, wrist_dim: int, demo_dim: int,
                 hidden_dim: int = 64, num_heads: int = 4, num_classes: int = 4):
        super(CrossModalAttention, self).__init__()

        # Modality encoders - project to common dimension
        self.chest_encoder = nn.Sequential(
            nn.Linear(chest_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.wrist_encoder = nn.Sequential(
            nn.Linear(wrist_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Cross-modal multi-head attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Self-attention for each modality
        self.chest_self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        self.wrist_self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + demo_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Store dimensions for reference
        self.chest_dim = chest_dim
        self.wrist_dim = wrist_dim
        self.demo_dim = demo_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

    def forward(self, chest_features, wrist_features, demo_features):
        batch_size = chest_features.size(0)

        # Encode modalities to common space
        chest_encoded = self.chest_encoder(
            chest_features)  # [batch, hidden_dim]
        wrist_encoded = self.wrist_encoder(
            wrist_features)   # [batch, hidden_dim]

        # Add sequence dimension for attention (treating each sample as sequence of 1)
        chest_seq = chest_encoded.unsqueeze(1)  # [batch, 1, hidden_dim]
        wrist_seq = wrist_encoded.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Self-attention within modalities
        chest_self, chest_attention = self.chest_self_attention(
            chest_seq, chest_seq, chest_seq)
        wrist_self, wrist_attention = self.wrist_self_attention(
            wrist_seq, wrist_seq, wrist_seq)

        # Cross-modal attention: chest attends to wrist
        chest_cross, cross_attention = self.cross_attention(
            chest_self, wrist_self, wrist_self)

        # Remove sequence dimension and combine
        chest_final = chest_cross.squeeze(1)  # [batch, hidden_dim]
        wrist_final = wrist_self.squeeze(1)   # [batch, hidden_dim]

        # Fuse all information
        fused = torch.cat([chest_final, wrist_final, demo_features], dim=1)
        fused_features = self.fusion(fused)

        # Classification
        logits = self.classifier(fused_features)

        # Return logits and attention weights for visualization
        return logits, {
            'cross_attention': cross_attention,
            'chest_attention': chest_attention,
            'wrist_attention': wrist_attention
        }


class ModelEnsemble:
    """
    Ensemble of TabPFN and Cross-Modal Attention models for WESAD inference.
    
    This class handles model loading, feature preprocessing, and ensemble predictions
    with confidence scores and attention pattern extraction.
    """

    def __init__(self, models_dir: str = "models"):
        """
        Initialize the model ensemble.
        
        Args:
            models_dir: Directory containing saved models and scalers
        """
        self.models_dir = Path(models_dir)
        self.device = torch.device('mps' if torch.backends.mps.is_available() else
                                   'cuda' if torch.cuda.is_available() else 'cpu')

        # Model components
        self.tabpfn_model = None
        self.attention_model = None
        self.chest_scaler = None
        self.wrist_scaler = None
        self.demo_scaler = None
        self.label_encoder = None

        # Configuration
        self.config = {}
        self.model_config = {}
        self.feature_config = {}

        # Ensemble weights (from training)
        self.ensemble_weights = {
            'tabpfn': 0.6,
            'attention': 0.4
        }

        # Feature cleaning lists (from training notebook)
        self.features_to_remove = [
            'chest_resp_mean', 'chest_resp_std', 'chest_resp_min', 'chest_resp_max',  # 100% missing
            'chest_resp_rate', 'chest_resp_rate_std'  # Zero variance
        ]

        # Condition mapping
        self.condition_names = ['Baseline',
                                'Stress', 'Amusement', 'Meditation']

        logger.info(f"ModelEnsemble initialized with device: {self.device}")
        logger.info(f"Models directory: {self.models_dir.absolute()}")

    def load_models(self) -> bool:
        """
        Load all saved models and preprocessing components.
        
        Returns:
            bool: True if all models loaded successfully
        """
        try:
            logger.info("Loading saved models and preprocessors...")

            # Load configuration files
            self._load_configurations()

            # Load preprocessors
            self._load_preprocessors()

            # Load TabPFN model
            self._load_tabpfn_model()

            # Load Cross-Modal Attention model
            self._load_attention_model()

            logger.info("✅ All models loaded successfully!")
            return True

        except Exception as e:
            logger.error(f"❌ Error loading models: {str(e)}")
            return False

    def _load_configurations(self):
        """Load configuration files."""
        config_dir = self.models_dir / "config"

        # Feature configuration
        feature_config_path = config_dir / "feature_config.json"
        if feature_config_path.exists():
            with open(feature_config_path) as f:
                self.feature_config = json.load(f)
            logger.info("📋 Feature configuration loaded")

        # Model configuration
        model_config_path = config_dir / "model_config.json"
        if model_config_path.exists():
            with open(model_config_path) as f:
                self.model_config = json.load(f)
            logger.info("🤖 Model configuration loaded")

            # Update ensemble weights if available
            if 'ensemble_weights' in self.model_config:
                self.ensemble_weights.update(
                    self.model_config['ensemble_weights'])

        # Pipeline configuration
        pipeline_config_path = config_dir / "pipeline_config.json"
        if pipeline_config_path.exists():
            with open(pipeline_config_path) as f:
                self.config = json.load(f)
            logger.info("⚙️ Pipeline configuration loaded")

    def _load_preprocessors(self):
        """Load feature scalers and label encoder."""
        scalers_dir = self.models_dir / "scalers"

        # Load scalers
        self.chest_scaler = joblib.load(scalers_dir / "chest_scaler.pkl")
        self.wrist_scaler = joblib.load(scalers_dir / "wrist_scaler.pkl")
        self.demo_scaler = joblib.load(scalers_dir / "demo_scaler.pkl")
        self.label_encoder = joblib.load(scalers_dir / "label_encoder.pkl")

        logger.info(
            "🔧 Preprocessors loaded: chest, wrist, demo scalers + label encoder")

    def _load_tabpfn_model(self):
        """Load TabPFN model."""
        trained_models_dir = self.models_dir / "trained_models"
        tabpfn_path = trained_models_dir / "tabpfn_model.pkl"

        if tabpfn_path.exists():
            self.tabpfn_model = joblib.load(tabpfn_path)
            logger.info("🏆 TabPFN model loaded (100% accuracy)")
        else:
            # Fallback to primary model
            primary_path = trained_models_dir / "primary_model.pkl"
            if primary_path.exists():
                self.tabpfn_model = joblib.load(primary_path)
                logger.info("📊 Primary model loaded (TabPFN alternative)")
            else:
                logger.warning("⚠️ No TabPFN model found")

    def _load_attention_model(self):
        """Load and reconstruct Cross-Modal Attention model."""
        trained_models_dir = self.models_dir / "trained_models"
        attention_path = trained_models_dir / "attention_model.pth"

        if attention_path.exists():
            try:
                # Load saved model data
                logger.info(f"Loading attention model from: {attention_path}")
                checkpoint = torch.load(
                    attention_path, map_location=self.device)

                # Verify checkpoint structure
                if 'model_config' not in checkpoint:
                    logger.error(
                        "model_config missing from attention model checkpoint")
                    logger.info(
                        f"Available checkpoint keys: {list(checkpoint.keys())}")
                    return

                if 'model_state_dict' not in checkpoint:
                    logger.error(
                        "model_state_dict missing from attention model checkpoint")
                    logger.info(
                        f"Available checkpoint keys: {list(checkpoint.keys())}")
                    return

                model_config = checkpoint['model_config']
                logger.info(f"Model config: {model_config}")

                # Validate required config keys
                required_keys = ['chest_dim', 'wrist_dim',
                                 'demo_dim', 'num_classes']
                missing_keys = [
                    key for key in required_keys if key not in model_config]
                if missing_keys:
                    logger.error(
                        f"Missing required config keys: {missing_keys}")
                    return

                # Reconstruct model architecture
                logger.info(
                    "Reconstructing Cross-Modal Attention architecture...")
                self.attention_model = CrossModalAttention(
                    chest_dim=model_config['chest_dim'],
                    wrist_dim=model_config['wrist_dim'],
                    demo_dim=model_config['demo_dim'],
                    hidden_dim=model_config.get('hidden_dim', 64),
                    num_heads=model_config.get('num_heads', 4),
                    num_classes=model_config.get('num_classes', 4)
                )

                # Load trained weights
                logger.info("Loading trained weights...")
                self.attention_model.load_state_dict(
                    checkpoint['model_state_dict'])
                self.attention_model.to(self.device)
                self.attention_model.eval()

                test_accuracy = checkpoint.get('test_accuracy', 0.841)
                logger.info(
                    f"🧠 Cross-Modal Attention model loaded ({test_accuracy:.3f} accuracy)")
                logger.info(
                    f"   📐 Architecture: {model_config['chest_dim']}-{model_config['wrist_dim']}-{model_config['demo_dim']} → {model_config['num_classes']}")

                # Test model with dummy input to verify it works
                logger.info("Testing model with dummy input...")
                with torch.no_grad():
                    dummy_chest = torch.randn(
                        1, model_config['chest_dim']).to(self.device)
                    dummy_wrist = torch.randn(
                        1, model_config['wrist_dim']).to(self.device)
                    dummy_demo = torch.randn(
                        1, model_config['demo_dim']).to(self.device)

                    logits, attention = self.attention_model(
                        dummy_chest, dummy_wrist, dummy_demo)
                    logger.info(
                        f"✅ Model test successful: logits shape {logits.shape}")

            except Exception as e:
                logger.error(f"Error loading attention model: {str(e)}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                self.attention_model = None
        else:
            logger.warning(
                f"⚠️ Attention model not found at: {attention_path}")
            logger.info("Expected file structure:")
            logger.info("  models/trained_models/attention_model.pth")

            # Check if config file exists as alternative
            config_path = trained_models_dir / "attention_config.json"
            if config_path.exists():
                logger.info(f"Found attention config at: {config_path}")
                logger.info(
                    "Model weights not available but config preserved for pipeline reconstruction")

    def clean_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean features by removing problematic columns (same as training).
        
        Args:
            features_df: Raw features dataframe
            
        Returns:
            Cleaned features dataframe
        """
        cleaned_df = features_df.copy()

        # Remove features identified during training
        for feature in self.features_to_remove:
            if feature in cleaned_df.columns:
                cleaned_df = cleaned_df.drop(columns=[feature])
                logger.debug(f"Removed feature: {feature}")

        logger.info(
            f"Feature cleaning: {len(features_df.columns)} → {len(cleaned_df.columns)} features")
        return cleaned_df

    def preprocess_features(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess features for model inference.
        
        Args:
            features_df: Features dataframe from data processor
            
        Returns:
            Tuple of (chest_features, wrist_features, demo_features, combined_features)
        """
        # Clean features first
        clean_df = self.clean_features(features_df)

        # Separate feature types
        chest_features = [
            col for col in clean_df.columns if col.startswith('chest_')]
        wrist_features = [
            col for col in clean_df.columns if col.startswith('wrist_')]
        demo_features = ['age', 'bmi']

        # Handle gender encoding
        if 'gender' in clean_df.columns:
            # Encode gender if it's string
            if clean_df['gender'].dtype == 'object':
                gender_encoded = (clean_df['gender'] == 'female').astype(int)
            else:
                gender_encoded = clean_df['gender']
            clean_df['gender_encoded'] = gender_encoded
            demo_features.append('gender_encoded')
        else:
            # Default gender encoding if missing
            clean_df['gender_encoded'] = 0
            demo_features.append('gender_encoded')

        # Extract feature matrices
        X_chest = clean_df[chest_features].values
        X_wrist = clean_df[wrist_features].values
        X_demo = clean_df[demo_features].values

        # Apply scaling
        X_chest_scaled = self.chest_scaler.transform(X_chest)
        X_wrist_scaled = self.wrist_scaler.transform(X_wrist)
        X_demo_scaled = self.demo_scaler.transform(X_demo)

        # Combined features for TabPFN
        X_combined = np.hstack([X_chest_scaled, X_wrist_scaled, X_demo_scaled])

        logger.info(f"Preprocessing complete:")
        logger.info(f"   🫀 Chest features: {X_chest_scaled.shape}")
        logger.info(f"   ⌚ Wrist features: {X_wrist_scaled.shape}")
        logger.info(f"   👤 Demo features: {X_demo_scaled.shape}")
        logger.info(f"   📊 Combined: {X_combined.shape}")

        return X_chest_scaled, X_wrist_scaled, X_demo_scaled, X_combined

    def predict_tabpfn(self, X_combined: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using TabPFN model.
        
        Args:
            X_combined: Combined feature matrix
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.tabpfn_model is None:
            logger.warning("TabPFN model not available")
            return np.array([]), np.array([])

        try:
            # Get predictions and probabilities
            predictions = self.tabpfn_model.predict(X_combined)
            probabilities = self.tabpfn_model.predict_proba(X_combined)

            logger.debug(f"TabPFN predictions: {len(predictions)} samples")
            return predictions, probabilities

        except Exception as e:
            logger.error(f"TabPFN prediction failed: {e}")
            return np.array([]), np.array([])

    def predict_attention(self, X_chest: np.ndarray, X_wrist: np.ndarray,
                          X_demo: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Make predictions using Cross-Modal Attention model.
        
        Args:
            X_chest: Chest feature matrix
            X_wrist: Wrist feature matrix  
            X_demo: Demographics feature matrix
            
        Returns:
            Tuple of (predictions, probabilities, attention_weights)
        """
        if self.attention_model is None:
            logger.warning("Attention model not available")
            return np.array([]), np.array([]), {}

        try:
            # Convert to tensors
            chest_tensor = torch.FloatTensor(X_chest).to(self.device)
            wrist_tensor = torch.FloatTensor(X_wrist).to(self.device)
            demo_tensor = torch.FloatTensor(X_demo).to(self.device)

            # Forward pass
            with torch.no_grad():
                logits, attention_weights = self.attention_model(
                    chest_tensor, wrist_tensor, demo_tensor)

                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)

                # Convert to numpy
                predictions_np = predictions.cpu().numpy()
                probabilities_np = probabilities.cpu().numpy()

                # Process attention weights
                attention_processed = {}
                for key, weights in attention_weights.items():
                    attention_processed[key] = weights.cpu().numpy()

            logger.debug(
                f"Attention predictions: {len(predictions_np)} samples")
            return predictions_np, probabilities_np, attention_processed

        except Exception as e:
            logger.error(f"Attention prediction failed: {e}")
            return np.array([]), np.array([]), {}

    def predict_ensemble(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Make ensemble predictions combining TabPFN and Attention models.
        
        Args:
            features_df: Features dataframe from data processor
            
        Returns:
            Dictionary containing predictions, probabilities, and attention patterns
        """
        logger.info(
            f"Making ensemble predictions for {len(features_df)} samples...")

        # Preprocess features
        X_chest, X_wrist, X_demo, X_combined = self.preprocess_features(
            features_df)

        results = {
            'success': False,
            'n_samples': len(features_df),
            'predictions': {},
            'probabilities': {},
            'attention_weights': {},
            'ensemble': {},
            'condition_names': self.condition_names
        }

        try:
            # TabPFN predictions
            if self.tabpfn_model is not None:
                tabpfn_pred, tabpfn_prob = self.predict_tabpfn(X_combined)
                if len(tabpfn_pred) > 0:
                    results['predictions']['tabpfn'] = tabpfn_pred
                    results['probabilities']['tabpfn'] = tabpfn_prob
                    logger.info("✅ TabPFN predictions completed")

            # Attention predictions
            if self.attention_model is not None:
                att_pred, att_prob, att_weights = self.predict_attention(
                    X_chest, X_wrist, X_demo)
                if len(att_pred) > 0:
                    results['predictions']['attention'] = att_pred
                    results['probabilities']['attention'] = att_prob
                    results['attention_weights'] = att_weights
                    logger.info("✅ Attention predictions completed")

            # Ensemble predictions
            if ('tabpfn' in results['probabilities'] and
                    'attention' in results['probabilities']):

                # Weighted ensemble of probabilities
                ensemble_prob = (self.ensemble_weights['tabpfn'] * results['probabilities']['tabpfn'] +
                                 self.ensemble_weights['attention'] * results['probabilities']['attention'])
                ensemble_pred = np.argmax(ensemble_prob, axis=1)

                # Confidence scores
                confidence = np.max(ensemble_prob, axis=1)

                results['ensemble'] = {
                    'predictions': ensemble_pred,
                    'probabilities': ensemble_prob,
                    'confidence': confidence,
                    'weights': self.ensemble_weights
                }

                logger.info("✅ Ensemble predictions completed")
                logger.info(f"   📊 Mean confidence: {np.mean(confidence):.3f}")

            results['success'] = True

        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            results['error'] = str(e)

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'models_loaded': {
                'tabpfn': self.tabpfn_model is not None,
                'attention': self.attention_model is not None,
                'preprocessors': all([
                    self.chest_scaler is not None,
                    self.wrist_scaler is not None,
                    self.demo_scaler is not None,
                    self.label_encoder is not None
                ])
            },
            'device': str(self.device),
            'ensemble_weights': self.ensemble_weights,
            'condition_names': self.condition_names,
            'expected_features': {
                'chest': len([col for col in self.chest_scaler.feature_names_in_]),
                'wrist': len([col for col in self.wrist_scaler.feature_names_in_]),
                'demo': 3,
                'total_after_cleaning': len(self.chest_scaler.feature_names_in_) + len(self.wrist_scaler.feature_names_in_) + 3
            } if hasattr(self.chest_scaler, 'feature_names_in_') else {'info': 'Feature names not available'},
            'model_config': self.model_config,
            'device_info': {
                'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
                'cuda_available': torch.cuda.is_available(),
                'current_device': str(self.device)
            }
        }

        return info

    def validate_features(self, features_df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that features match expected format.
        
        Args:
            features_df: Features dataframe
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check basic structure
        if features_df.empty:
            issues.append("Features dataframe is empty")

        # Check for required feature categories
        chest_features = [
            col for col in features_df.columns if col.startswith('chest_')]
        wrist_features = [
            col for col in features_df.columns if col.startswith('wrist_')]

        if len(chest_features) == 0:
            issues.append("No chest features found")
        if len(wrist_features) == 0:
            issues.append("No wrist features found")

        # Check for demographic features
        required_demo = ['age']  # BMI and gender can have defaults
        missing_demo = [
            col for col in required_demo if col not in features_df.columns]
        if missing_demo:
            issues.append(f"Missing demographic features: {missing_demo}")

        # Check for infinite or NaN values
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        if features_df[numeric_cols].isnull().sum().sum() > 0:
            issues.append("Contains NaN values")
        if np.isinf(features_df[numeric_cols]).sum().sum() > 0:
            issues.append("Contains infinite values")

        is_valid = len(issues) == 0
        return is_valid, issues


# Helper functions for pipeline integration
def load_ensemble(models_dir: str = "models") -> ModelEnsemble:
    """
    Convenience function to load model ensemble.
    
    Args:
        models_dir: Directory containing saved models
        
    Returns:
        Loaded ModelEnsemble instance
    """
    ensemble = ModelEnsemble(models_dir)
    success = ensemble.load_models()

    if not success:
        logger.error("Failed to load model ensemble")
        return None

    return ensemble


def predict_from_features(features_df: pd.DataFrame, models_dir: str = "models") -> Dict[str, Any]:
    """
    Convenience function for end-to-end prediction.
    
    Args:
        features_df: Features dataframe
        models_dir: Directory containing saved models
        
    Returns:
        Prediction results
    """
    ensemble = load_ensemble(models_dir)
    if ensemble is None:
        return {'success': False, 'error': 'Failed to load models'}

    return ensemble.predict_ensemble(features_df)
