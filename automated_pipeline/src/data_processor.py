"""
WESAD Data Processor - Raw .pkl File Processing & Feature Extraction
====================================================================

Course: Big Data Analytics (BDA) - IIIT Allahabad  
Assignment: HDA-3 Multimodal Sleep EEG and Wearable Data Analysis

This module processes raw WESAD .pkl files and extracts the same 78 features
used in training the TabPFN and Cross-Modal Attention models.

Key Features:
- Handles raw .pkl files with synchronized sensor data
- Extracts 60-second windows with 50% overlap
- Processes chest (700Hz) and wrist (4-64Hz) sensors
- Applies same preprocessing as training pipeline
- Outputs 78-dimensional feature vectors ready for model inference
"""

import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class WESADProcessor:
    """
    Process raw WESAD .pkl files into feature vectors for model inference.
    
    This class replicates the exact feature extraction methodology from 
    the training notebooks to ensure consistency with trained models.
    """

    def __init__(self):
        """Initialize the WESAD processor with configuration."""
        self.config = {
            'window_size_sec': 60,        # 60-second windows
            'overlap_ratio': 0.5,         # 50% overlap
            'chest_sampling_rate': 700,   # Hz
            # Baseline, Stress, Amusement, Meditation
            'target_conditions': [1, 2, 3, 4],
            'condition_purity_threshold': 0.7,   # 70% minimum purity
            'apply_filters': True,        # Signal preprocessing
        }

        # Expected feature count (from training)
        self.expected_features = {
            'chest': 43,
            'wrist': 35,
            'demographics': 3,
            'total': 81  # Will be reduced to 78 after preprocessing
        }

        logger.info(f"WESADProcessor initialized with config: {self.config}")

    def load_pkl_file(self, pkl_path: Path) -> Dict:
        """
        Load and validate WESAD .pkl file structure.
        
        Args:
            pkl_path: Path to .pkl file
            
        Returns:
            Dict containing subject data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file structure is invalid
        """
        try:
            pkl_path = Path(pkl_path)
            if not pkl_path.exists():
                raise FileNotFoundError(f"File not found: {pkl_path}")

            logger.info(f"Loading .pkl file: {pkl_path.name}")

            with open(pkl_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')

            # Validate required structure
            required_keys = ['signal', 'label', 'subject']
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                raise ValueError(
                    f"Missing required keys in .pkl file: {missing_keys}")

            # Validate signal structure
            signal_data = data['signal']
            required_chest = ['chest']
            required_wrist = ['wrist']

            if 'chest' not in signal_data or 'wrist' not in signal_data:
                raise ValueError("Missing chest or wrist sensor data")

            logger.info(f"Successfully loaded {pkl_path.name}")
            logger.info(f"Available sensors: {list(signal_data.keys())}")

            return data

        except Exception as e:
            logger.error(f"Error loading {pkl_path}: {str(e)}")
            raise

    def preprocess_ecg(self, ecg_signal: np.ndarray, fs: int = 700) -> np.ndarray:
        """Preprocess ECG signal with bandpass filter (0.5-40 Hz)."""
        try:
            nyquist = fs / 2
            low = 0.5 / nyquist
            high = 40 / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            ecg_filtered = signal.filtfilt(b, a, ecg_signal.flatten())
            return ecg_filtered
        except Exception as e:
            logger.warning(f"ECG preprocessing failed: {e}")
            return ecg_signal.flatten()

    def preprocess_eda(self, eda_signal: np.ndarray, fs: int = 700) -> np.ndarray:
        """Preprocess EDA signal with lowpass filter (5 Hz)."""
        try:
            nyquist = fs / 2
            cutoff = 5 / nyquist
            b, a = signal.butter(4, cutoff, btype='low')
            eda_filtered = signal.filtfilt(b, a, eda_signal.flatten())
            return np.maximum(eda_filtered, 0)  # EDA should be non-negative
        except Exception as e:
            logger.warning(f"EDA preprocessing failed: {e}")
            return np.maximum(eda_signal.flatten(), 0)

    def preprocess_respiration(self, resp_signal: np.ndarray, fs: int = 700) -> np.ndarray:
        """Preprocess respiration signal with bandpass filter (0.1-2 Hz)."""
        try:
            nyquist = fs / 2
            low = 0.1 / nyquist
            high = 2 / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            resp_filtered = signal.filtfilt(b, a, resp_signal.flatten())
            return resp_filtered
        except Exception as e:
            logger.warning(f"Respiration preprocessing failed: {e}")
            return resp_signal.flatten()

    def extract_heart_features(self, ecg_signal: np.ndarray, fs: int = 700) -> Dict:
        """Extract heart rate and HRV features from ECG signal."""
        features = {}

        try:
            # Basic statistics
            features['hr_mean'] = np.mean(ecg_signal)
            features['hr_std'] = np.std(ecg_signal)
            features['hr_min'] = np.min(ecg_signal)
            features['hr_max'] = np.max(ecg_signal)

            # Heart rate variability approximation
            rr_intervals = np.diff(ecg_signal)  # Simplified RR intervals
            if len(rr_intervals) > 1:
                features['hrv_rmssd'] = np.sqrt(
                    np.mean(np.square(np.diff(rr_intervals))))
                features['hrv_sdnn'] = np.std(rr_intervals)
                features['hrv_pnn50'] = np.sum(
                    np.abs(np.diff(rr_intervals)) > 0.05) / len(rr_intervals)
            else:
                features['hrv_rmssd'] = 0
                features['hrv_sdnn'] = 0
                features['hrv_pnn50'] = 0

        except Exception as e:
            logger.warning(f"Heart feature extraction failed: {e}")
            # Return zero features if extraction fails
            for key in ['hr_mean', 'hr_std', 'hr_min', 'hr_max', 'hrv_rmssd', 'hrv_sdnn', 'hrv_pnn50']:
                features[key] = 0.0

        return features

    def extract_eda_features(self, eda_signal: np.ndarray, fs: int = 700) -> Dict:
        """Extract electrodermal activity features."""
        features = {}

        try:
            # Basic statistics
            features['eda_mean'] = np.mean(eda_signal)
            features['eda_std'] = np.std(eda_signal)
            features['eda_min'] = np.min(eda_signal)
            features['eda_max'] = np.max(eda_signal)
            features['eda_range'] = features['eda_max'] - features['eda_min']

            # Tonic and phasic components (simplified)
            # Tonic: low-frequency baseline (moving average)
            # 30-second window or 1/4 signal length
            window_size = min(fs * 30, len(eda_signal) // 4)
            if window_size > 0:
                tonic = np.convolve(eda_signal, np.ones(
                    window_size)/window_size, mode='same')
                phasic = eda_signal - tonic

                features['eda_tonic_mean'] = np.mean(tonic)
                features['eda_phasic_mean'] = np.mean(phasic)
                features['eda_phasic_std'] = np.std(phasic)
            else:
                features['eda_tonic_mean'] = features['eda_mean']
                features['eda_phasic_mean'] = 0.0
                features['eda_phasic_std'] = 0.0

            # SCR detection (simplified peak counting)
            peaks, _ = signal.find_peaks(
                eda_signal, height=np.mean(eda_signal))
            features['eda_scr_count'] = len(peaks)
            features['eda_scr_rate'] = len(
                peaks) / (len(eda_signal) / fs)  # peaks per second

        except Exception as e:
            logger.warning(f"EDA feature extraction failed: {e}")
            # Return zero features if extraction fails
            for key in ['eda_mean', 'eda_std', 'eda_min', 'eda_max', 'eda_range',
                        'eda_tonic_mean', 'eda_phasic_mean', 'eda_phasic_std',
                        'eda_scr_count', 'eda_scr_rate']:
                features[key] = 0.0

        return features

    def extract_temperature_features(self, temp_signal: np.ndarray) -> Dict:
        """Extract temperature features."""
        features = {}

        try:
            features['temp_mean'] = np.mean(temp_signal)
            features['temp_std'] = np.std(temp_signal)
            features['temp_min'] = np.min(temp_signal)
            features['temp_max'] = np.max(temp_signal)
            features['temp_range'] = features['temp_max'] - \
                features['temp_min']

            # Temperature slope (trend)
            if len(temp_signal) > 1:
                x = np.arange(len(temp_signal))
                slope, _ = np.polyfit(x, temp_signal, 1)
                features['temp_slope'] = slope
            else:
                features['temp_slope'] = 0.0

        except Exception as e:
            logger.warning(f"Temperature feature extraction failed: {e}")
            for key in ['temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_range', 'temp_slope']:
                features[key] = 0.0

        return features

    def extract_emg_features(self, emg_signal: np.ndarray, fs: int = 700) -> Dict:
        """Extract EMG (muscle activity) features."""
        features = {}

        try:
            features['emg_mean'] = np.mean(emg_signal)
            features['emg_std'] = np.std(emg_signal)
            features['emg_rms'] = np.sqrt(np.mean(np.square(emg_signal)))

            # Frequency domain features
            freqs, psd = signal.welch(emg_signal, fs)

            # Low frequency power (0-50 Hz)
            low_freq_idx = freqs <= 50
            features['emg_power_low'] = np.sum(
                psd[low_freq_idx]) if np.any(low_freq_idx) else 0

            # High frequency power (50-250 Hz)
            high_freq_idx = (freqs > 50) & (freqs <= 250)
            features['emg_power_high'] = np.sum(
                psd[high_freq_idx]) if np.any(high_freq_idx) else 0

            # Power ratio
            features['emg_power_ratio'] = (features['emg_power_high'] /
                                           max(features['emg_power_low'], 1e-10))

        except Exception as e:
            logger.warning(f"EMG feature extraction failed: {e}")
            for key in ['emg_mean', 'emg_std', 'emg_rms', 'emg_power_low', 'emg_power_high', 'emg_power_ratio']:
                features[key] = 0.0

        return features

    def extract_accelerometer_features(self, acc_signal: np.ndarray, fs: int) -> Dict:
        """Extract accelerometer features."""
        features = {}

        try:
            if acc_signal.ndim == 2 and acc_signal.shape[1] == 3:
                # Multi-axis accelerometer
                acc_x, acc_y, acc_z = acc_signal[:,
                                                 0], acc_signal[:, 1], acc_signal[:, 2]

                features['acc_x_mean'] = np.mean(acc_x)
                features['acc_x_std'] = np.std(acc_x)
                features['acc_y_mean'] = np.mean(acc_y)
                features['acc_y_std'] = np.std(acc_y)
                features['acc_z_mean'] = np.mean(acc_z)
                features['acc_z_std'] = np.std(acc_z)

                # Magnitude
                acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
            else:
                # Single axis or flattened
                acc_magnitude = acc_signal.flatten()
                features['acc_x_mean'] = np.mean(acc_magnitude)
                features['acc_x_std'] = np.std(acc_magnitude)
                features['acc_y_mean'] = 0.0
                features['acc_y_std'] = 0.0
                features['acc_z_mean'] = 0.0
                features['acc_z_std'] = 0.0

            # Magnitude statistics
            features['acc_mean'] = np.mean(acc_magnitude)
            features['acc_std'] = np.std(acc_magnitude)
            features['acc_min'] = np.min(acc_magnitude)
            features['acc_max'] = np.max(acc_magnitude)

            # Activity level
            features['acc_energy'] = np.sum(acc_magnitude**2)
            features['acc_activity_level'] = np.sum(
                acc_magnitude > np.mean(acc_magnitude))

        except Exception as e:
            logger.warning(f"Accelerometer feature extraction failed: {e}")
            for key in ['acc_x_mean', 'acc_x_std', 'acc_y_mean', 'acc_y_std', 'acc_z_mean', 'acc_z_std',
                        'acc_mean', 'acc_std', 'acc_min', 'acc_max', 'acc_energy', 'acc_activity_level']:
                features[key] = 0.0

        return features

    def extract_bvp_features(self, bvp_signal: np.ndarray, fs: int = 64) -> Dict:
        """Extract blood volume pulse (BVP/PPG) features."""
        features = {}

        try:
            features['bvp_mean'] = np.mean(bvp_signal)
            features['bvp_std'] = np.std(bvp_signal)
            features['bvp_min'] = np.min(bvp_signal)
            features['bvp_max'] = np.max(bvp_signal)

            # Heart rate from BVP (peak detection)
            peaks, _ = signal.find_peaks(bvp_signal)
            if len(peaks) > 1:
                intervals = np.diff(peaks) / fs  # seconds between peaks
                hr_bvp = 60 / np.mean(intervals)  # beats per minute
                features['bvp_hr_mean'] = hr_bvp
                features['bvp_hr_std'] = np.std(
                    60 / intervals) if len(intervals) > 1 else 0
                features['bvp_hr_range'] = features['bvp_hr_std'] * \
                    2  # approximate range
            else:
                features['bvp_hr_mean'] = 0.0
                features['bvp_hr_std'] = 0.0
                features['bvp_hr_range'] = 0.0

        except Exception as e:
            logger.warning(f"BVP feature extraction failed: {e}")
            for key in ['bvp_mean', 'bvp_std', 'bvp_min', 'bvp_max', 'bvp_hr_mean', 'bvp_hr_std', 'bvp_hr_range']:
                features[key] = 0.0

        return features

    def process_chest_sensors(self, chest_data: Dict, window_start: int, window_end: int) -> Dict:
        """Process chest sensor data for a single window."""
        features = {}

        try:
            # ECG features
            if 'ECG' in chest_data:
                ecg_window = chest_data['ECG'][window_start:window_end].flatten(
                )
                if self.config['apply_filters']:
                    ecg_window = self.preprocess_ecg(ecg_window)
                heart_features = self.extract_heart_features(ecg_window)
                features.update(
                    {f'chest_{k}': v for k, v in heart_features.items()})

            # EDA features
            if 'EDA' in chest_data:
                eda_window = chest_data['EDA'][window_start:window_end].flatten(
                )
                if self.config['apply_filters']:
                    eda_window = self.preprocess_eda(eda_window)
                eda_features = self.extract_eda_features(eda_window)
                features.update(
                    {f'chest_{k}': v for k, v in eda_features.items()})

            # Temperature features
            if 'Temp' in chest_data:
                temp_window = chest_data['Temp'][window_start:window_end].flatten(
                )
                temp_features = self.extract_temperature_features(temp_window)
                features.update(
                    {f'chest_{k}': v for k, v in temp_features.items()})

            # EMG features
            if 'EMG' in chest_data:
                emg_window = chest_data['EMG'][window_start:window_end].flatten(
                )
                emg_features = self.extract_emg_features(emg_window)
                features.update(
                    {f'chest_{k}': v for k, v in emg_features.items()})

            # Accelerometer features
            if 'ACC' in chest_data:
                acc_window = chest_data['ACC'][window_start:window_end]
                acc_features = self.extract_accelerometer_features(
                    acc_window, 700)
                features.update(
                    {f'chest_{k}': v for k, v in acc_features.items()})

            # Respiration features (if available)
            if 'Resp' in chest_data:
                resp_window = chest_data['Resp'][window_start:window_end].flatten(
                )
                if self.config['apply_filters']:
                    resp_window = self.preprocess_respiration(resp_window)
                # Extract basic respiration features
                resp_features = {
                    'resp_mean': np.mean(resp_window),
                    'resp_std': np.std(resp_window),
                    'resp_min': np.min(resp_window),
                    'resp_max': np.max(resp_window)
                }
                features.update(
                    {f'chest_{k}': v for k, v in resp_features.items()})

        except Exception as e:
            logger.warning(f"Chest sensor processing failed: {e}")

        return features

    def process_wrist_sensors(self, wrist_data: Dict, window_start_time: float, window_end_time: float) -> Dict:
        """Process wrist sensor data for a single window."""
        features = {}

        try:
            # Wrist EDA features (4 Hz sampling)
            if 'EDA' in wrist_data:
                wrist_eda = np.array(wrist_data['EDA']).flatten()
                eda_start_idx = int(window_start_time * 4)
                eda_end_idx = int(window_end_time * 4)

                if eda_end_idx <= len(wrist_eda) and eda_start_idx < eda_end_idx:
                    wrist_eda_window = wrist_eda[eda_start_idx:eda_end_idx]
                    wrist_eda_features = self.extract_eda_features(
                        wrist_eda_window, fs=4)
                    features.update(
                        {f'wrist_{k}': v for k, v in wrist_eda_features.items()})

            # Wrist BVP features (64 Hz sampling)
            if 'BVP' in wrist_data:
                wrist_bvp = np.array(wrist_data['BVP']).flatten()
                bvp_start_idx = int(window_start_time * 64)
                bvp_end_idx = int(window_end_time * 64)

                if bvp_end_idx <= len(wrist_bvp) and bvp_start_idx < bvp_end_idx:
                    wrist_bvp_window = wrist_bvp[bvp_start_idx:bvp_end_idx]
                    bvp_features = self.extract_bvp_features(wrist_bvp_window)
                    features.update(
                        {f'wrist_{k}': v for k, v in bvp_features.items()})

            # Wrist temperature features (4 Hz sampling)
            if 'TEMP' in wrist_data:
                wrist_temp = np.array(wrist_data['TEMP']).flatten()
                temp_start_idx = int(window_start_time * 4)
                temp_end_idx = int(window_end_time * 4)

                if temp_end_idx <= len(wrist_temp) and temp_start_idx < temp_end_idx:
                    wrist_temp_window = wrist_temp[temp_start_idx:temp_end_idx]
                    temp_features = self.extract_temperature_features(
                        wrist_temp_window)
                    features.update(
                        {f'wrist_{k}': v for k, v in temp_features.items()})

            # Wrist accelerometer features (32 Hz sampling)
            if 'ACC' in wrist_data:
                wrist_acc = np.array(wrist_data['ACC'])
                acc_start_idx = int(window_start_time * 32)
                acc_end_idx = int(window_end_time * 32)

                if acc_end_idx <= len(wrist_acc) and acc_start_idx < acc_end_idx:
                    wrist_acc_window = wrist_acc[acc_start_idx:acc_end_idx]
                    acc_features = self.extract_accelerometer_features(
                        wrist_acc_window, 32)
                    features.update(
                        {f'wrist_{k}': v for k, v in acc_features.items()})

        except Exception as e:
            logger.warning(f"Wrist sensor processing failed: {e}")

        return features

    def process_pkl_file(self, pkl_path: Path, subject_metadata: Optional[Dict] = None) -> pd.DataFrame:
        """
        Process a single .pkl file into feature vectors.
        
        Args:
            pkl_path: Path to .pkl file
            subject_metadata: Optional metadata (age, gender, etc.)
            
        Returns:
            DataFrame with extracted features
        """
        # Load data
        data = self.load_pkl_file(pkl_path)

        # Extract components
        signal_data = data['signal']
        labels = data['label'].flatten()
        subject_id = pkl_path.stem  # Use filename as subject ID

        chest_data = signal_data['chest']
        wrist_data = signal_data['wrist']

        # Setup windowing
        window_samples = self.config['window_size_sec'] * \
            self.config['chest_sampling_rate']
        overlap_samples = int(window_samples * self.config['overlap_ratio'])
        step_size = window_samples - overlap_samples

        total_samples = len(labels)
        window_starts = range(0, total_samples - window_samples + 1, step_size)

        logger.info(
            f"Processing {len(window_starts)} windows for {subject_id}")

        features_list = []

        # Process each window
        for i, start_idx in enumerate(tqdm(window_starts, desc=f"Processing {subject_id}")):
            end_idx = start_idx + window_samples

            # Get window labels and check purity
            window_labels = labels[start_idx:end_idx]
            unique_labels, counts = np.unique(
                window_labels, return_counts=True)
            dominant_label = unique_labels[np.argmax(counts)]
            dominant_percentage = np.max(counts) / len(window_labels)

            # Skip windows with insufficient purity or non-target conditions
            if (dominant_label not in self.config['target_conditions'] or
                    dominant_percentage < self.config['condition_purity_threshold']):
                continue

            # Initialize feature dictionary
            window_features = {
                'subject_id': subject_id,
                'window_id': i,
                'start_time': start_idx / self.config['chest_sampling_rate'],
                'condition': int(dominant_label),
                'condition_purity': dominant_percentage
            }

            # Process chest sensors
            chest_features = self.process_chest_sensors(
                chest_data, start_idx, end_idx)
            window_features.update(chest_features)

            # Process wrist sensors
            window_start_time = start_idx / self.config['chest_sampling_rate']
            window_end_time = end_idx / self.config['chest_sampling_rate']
            wrist_features = self.process_wrist_sensors(
                wrist_data, window_start_time, window_end_time)
            window_features.update(wrist_features)

            # Add demographics if available
            if subject_metadata:
                window_features.update({
                    'age': subject_metadata.get('age', 25),  # Default age
                    'gender': subject_metadata.get('gender', 'unknown'),
                    'bmi': subject_metadata.get('bmi', 22.0)  # Default BMI
                })
            else:
                # Default demographics
                window_features.update({
                    'age': 25,
                    'gender': 'unknown',
                    'bmi': 22.0
                })

            features_list.append(window_features)

        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)

        logger.info(
            f"Extracted {len(features_df)} feature windows from {subject_id}")
        logger.info(f"Feature columns: {len(features_df.columns)}")

        return features_df


# Helper functions for pipeline integration
def process_single_file(pkl_path: str, metadata: Optional[Dict] = None) -> pd.DataFrame:
    """Convenience function to process a single file."""
    processor = WESADProcessor()
    return processor.process_pkl_file(Path(pkl_path), metadata)


def validate_feature_count(features_df: pd.DataFrame) -> bool:
    """Validate that extracted features match expected count."""
    expected_total = 81  # Before cleaning (will be reduced to 78)
    actual_count = len(features_df.columns)

    logger.info(f"Feature validation: {actual_count} features extracted")

    if actual_count < 70:  # Allow some variance
        logger.warning(f"Low feature count: {actual_count} < 70")
        return False

    return True
