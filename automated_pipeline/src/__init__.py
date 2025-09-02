"""
WESAD Automated Pipeline - Source Package
==========================================

Course: Big Data Analytics (BDA) - IIIT Allahabad  
Assignment: HDA-3 Multimodal Sleep EEG and Wearable Data Analysis

This package contains the core pipeline components for processing
raw WESAD .pkl files through advanced ML models (TabPFN + Cross-Modal Attention)
and generating professional PDF reports.

Modules:
--------
- data_processor: Raw .pkl file processing and feature extraction
- model_ensemble: Model loading and inference pipeline  
- report_generator: PDF report creation with attention visualizations
- streamlit_app: Web interface for the pipeline

Dependencies:
-------------
- torch>=2.0.0 (Cross-Modal Attention)
- scikit-learn>=1.3.0 (TabPFN and traditional ML)
- pandas>=2.0.0 (Data manipulation)
- streamlit>=1.28.0 (Web interface)

Usage:
------
Import modules as needed:
    from src.data_processor import WESADProcessor
    from src.model_ensemble import ModelEnsemble
    from src.report_generator import ReportGenerator
"""

__version__ = "1.0.0"
__author__ = "Rishabh"
__course__ = "Big Data Analytics - IIIT Allahabad"
__assignment__ = "HDA-3"

# Package metadata
__all__ = [
    'data_processor',
    'model_ensemble',
    'report_generator',
    'streamlit_app'
]
