"""
Core Data Processing Package
============================

Course: Big Data Analytics (BDA) - IIIT Allahabad  
Assignment: HDA-3 Multimodal Sleep EEG and Wearable Data Analysis

This package contains core business logic for data processing, model management,
and prediction orchestration.

Modules:
--------
- data_manager: CSV data loading, validation, and subject filtering
- model_manager: Advanced ML model loading (TabPFN + Cross-Modal Attention)  
- preprocessing: Feature scaling and demographic preprocessing
- report_orchestrator: Coordinates report generation workflow

Usage:
------
from core.data_manager import DataManager
from core.model_manager import ModelManager
from core.preprocessing import PreprocessingManager
from core.report_orchestrator import ReportOrchestrator
"""

__version__ = "1.0.0"
__author__ = "Rishabh"
__course__ = "Big Data Analytics - IIIT Allahabad"

# Package metadata
__all__ = [
    'data_manager',
    'model_manager',
    'preprocessing',
    'report_orchestrator'
]
