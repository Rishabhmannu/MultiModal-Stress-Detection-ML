"""
Enhanced 4-page clinical PDF generator for WESAD
===============================================

FIXED VERSION - No circular imports, actual report generation code.

This module generates professional 4-page clinical PDF reports that preserve
the exact look/flow of your original notebook S5-style reports.

Features:
- Model-agnostic: accepts unified "prediction adapter" payload
- 4-page A4 format with consistent fonts/colors
- High-DPI matplotlib figures with deterministic styling
- Simple validation to catch "blank/1-page" regressions

Location: core/enhanced_report_generator.py (REPLACE EXISTING FILE)
"""

import datetime as dt
import json
import os
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# ReportLab imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportStyle:
    """Style configuration for clinical reports."""

    def __init__(self):
        # Page settings
        self.page_size = A4
        self.margin_mm = 16.0

        # Typography
        self.font_name = "Helvetica"
        self.title_size = 18
        self.heading_size = 14
        self.body_size = 10
        self.small_size = 8

        # Colors (professional medical palette)
        self.palette = {
            "primary": "#2E86AB",      # Professional blue
            "secondary": "#A23B72",    # Clinical accent
            "success": "#4ECDC4",      # Info/success
            "warning": "#F18F01",      # Warning
            "danger": "#C73E1D",       # Alert/danger
            "text": "#212529",         # Dark text
            "background": "#FFFFFF",   # White background
            "grid": "#E9ECEF"          # Light grid
        }

        # Figure settings
        self.figure_dpi = 300
        self.figure_width = 7.0
        self.figure_height = 6.0


class EnhancedReportGenerator:
    """
    Enhanced clinical report generator for WESAD predictions.
    
    Generates professional 4-page PDF reports with clinical analysis,
    visualizations, and interpretations based on model predictions.
    """

    def __init__(self, output_dir: str = "outputs/reports",
                 figures_dir: str = "outputs/reports/figures",
                 style: Optional[ReportStyle] = None):
        """
        Initialize the enhanced report generator.
        
        Args:
            output_dir: Directory to save generated reports
            figures_dir: Directory to save generated figures
            style: Optional style configuration
        """
        self.output_dir = output_dir
        self.figures_dir = figures_dir
        self.style = style or ReportStyle()

        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

        # Condition configuration
        self.condition_names = ['Baseline',
                                'Stress', 'Amusement', 'Meditation']
        self.condition_colors = {
            'Baseline': self.style.palette['primary'],
            'Stress': self.style.palette['danger'],
            'Amusement': self.style.palette['warning'],
            'Meditation': self.style.palette['success']
        }

        logger.info(f"EnhancedReportGenerator initialized: {self.output_dir}")

    def generate_report(self, subject_id: str, prediction_adapter: Dict[str, Any],
                        enable_validation: bool = True) -> str:
        """
        Generate a complete 4-page clinical report.
        
        Args:
            subject_id: Subject identifier
            prediction_adapter: Adapted prediction payload from ModelPredictionsAdapter
            enable_validation: Enable PDF size/content validation
            
        Returns:
            Path to generated PDF report
        """
        logger.info(
            f"Generating enhanced clinical report for subject {subject_id}")

        try:
            # Extract data from adapter payload
            clinical_analysis = prediction_adapter.get('clinical_analysis', {})
            predictions = prediction_adapter.get('predictions', [])
            probabilities = prediction_adapter.get('probabilities', [])
            condition_names = prediction_adapter.get(
                'condition_names', self.condition_names)

            # Generate visualizations
            fig_summary = self._create_summary_visualization(
                subject_id, predictions, probabilities, condition_names, clinical_analysis
            )

            fig_clinical = self._create_clinical_visualization(
                subject_id, prediction_adapter
            )

            # Generate PDF report
            report_path = self._create_pdf_report(
                subject_id, prediction_adapter, fig_summary, fig_clinical
            )

            # Validate if enabled
            if enable_validation:
                self._validate_report(report_path)

            logger.info(f"Enhanced clinical report generated: {report_path}")
            return report_path

        except Exception as e:
            logger.error(f"Error generating enhanced report: {e}")
            raise

    def _create_summary_visualization(self, subject_id: str, predictions: List[int],
                                      probabilities: List[List[float]], condition_names: List[str],
                                      clinical_analysis: Dict[str, Any]) -> str:
        """Create summary visualization for page 2 of report."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Physiological Analysis Summary - Subject {subject_id}',
                     fontsize=16, fontweight='bold', y=0.95)

        # 1. Prediction distribution
        if predictions:
            pred_counts = np.bincount(
                predictions, minlength=len(condition_names))
            bars = ax1.bar(condition_names, pred_counts,
                           color=[self.condition_colors.get(name, self.style.palette['primary'])
                                  for name in condition_names],
                           alpha=0.8, edgecolor='black')

            # Add value labels
            for bar, count in zip(bars, pred_counts):
                if count > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                             str(count), ha='center', va='bottom', fontweight='bold')

            ax1.set_ylabel('Number of Windows')
            ax1.set_title('Condition Distribution')
            ax1.tick_params(axis='x', rotation=45)
        else:
            ax1.text(0.5, 0.5, 'No prediction data available',
                     ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Condition Distribution')

        # 2. Mean probabilities
        if probabilities:
            mean_probs = np.mean(probabilities, axis=0)
            bars = ax2.bar(condition_names, mean_probs,
                           color=[self.condition_colors.get(name, self.style.palette['primary'])
                                  for name in condition_names],
                           alpha=0.8, edgecolor='black')

            # Add value labels
            for bar, prob in zip(bars, mean_probs):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')

            ax2.set_ylabel('Average Probability')
            ax2.set_title('Mean Condition Probabilities')
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No probability data available',
                     ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Mean Condition Probabilities')

        # 3. Clinical summary
        if clinical_analysis:
            summary_text = []
            dominant_condition = clinical_analysis.get(
                'dominant_condition', 'Unknown')
            dominant_percentage = clinical_analysis.get(
                'dominant_percentage', 0)
            summary_text.append(f"Dominant Pattern: {dominant_condition}")
            summary_text.append(
                f"Pattern Frequency: {dominant_percentage:.1f}%")

            if 'stress_classification' in clinical_analysis:
                stress_class = clinical_analysis['stress_classification'].get(
                    'classification', 'Unknown')
                summary_text.append(f"Stress Level: {stress_class}")

            if 'risk_assessment' in clinical_analysis:
                risk_level = clinical_analysis['risk_assessment'].get(
                    'level', 'Unknown')
                summary_text.append(f"Risk Assessment: {risk_level}")

            n_windows = clinical_analysis.get('n_windows', 0)
            summary_text.append(f"Analysis Windows: {n_windows}")

            ax3.text(0.1, 0.8, '\n'.join(summary_text), transform=ax3.transAxes,
                     fontsize=11, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        else:
            ax3.text(0.5, 0.5, 'No clinical analysis available',
                     ha='center', va='center', transform=ax3.transAxes)

        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('Clinical Summary')

        # 4. Model performance
        performance_text = ["TabPFN Model: 100.0% accuracy",
                            "Cross-Modal Attention: 84.1% accuracy",
                            "Ensemble: Weighted combination",
                            f"Processing: {len(predictions)} windows"]

        ax4.text(0.1, 0.8, '\n'.join(performance_text), transform=ax4.transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Model Performance')

        plt.tight_layout()

        # Save visualization
        fig_path = os.path.join(self.figures_dir, f"{subject_id}_summary.png")
        plt.savefig(fig_path, dpi=self.style.figure_dpi, bbox_inches='tight',
                    facecolor='white')
        plt.close()

        return fig_path

    def _create_clinical_visualization(self, subject_id: str,
                                       prediction_adapter: Dict[str, Any]) -> str:
        """Create clinical analysis visualization for page 3 of report."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Clinical Analysis - Subject {subject_id}',
                     fontsize=16, fontweight='bold', y=0.95)

        clinical_analysis = prediction_adapter.get('clinical_analysis', {})
        probabilities = prediction_adapter.get('probabilities', [])
        condition_names = prediction_adapter.get(
            'condition_names', self.condition_names)

        # 1. Probability heatmap over time
        if probabilities:
            # Show up to 20 windows
            prob_matrix = np.array(probabilities[:min(20, len(probabilities))])
            im = ax1.imshow(prob_matrix.T, cmap='Blues',
                            aspect='auto', vmin=0, vmax=1)
            ax1.set_xlabel('Time Windows')
            ax1.set_ylabel('Conditions')
            ax1.set_title('Probability Patterns Over Time')
            ax1.set_yticks(range(len(condition_names)))
            ax1.set_yticklabels(condition_names)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
            cbar.set_label('Probability')
        else:
            ax1.text(0.5, 0.5, 'No probability data available',
                     ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Probability Patterns Over Time')

        # 2. Population ranking (if available)
        if 'population_ranking' in clinical_analysis:
            ranking = clinical_analysis['population_ranking']
            metrics = []
            percentiles = []

            if 'stress_reactivity_percentile' in ranking:
                metrics.append('Stress Reactivity')
                percentiles.append(ranking['stress_reactivity_percentile'])

            if 'heart_rate_percentile' in ranking:
                metrics.append('Heart Rate')
                percentiles.append(ranking['heart_rate_percentile'])

            if metrics:
                bars = ax2.barh(metrics, percentiles,
                                color=self.style.palette['primary'], alpha=0.7)

                # Add value labels
                for bar, pct in zip(bars, percentiles):
                    ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                             f'{pct}th', ha='left', va='center', fontweight='bold')

                ax2.set_xlabel('Population Percentile')
                ax2.set_title('Population Ranking')
                ax2.set_xlim(0, 100)
            else:
                ax2.text(0.5, 0.5, 'No population ranking available',
                         ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Population Ranking')
        else:
            ax2.text(0.5, 0.5, 'No population ranking available',
                     ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Population Ranking')

        # 3. Risk assessment visualization
        if 'risk_assessment' in clinical_analysis:
            risk = clinical_analysis['risk_assessment']
            risk_level = risk.get('level', 'Unknown')
            risk_description = risk.get(
                'description', 'No description available')

            # Risk level color coding
            risk_colors = {
                'Low': self.style.palette['success'],
                'Normal': self.style.palette['primary'],
                'Elevated': self.style.palette['warning'],
                'Uncertain': self.style.palette['secondary']
            }

            risk_color = risk_colors.get(
                risk_level, self.style.palette['primary'])

            # Create risk level indicator
            circle = plt.Circle((0.5, 0.7), 0.2, color=risk_color, alpha=0.8)
            ax3.add_patch(circle)
            ax3.text(0.5, 0.7, risk_level, ha='center', va='center',
                     fontweight='bold', fontsize=12, color='white')

            # Add description
            ax3.text(0.5, 0.3, risk_description, ha='center', va='center',
                     transform=ax3.transAxes, fontsize=10, wrap=True,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.axis('off')
            ax3.set_title('Risk Assessment')
        else:
            ax3.text(0.5, 0.5, 'No risk assessment available',
                     ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Risk Assessment')

        # 4. Feature importance (if available from attention)
        if 'interpretability' in prediction_adapter:
            interpretability = prediction_adapter['interpretability']
            attention_summary = interpretability.get('attention_summary', [])

            if attention_summary:
                # Show top 5 most important features
                top_features = attention_summary[:5]
                feature_names = [item[0] if isinstance(
                    item, tuple) else str(item) for item in top_features]
                importance_scores = [item[1] if isinstance(item, tuple) and len(
                    item) > 1 else 1.0 for item in top_features]

                bars = ax4.barh(feature_names, importance_scores,
                                color=self.style.palette['secondary'], alpha=0.7)

                ax4.set_xlabel('Importance Score')
                ax4.set_title('Top Feature Importance')
                ax4.tick_params(axis='y', labelsize=8)
            else:
                ax4.text(0.5, 0.5, 'No feature importance available',
                         ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Top Feature Importance')
        else:
            ax4.text(0.5, 0.5, 'No feature importance available',
                     ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Top Feature Importance')

        plt.tight_layout()

        # Save visualization
        fig_path = os.path.join(self.figures_dir, f"{subject_id}_clinical.png")
        plt.savefig(fig_path, dpi=self.style.figure_dpi, bbox_inches='tight',
                    facecolor='white')
        plt.close()

        return fig_path

    def _create_pdf_report(self, subject_id: str, prediction_adapter: Dict[str, Any],
                           fig_summary: str, fig_clinical: str) -> str:
        """Create the 4-page PDF report."""
        # Generate report filename
        date_str = dt.date.today().isoformat()
        report_filename = f"WESAD_Clinical_Report_{subject_id}_{date_str}.pdf"
        report_path = os.path.join(self.output_dir, report_filename)

        # Create PDF document
        doc = SimpleDocTemplate(
            report_path,
            pagesize=self.style.page_size,
            leftMargin=self.style.margin_mm * mm,
            rightMargin=self.style.margin_mm * mm,
            topMargin=self.style.margin_mm * mm,
            bottomMargin=self.style.margin_mm * mm,
            title=f"WESAD Clinical Report - {subject_id}",
            author="WESAD Clinical Pipeline"
        )

        # Get styles
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=self.style.title_size,
            textColor=colors.HexColor(self.style.palette['primary']),
            alignment=TA_CENTER,
            spaceAfter=20,
            fontName=self.style.font_name + '-Bold'
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=self.style.heading_size,
            textColor=colors.HexColor(self.style.palette['text']),
            spaceAfter=12,
            fontName=self.style.font_name + '-Bold'
        )

        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=self.style.body_size,
            textColor=colors.HexColor(self.style.palette['text']),
            spaceAfter=8,
            fontName=self.style.font_name,
            alignment=TA_JUSTIFY
        )

        # Build report content
        story = []

        # PAGE 1: Title and Executive Summary
        story.append(Paragraph("WESAD Clinical Report", title_style))
        story.append(Spacer(1, 20))

        # Subject information table
        clinical_analysis = prediction_adapter.get('clinical_analysis', {})
        processing_info = prediction_adapter.get('processing_info', {})

        subject_data = [
            ['Subject ID', subject_id],
            ['Session Date', date_str],
            ['Analysis Type', 'Multimodal Physiological Assessment'],
            ['Primary Model', processing_info.get(
                'tabpfn_accuracy', 'TabPFN (100% accuracy)')],
            ['Secondary Model', 'Cross-Modal Attention (84.1% accuracy)'],
            ['Processing Time', processing_info.get('timestamp', 'N/A')]
        ]

        subject_table = Table(subject_data, colWidths=[2.5*72, 3.5*72])
        subject_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1),
             colors.HexColor(self.style.palette['grid'])),
            ('TEXTCOLOR', (0, 0), (-1, -1),
             colors.HexColor(self.style.palette['text'])),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), self.style.font_name),
            ('FONTSIZE', (0, 0), (-1, -1), self.style.body_size),
            ('GRID', (0, 0), (-1, -1), 1,
             colors.HexColor(self.style.palette['grid'])),
        ]))

        story.append(subject_table)
        story.append(Spacer(1, 20))

        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))

        if clinical_analysis:
            executive_summary = self._generate_executive_summary(
                clinical_analysis)
            story.append(Paragraph(executive_summary, body_style))
        else:
            story.append(Paragraph(
                "Analysis completed using advanced machine learning models.", body_style))

        story.append(PageBreak())

        # PAGE 2: Summary Visualization
        story.append(
            Paragraph("Physiological Analysis Summary", heading_style))
        story.append(Spacer(1, 12))

        if os.path.exists(fig_summary):
            story.append(Image(fig_summary, width=6*72, height=5*72))
        else:
            story.append(
                Paragraph("Summary visualization not available.", body_style))

        story.append(PageBreak())

        # PAGE 3: Clinical Analysis
        story.append(
            Paragraph("Clinical Analysis & Interpretation", heading_style))
        story.append(Spacer(1, 12))

        if os.path.exists(fig_clinical):
            story.append(Image(fig_clinical, width=6*72, height=5*72))
        else:
            story.append(
                Paragraph("Clinical visualization not available.", body_style))

        story.append(PageBreak())

        # PAGE 4: Clinical Recommendations
        story.append(
            Paragraph("Clinical Interpretation & Recommendations", heading_style))
        story.append(Spacer(1, 12))


        # Clinical interpretation text
        # Clinical interpretation text
        if clinical_analysis:
            story.extend(self._generate_clinical_interpretation_text(
                clinical_analysis, body_style))

        # Technical notes
        story.append(Spacer(1, 20))
        story.append(Paragraph("Technical Information", heading_style))

        technical_info = f"""
        Analysis performed using TabPFN ({processing_info.get('tabpfn_accuracy', '100%')} accuracy) 
        and Cross-Modal Attention ({processing_info.get('attention_accuracy', '84.1%')} accuracy) models. 
        Feature extraction from multimodal physiological sensors with 60-second time windows. 
        Report generated by WESAD Clinical Pipeline for research and educational purposes.
        """

        story.append(Paragraph(technical_info, body_style))

        # Disclaimer
        story.append(Spacer(1, 20))
        disclaimer = """
        <b>Disclaimer:</b> This report is generated for research and educational purposes only. 
        Results should not be used for medical diagnosis without consultation with qualified 
        healthcare professionals. Generated as part of Big Data Analytics coursework at IIIT Allahabad.
        """
        story.append(Paragraph(disclaimer, body_style))

        # Build PDF
        doc.build(story)

        return report_path

    def _generate_executive_summary(self, clinical_analysis: Dict[str, Any]) -> str:
        """Generate executive summary text from clinical analysis."""
        dominant_condition = clinical_analysis.get(
            'dominant_condition', 'Unknown')
        dominant_percentage = clinical_analysis.get('dominant_percentage', 0)
        n_windows = clinical_analysis.get('n_windows', 0)

        summary = f"""
        Subject demonstrates predominantly {dominant_condition.lower()} response patterns 
        ({dominant_percentage:.1f}% of {n_windows} analyzed time windows). 
        """

        if 'stress_classification' in clinical_analysis:
            stress_class = clinical_analysis['stress_classification'].get(
                'classification', 'Unknown')
            summary += f"Stress response classification: {stress_class}. "

        if 'risk_assessment' in clinical_analysis:
            risk_level = clinical_analysis['risk_assessment'].get(
                'level', 'Unknown')
            summary += f"Clinical risk assessment: {risk_level} level. "

        summary += "Analysis performed using state-of-the-art TabPFN and Cross-Modal Attention models."

        return summary

    def _generate_clinical_interpretation_text(self, clinical_analysis: Dict[str, Any],
                                               body_style: ParagraphStyle) -> List:
        """Generate clinical interpretation paragraphs."""
        elements = []

        # Stress classification
        if 'stress_classification' in clinical_analysis:
            stress_info = clinical_analysis['stress_classification']
            elements.append(
                Paragraph("<b>Stress Response Assessment:</b>", body_style))
            elements.append(Paragraph(f"{stress_info.get('classification', 'Unknown')}: "
                                      f"{stress_info.get('description', 'No description available')}",
                                      body_style))
            elements.append(Spacer(1, 12))

        # Risk assessment
        if 'risk_assessment' in clinical_analysis:
            risk_info = clinical_analysis['risk_assessment']
            elements.append(
                Paragraph("<b>Clinical Risk Assessment:</b>", body_style))
            elements.append(Paragraph(f"Risk Level: {risk_info.get('level', 'Unknown')}. "
                                      f"{risk_info.get('description', 'No description available')}",
                                      body_style))
            elements.append(Spacer(1, 12))

        # Population ranking
        if 'population_ranking' in clinical_analysis:
            ranking = clinical_analysis['population_ranking']
            elements.append(
                Paragraph("<b>Population Comparison:</b>", body_style))

            if 'stress_reactivity_percentile' in ranking:
                percentile = ranking['stress_reactivity_percentile']
                elements.append(Paragraph(f"Subject ranks at {percentile}th percentile for stress reactivity "
                                          f"compared to population norms.", body_style))

            elements.append(Spacer(1, 12))

        return elements

    def _validate_report(self, report_path: str):
        """Validate generated report meets quality standards."""
        if not os.path.exists(report_path):
            raise ValueError(f"Report file not found: {report_path}")

        file_size = os.path.getsize(report_path)
        if file_size < 140_000:  # 140KB minimum for 4-page PDF with images
            raise ValueError(f"Generated PDF seems too small ({file_size} bytes). "
                             f"Expected at least 140KB for complete 4-page report.")

        logger.info(
            f"Report validation passed: {report_path} ({file_size} bytes)")


# Helper functions
def generate_single_enhanced_report(subject_id: str, prediction_adapter: Dict[str, Any],
                                    output_dir: str = "outputs/reports") -> str:
    """
    Convenience function for single enhanced report generation.
    
    Args:
        subject_id: Subject identifier
        prediction_adapter: Adapted prediction payload
        output_dir: Output directory for reports
        
    Returns:
        Path to generated report
    """
    generator = EnhancedReportGenerator(output_dir)
    return generator.generate_report(subject_id, prediction_adapter)


def validate_prediction_adapter(prediction_adapter: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate prediction adapter payload for report generation.
    
    Args:
        prediction_adapter: Prediction payload to validate
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check required top-level keys
    required_keys = ['success', 'predictions', 'condition_names']
    for key in required_keys:
        if key not in prediction_adapter:
            issues.append(f"Missing required key: {key}")

    # Check predictions structure
    predictions = prediction_adapter.get('predictions', [])
    if not isinstance(predictions, (list, np.ndarray)):
        issues.append("Predictions must be a list or array")
    elif len(predictions) == 0:
        issues.append("Predictions array is empty")

    # Check probabilities if present
    probabilities = prediction_adapter.get('probabilities', [])
    if probabilities:
        if len(probabilities) != len(predictions):
            issues.append(
                "Probabilities length doesn't match predictions length")

    is_valid = len(issues) == 0
    return is_valid, issues
