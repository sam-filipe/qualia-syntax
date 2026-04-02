"""
Qualia Syntax Analysis: Main Script
====================================

This script performs Fourier Qualia Space (FQS) analysis on the Yale Classical
Archives Corpus (YCAC) or similar musical corpora. The analysis pipeline:

1. **Data Loading**: Reads corpus files and metadata, converts chord notation
   to pitch class vectors.

2. **Segmentation**: Divides pieces into sections using either:
   - Fixed-size windows with optional overlap
   - Distance-sensitive peak detection

3. **FQS Computation**: Applies DFT to pitch class vectors and projects the
   resulting magnitude vectors onto a 2D RadViz space.

4. **Qualia Classification**: Assigns harmonic quality labels based on FQS
   coordinates.

5. **Statistical Analysis**: Computes transition matrices, clustering, and
   Zipf distribution analysis.

Usage
-----
Run the script and follow the prompts to select:
- Mode (specific composers or entire corpus)
- Composer names (if mode 1)

The script will output statistical reports and visualisations.

Configuration
-------------
Modify the constants in the CONFIGURATION section below to adjust:
- Corpus directory path
- Segmentation strategy and parameters
- Visualisation options

Written for Python 3.11+
"""

import numpy as np

# Import project modules
import corpus
import segmentation
import fourier_qualia_space as fqs
import analysis


# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to the YCAC corpus directory (must contain CSV files and 0_Metadata.csv)
CORPUS_DIRECTORY = '/Users/sp/Documents/InvestigacÌ§aÌƒo/#Code/YCAC corpus/'

# DFT coefficient ordering for RadViz projection
# This determines which DFT coefficients map to which RadViz anchors
# The ordering affects the geometric interpretation of qualia regions
DFT_COEFFICIENT_ORDER = [1, 5, 3, 4, 6, 2]

# Number of anchors in the RadViz space (always 6 for standard FQS)
NUM_ANCHORS = 6

# Radius of the central ambiguity region in the RadViz space
# Points closer to the centre than this are classified as 'A' (ambiguous)
AMBIGUITY_RADIUS = 0.166

# Segmentation strategy: 'windowed' or 'distance-sensitive'
SEGMENTATION_STRATEGY = 'distance-sensitive'

# Windowed analysis parameters (only used if SEGMENTATION_STRATEGY == 'windowed')
WINDOW_SIZE = 30          # Number of chords per window
OVERLAP_SIZE = 0          # Number of chords overlap between windows

# Distance-sensitive analysis parameters
# Method for calculating distances: 'RadViz' or 'DFTMag'
# - 'RadViz': Euclidean distance in 2D RadViz space
# - 'DFTMag': Cosine distance in 6D DFT magnitude space
DISTANCE_METHOD = 'RadViz'

# Visualisation options
SHOW_SEGMENTATION_GRAPHS = False   # Show peak detection plots
SHOW_QUALIA_SEQUENCE = False       # Show qualia progression visualisations

# Display settings
DATAFRAME_COLUMN_WIDTH = 30        # Maximum width for DataFrame columns


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def main():
    """
    Execute the complete Qualia Syntax analysis pipeline.
    """
    
    # -------------------------------------------------------------------------
    # INITIALISATION
    # -------------------------------------------------------------------------
    
    # Set pandas display options
    corpus.col_width(DATAFRAME_COLUMN_WIDTH)
    
    # Prompt user for analysis mode
    mode = corpus.mode_definition()
    
    # Load metadata
    metadata_filepath = CORPUS_DIRECTORY + '0_Metadata.csv'
    metadata = corpus.read_metadata(metadata_filepath)
    
    # -------------------------------------------------------------------------
    # DATA LOADING (based on selected mode)
    # -------------------------------------------------------------------------
    
    if mode == 1:
        # Mode 1: Analyse specific composer(s)
        composers = corpus.composer_definition()
        composers = corpus.check_composers(metadata, composers)
        files = corpus.files_definition(CORPUS_DIRECTORY, composers)
        db = corpus.dataLoop_m1(CORPUS_DIRECTORY, composers, metadata, files)
    
    elif mode == 2:
        # Mode 2: Analyse entire corpus
        db = corpus.dataLoop_m2(CORPUS_DIRECTORY, metadata)
    
    # Estimate composition years for pieces with undefined dates
    db = corpus.year_definition(db)
    
    # -------------------------------------------------------------------------
    # SEGMENTATION AND FQS COMPUTATION
    # -------------------------------------------------------------------------
    
    if SEGMENTATION_STRATEGY == 'windowed':
        # Windowed analysis: divide into fixed-size segments
        db = segmentation.compute_window_sums(db, WINDOW_SIZE, OVERLAP_SIZE)
        
        # Compute FQS representation
        db = fqs.add_dft_magnitudes(db, source_column='windows')
        anchors = fqs.compute_radviz_anchors()
        db = fqs.compute_radviz_coordinates(db, anchors, DFT_COEFFICIENT_ORDER)
        
        # Compute qualia classifications and transition matrices
        (db, global_matrix, conditional_matrix, df_qualia, 
         n_transitions, qualia_list) = analysis.compute_qualia_matrix(
            db, anchors, NUM_ANCHORS, AMBIGUITY_RADIUS
        )
        
        # Visualise results
        if SHOW_QUALIA_SEQUENCE:
            segmentation.visualise_windowed_qualia(db, WINDOW_SIZE)
    
    elif SEGMENTATION_STRATEGY == 'distance-sensitive':
        # Distance-sensitive analysis: segment at detected peaks
        
        if DISTANCE_METHOD == 'DFTMag':
            # Use DFT magnitude space for distance calculation
            db = fqs.add_dft_magnitudes(db, source_column='Chords')
            anchors = fqs.compute_radviz_anchors()
            
            # Detect segmentation points based on DFT magnitude distances
            db = segmentation.compute_euclidean_distances(db, method='DFTMag')
            db = segmentation.detect_segmentation_peaks(
                db, method='DFTMag', show_graphs=SHOW_SEGMENTATION_GRAPHS
            )
            db = segmentation.aggregate_chords_by_segments(db)
            
            # Compute FQS for aggregated segments
            db = fqs.add_dft_magnitudes(db, source_column='ChordSections')
            db = fqs.compute_radviz_coordinates(db, anchors, DFT_COEFFICIENT_ORDER)
            
            # Compute qualia classifications and transition matrices
            (db, global_matrix, conditional_matrix, df_qualia, 
             n_transitions, qualia_list) = analysis.compute_qualia_matrix(
                db, anchors, NUM_ANCHORS, AMBIGUITY_RADIUS
            )
        
        elif DISTANCE_METHOD == 'RadViz':
            # Use RadViz space for distance calculation
            db = fqs.add_dft_magnitudes(db, source_column='Chords')
            anchors = fqs.compute_radviz_anchors()
            db = fqs.compute_radviz_coordinates(db, anchors, DFT_COEFFICIENT_ORDER)
            
            # Detect segmentation points based on RadViz distances
            db = segmentation.compute_euclidean_distances(db, method='RadViz')
            db = segmentation.detect_segmentation_peaks(
                db, method='RadViz', show_graphs=SHOW_SEGMENTATION_GRAPHS
            )
            db = segmentation.aggregate_chords_by_segments(db)
            
            # Recompute FQS for aggregated segments
            db = fqs.add_dft_magnitudes(db, source_column='ChordSections')
            db = fqs.compute_radviz_coordinates(db, anchors, DFT_COEFFICIENT_ORDER)
            
            # Compute qualia classifications and transition matrices
            (db, global_matrix, conditional_matrix, df_qualia, 
             n_transitions, qualia_list) = analysis.compute_qualia_matrix(
                db, anchors, NUM_ANCHORS, AMBIGUITY_RADIUS
            )
        
        # Visualise qualia sequence if requested
        if SHOW_QUALIA_SEQUENCE:
            segmentation.visualise_peak_based_qualia(db)
    
    # -------------------------------------------------------------------------
    # STATISTICAL ANALYSIS
    # -------------------------------------------------------------------------
    
    # Hierarchical clustering of qualia based on transition patterns
    linkage_matrix = analysis.compute_qualia_clustering(global_matrix)
    
    # Zipf distribution analysis
    ranked_counts = analysis.compute_ranked_counts(global_matrix, n_transitions)
    coefficients = analysis.compute_regression_coefficients(ranked_counts)
    
    # Generate regression line data for plotting
    line_func = np.poly1d(coefficients)
    x_line = np.linspace(
        min(ranked_counts['LogRank']), 
        max(ranked_counts['LogRank']), 
        100
    )
    y_line = line_func(x_line)
    
    # -------------------------------------------------------------------------
    # OUTPUT RESULTS
    # -------------------------------------------------------------------------
    
    analysis.print_analysis_report(
        db=db,
        linkage_matrix=linkage_matrix,
        global_matrix=global_matrix,
        conditional_matrix=conditional_matrix,
        df_qualia=df_qualia,
        ranked_counts=ranked_counts,
        coefficients=coefficients
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    main()
