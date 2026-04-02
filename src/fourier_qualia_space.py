"""
Fourier Qualia Space (FQS) Utilities Module
============================================

This module provides functionality for computing the Fourier Qualia Space (FQS)
representation of musical data. The FQS is a geometric representation that
maps pitch class distributions to a 2D space using:

1. **Discrete Fourier Transform (DFT)**: Transforms pitch class vectors into
   frequency-domain representations, capturing harmonic properties.

2. **RadViz Projection**: Projects the 6 non-trivial DFT coefficient magnitudes
   onto a 2D unit circle using the RadViz visualisation technique.

The resulting 2D coordinates encode harmonic qualities (qualia) that can be
used for subsequent analysis of harmonic progressions.

Written for Python 3.11+
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# DISCRETE FOURIER TRANSFORM FUNCTIONS
# =============================================================================

def compute_dft(pitch_class_vector: List[float]) -> List[complex]:
    """
    Compute the Discrete Fourier Transform of a pitch class vector.
    
    The DFT transforms a 12-element pitch class distribution into the frequency
    domain, revealing harmonic properties such as interval content and chord
    quality characteristics.
    
    Parameters
    ----------
    pitch_class_vector : List[float]
        A 12-element vector where each element represents the weight or count
        of a pitch class (C=0, C#=1, ..., B=11).
    
    Returns
    -------
    List[complex]
        The 12-element DFT of the input vector, containing complex coefficients.
    
    Notes
    -----
    The DFT coefficients have the following musical interpretations:
    - f₁: Chromaticity (uneven distribution)
    - f₂: Major/minor third balance
    - f₃: Major third content (augmented quality)
    - f₄: Minor third content (diminished quality)
    - f₅: Diatonic quality
    - f₆: Tritone content (whole-tone quality)
    """
    dft_result = np.fft.fft(pitch_class_vector)
    return dft_result.tolist()


def add_dft_magnitudes(df: pd.DataFrame, source_column: str) -> pd.DataFrame:
    """
    Compute DFT coefficient magnitudes for all pitch class vectors in a DataFrame.
    
    This function calculates the DFT for each pitch class vector and extracts
    the magnitudes of the 6 non-trivial coefficients (f₁ through f₆), which
    encode musically meaningful harmonic properties.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a column with lists of pitch class vectors.
    source_column : str
        The name of the column containing pitch class vectors. Each cell should
        contain a list of 12-element vectors.
    
    Returns
    -------
    pd.DataFrame
        The input DataFrame with an added 'DFTMag' column containing lists of
        6-element magnitude vectors (one per input vector).
    
    Notes
    -----
    Only coefficients f₁-f₆ are retained because:
    - f₀ represents the total energy (sum of all pitch classes), not harmonic quality
    - f₇-f₁₁ are conjugate symmetric with f₅-f₁, providing no additional information
    """
    print('\n\nDFT Computation')
    
    # Step 1: Compute DFT magnitudes for each vector
    print(' - Computing DFT coefficients magnitude:')
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#fff3b0')
    df['DFTMag'] = df[source_column].progress_apply(
        lambda vectors: [np.abs(compute_dft(vec)) for vec in vectors]
    )
    
    # Step 2: Extract only the relevant coefficients (indices 1-6)
    # These correspond to f₁ through f₆, the non-trivial DFT components
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#fff3b0')
    df['DFTMag'] = df['DFTMag'].progress_apply(
        lambda magnitudes: [mag[1:7] for mag in magnitudes]
    )
    
    return df


# =============================================================================
# RADVIZ PROJECTION FUNCTIONS
# =============================================================================

def compute_radviz_anchors(num_anchors: int = 6) -> List[List[float]]:
    """
    Calculate equidistant anchor coordinates on a unit circle for RadViz.
    
    RadViz is a visualisation technique that projects multidimensional data
    onto a 2D plane using anchor points arranged on a circle. Each data
    dimension is associated with an anchor, and data points are positioned
    based on a spring-force metaphor.
    
    Parameters
    ----------
    num_anchors : int, optional
        The number of anchor points (default: 6 for the 6 DFT coefficients).
    
    Returns
    -------
    List[List[float]]
        A list of [x, y] coordinate pairs for each anchor, evenly distributed
        around a unit circle starting from angle 0 (positive x-axis).
    
    Notes
    -----
    For the standard FQS representation with 6 anchors, the anchors are
    positioned at angles: 0°, 60°, 120°, 180°, 240°, 300°.
    """
    angle_step = 2 * np.pi / num_anchors
    anchors = [
        [np.cos(i * angle_step), np.sin(i * angle_step)]
        for i in range(num_anchors)
    ]
    return anchors


def compute_radviz_coordinates(df: pd.DataFrame, 
                               anchors: List[List[float]],
                               coefficient_order: List[int]) -> pd.DataFrame:
    """
    Project DFT magnitudes onto 2D coordinates using the RadViz technique.
    
    This function maps the 6-dimensional DFT magnitude vectors to 2D points
    on the unit circle. The position of each point is determined by treating
    the DFT magnitudes as weights that pull the point towards the corresponding
    anchor positions.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'DFTMag' column with lists of 6-element
        magnitude vectors.
    anchors : List[List[float]]
        The RadViz anchor coordinates from compute_radviz_anchors().
    coefficient_order : List[int]
        The order in which DFT coefficients should be mapped to anchors.
        For example, [1, 5, 3, 4, 6, 2] assigns f₁ to anchor 0, f₅ to anchor 1,
        etc. This ordering affects the geometric interpretation of regions.
    
    Returns
    -------
    pd.DataFrame
        The input DataFrame with added columns:
        - 'Order': DFT magnitudes rearranged according to coefficient_order
        - 'RadViz': List of [x, y] coordinates for each vector
    
    Notes
    -----
    The RadViz formula for a point with weights w₁, ..., wₙ and anchor
    positions a₁, ..., aₙ is:
    
        point = Σ(wᵢ * aᵢ) / Σ(wᵢ)
    
    This creates a weighted centroid where stronger DFT components pull the
    point closer to their associated anchors.
    """
    num_anchors = len(anchors)
    
    print('\n\nRadViz Data')
    
    # Step 1: Rearrange DFT magnitudes to the specified coefficient order
    print(' - Rearranging elements to RadViz order:')
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#7adfbb')
    df['Order'] = df['DFTMag'].progress_apply(
        lambda magnitudes: [
            [magnitudes[i][j - 1] for j in coefficient_order]  # j-1 for 0-indexing
            for i in range(len(magnitudes))
        ]
    )
    
    # Step 2: Compute RadViz coordinates using weighted centroid formula
    print(' - Computing RadViz coordinates:')
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#7adfbb')
    df['RadViz'] = df['Order'].progress_apply(
        lambda weights_list: [
            [
                # X coordinate: weighted sum of anchor x-coordinates
                sum(anchors[j][0] * weights[j] for j in range(num_anchors)) / sum(weights),
                # Y coordinate: weighted sum of anchor y-coordinates
                sum(anchors[j][1] * weights[j] for j in range(num_anchors)) / sum(weights)
            ]
            for weights in weights_list
        ]
    )
    
    return df


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_fqs_representation(df: pd.DataFrame,
                               source_column: str,
                               coefficient_order: List[int]) -> Tuple[pd.DataFrame, List[List[float]]]:
    """
    Compute the complete Fourier Qualia Space representation for a corpus.
    
    This convenience function chains together all FQS computation steps:
    1. Compute DFT magnitudes from pitch class vectors
    2. Calculate RadViz anchor positions
    3. Project DFT magnitudes to 2D RadViz coordinates
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a column with lists of pitch class vectors.
    source_column : str
        The name of the column containing pitch class vectors.
    coefficient_order : List[int]
        The order in which DFT coefficients map to RadViz anchors.
    
    Returns
    -------
    Tuple[pd.DataFrame, List[List[float]]]
        A tuple containing:
        - The DataFrame with added 'DFTMag', 'Order', and 'RadViz' columns
        - The list of RadViz anchor coordinates
    
    Example
    -------
    >>> df, anchors = compute_fqs_representation(
    ...     df, 
    ...     source_column='Chords', 
    ...     coefficient_order=[1, 5, 3, 4, 6, 2]
    ... )
    """
    # Compute DFT magnitudes
    df = add_dft_magnitudes(df, source_column)
    
    # Generate RadViz anchors
    anchors = compute_radviz_anchors(num_anchors=6)
    
    # Project to RadViz coordinates
    df = compute_radviz_coordinates(df, anchors, coefficient_order)
    
    return df, anchors
