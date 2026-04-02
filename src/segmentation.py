"""
Segmentation Utilities Module for Qualia Syntax Analysis
=========================================================

This module provides segmentation strategies for dividing musical pieces into
sections based on their harmonic content. Two main approaches are supported:

1. **Windowed Analysis**: Divides pieces into fixed-size windows with optional
   overlap, summing pitch class vectors within each window.

2. **Distance-Sensitive Analysis**: Uses signal processing techniques to detect
   significant harmonic changes (peaks in the variation of distances) and
   segments pieces at these transition points.

The resulting segments can then be analysed for their harmonic qualities
(qualia) using the Fourier Qualia Space representation.

Written for Python 3.11+
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.spatial import distance as spatial_distance
from tqdm import tqdm


# =============================================================================
# CONSTANTS
# =============================================================================

# Colour mapping for qualia types in visualisations
QUALIA_COLOURS = {
    'WT': 'red',      # Whole-tone
    'A': 'blue',      # Ambiguous (centre)
    'T': 'green',     # Tritone
    'DT': 'orange',   # Dominant-tonic
    'O': 'purple',    # Octatonic
    'C': 'brown',     # Chromatic
    'D': 'pink',      # Diatonic
    'Tr': 'gray'      # Transition
}


# =============================================================================
# WINDOWED ANALYSIS FUNCTIONS
# =============================================================================

def compute_window_sums(df: pd.DataFrame, 
                        window_size: int, 
                        overlap_size: int) -> pd.DataFrame:
    """
    Segment pieces using fixed-size windows and sum pitch class vectors.
    
    This function divides each piece into windows of a specified size and
    computes the sum of pitch class vectors within each window. This creates
    a coarser representation that smooths out individual chord variations.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'Chords' column with lists of pitch class vectors.
    window_size : int
        The number of chords per window. Must be greater than 6; even numbers
        are preferred.
    overlap_size : int
        The number of chords to advance between windows. If 0, windows are
        non-overlapping. If less than window_size, windows will overlap.
    
    Returns
    -------
    pd.DataFrame
        The input DataFrame with an added 'windows' column containing lists
        of summed pitch class vectors (one per window).
    
    Notes
    -----
    The window parameters control the temporal resolution of the analysis:
    - Larger windows provide more stable harmonic estimates but lose detail
    - Smaller windows capture more variation but may be noisy
    - Overlap creates smoother transitions between adjacent windows
    """
    # Validate and adjust window size
    if window_size % 2 != 0:
        window_size = window_size - 1  # Convert to even number
    if window_size <= 6:
        window_size = 8  # Minimum size for meaningful DFT analysis
    
    print('\n\nPerforming windowed analysis:')
    
    results = []
    for i in tqdm(range(len(df)), bar_format='{l_bar}{bar:50}{r_bar}', colour='#2a9d8f'):
        chords = df['Chords'].iloc[i]
        num_chords = len(chords)
        
        # Calculate window boundaries
        window_indices = []
        start = 0
        while start <= num_chords:
            end = min(start + window_size - 1, num_chords - 1)
            window_indices.append([start, end])
            
            # Check if we've reached the end of the piece
            if end >= num_chords - 1:
                break
            
            # Advance to next window
            if overlap_size == 0:
                start += window_size
            else:
                start += overlap_size
        
        # Sum pitch class vectors within each window
        window_sums = []
        for start_idx, end_idx in window_indices:
            window_chords = chords[start_idx:end_idx + 1]
            # Element-wise sum across all chords in the window
            summed = [sum(values) for values in zip(*window_chords)]
            window_sums.append(summed)
        
        results.append(window_sums)
    
    df['windows'] = results
    return df


def visualise_windowed_qualia(db: pd.DataFrame, window_size: int) -> None:
    """
    Visualise the qualia progression for windowed analysis.
    
    Creates a horizontal bar chart for each piece showing the sequence of
    harmonic qualities (qualia) with colour-coded regions proportional to
    their duration in the piece.
    
    Parameters
    ----------
    db : pd.DataFrame
        DataFrame containing 'Composer', 'Title', 'Chords', and 
        'QualiaProgression' columns.
    window_size : int
        The window size used in the analysis (for calculating segment lengths).
    """
    for idx, row in db.iterrows():
        qualia_progression = row['QualiaProgression']
        total_points = len(row['Chords'])
        
        # Group consecutive identical qualia
        grouped_qualia = _group_consecutive_elements(qualia_progression)
        
        # Calculate pixel positions for each group
        positions = _calculate_group_positions(grouped_qualia, window_size, total_points)
        
        # Create visualisation
        _create_qualia_visualisation(
            positions=positions,
            total_points=total_points,
            title=f"QualiaProgression for {row['Composer']} - {row['Title']}"
        )


def _group_consecutive_elements(sequence: List[str]) -> List[Tuple[str, int]]:
    """
    Group consecutive identical elements and count their occurrences.
    
    Parameters
    ----------
    sequence : List[str]
        A list of qualia labels.
    
    Returns
    -------
    List[Tuple[str, int]]
        A list of (qualia, count) tuples representing runs of identical values.
    """
    if not sequence:
        return []
    
    groups = []
    current_qualia = sequence[0]
    count = 1
    
    for i in range(1, len(sequence)):
        if sequence[i] == current_qualia:
            count += 1
        else:
            groups.append((current_qualia, count))
            current_qualia = sequence[i]
            count = 1
    
    groups.append((current_qualia, count))
    return groups


def _calculate_group_positions(grouped_qualia: List[Tuple[str, int]], 
                               window_size: int,
                               total_points: int) -> List[Tuple[int, int, str]]:
    """
    Calculate start and end positions for each qualia group.
    
    Parameters
    ----------
    grouped_qualia : List[Tuple[str, int]]
        List of (qualia, count) tuples from _group_consecutive_elements().
    window_size : int
        The window size used in analysis.
    total_points : int
        The total number of chord points in the piece.
    
    Returns
    -------
    List[Tuple[int, int, str]]
        A list of (start_pos, end_pos, qualia) tuples.
    """
    positions = []
    current_pos = 0
    
    for i, (qualia, count) in enumerate(grouped_qualia):
        start_pos = current_pos
        if i < len(grouped_qualia) - 1:
            end_pos = start_pos + count * window_size - 1
        else:
            end_pos = total_points - 1
        positions.append((start_pos, end_pos, qualia))
        current_pos = end_pos + 1
    
    # Ensure positions don't exceed total points
    positions = [
        (start, min(end, total_points - 1), qualia)
        for start, end, qualia in positions if start < total_points
    ]
    
    return positions


# =============================================================================
# DISTANCE-SENSITIVE ANALYSIS FUNCTIONS
# =============================================================================

def compute_euclidean_distances(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Calculate variation distances between consecutive points.
    
    For each point in the sequence (except first and last), computes the
    distance between the preceding and following points. This captures the
    rate of harmonic change at each position.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing either 'RadViz' or 'DFTMag' column depending
        on the method parameter.
    method : str
        The distance calculation method:
        - 'RadViz': Euclidean distance between 2D RadViz coordinates
        - 'DFTMag': Cosine distance between 6D DFT magnitude vectors
    
    Returns
    -------
    pd.DataFrame
        The input DataFrame with an added 'EuclideanDistances' column
        containing lists of distance values.
    
    Notes
    -----
    The distance at position n is calculated as d(p_{n-1}, p_{n+1}), i.e.,
    the distance between the point before and after position n. This measures
    how much the harmonic content changes around that position.
    
    First and last positions are assigned distance 0 as they have no
    predecessor/successor respectively.
    """
    df['EuclideanDistances'] = None
    
    for idx, row in df.iterrows():
        points = row[method]
        
        # First point has no predecessor
        distances = [0.0]
        
        # Calculate distances for middle points
        for i in range(1, len(points) - 1):
            p_next = np.array(points[i + 1])
            p_prev = np.array(points[i - 1])
            
            if method == 'RadViz':
                # Euclidean distance in 2D space
                distance = np.linalg.norm(p_next - p_prev)
            elif method == 'DFTMag':
                # Cosine distance in 6D space
                distance = spatial_distance.cosine(p_next, p_prev)
            
            distances.append(distance)
        
        # Last point has no successor
        if len(points) > 1:
            distances.append(0.0)
        
        df.at[idx, 'EuclideanDistances'] = distances
    
    return df


def detect_segmentation_peaks(db: pd.DataFrame, 
                              method: str = 'DFTMag',
                              show_graphs: bool = True) -> pd.DataFrame:
    """
    Detect peaks in the distance signal to identify segment boundaries.
    
    This function applies signal processing techniques to find significant
    peaks in the harmonic distance curve, which indicate transition points
    between distinct harmonic regions.
    
    Parameters
    ----------
    db : pd.DataFrame
        DataFrame containing 'EuclideanDistances', 'Composer', and 'Title' columns.
    method : str, optional
        The distance method used ('DFTMag' or 'RadViz'). Currently both use
        the same peak detection parameters (default: 'DFTMag').
    show_graphs : bool, optional
        Whether to display visualisations of the peak detection process
        (default: True).
    
    Returns
    -------
    pd.DataFrame
        The input DataFrame with an added 'PeakPositions' column containing
        lists of indices where peaks were detected.
    
    Notes
    -----
    Peak detection uses:
    1. Savitzky-Golay smoothing to reduce noise while preserving peak shapes
    2. Scipy's find_peaks with constraints on height, distance, prominence,
       and width to identify significant harmonic transitions
    3. Edge exclusion to avoid spurious peaks near piece boundaries
    """
    db_result = db.copy()
    db_result['PeakPositions'] = None
    
    # Peak detection parameters
    SAVGOL_WINDOW = 65      # Smoothing window (must be odd)
    SAVGOL_POLY_ORDER = 3   # Polynomial order for smoothing
    EDGE_THRESHOLD = 40     # Exclude peaks within this distance from edges
    
    for index, row in db_result.iterrows():
        composer = row['Composer']
        title = row['Title']
        distances = row['EuclideanDistances']
        total_points = len(distances)
        
        x_values = np.array(range(len(distances)))
        y_values = np.array(distances)
        peak_positions = []
        
        if show_graphs:
            fig, ax = plt.subplots(figsize=(18, 6))
            # Plot original data (semi-transparent)
            plt.plot(x_values, y_values, '-', markersize=1, alpha=0.2,
                     label='Original data', color='black', linewidth=0.5)
        
        # Apply smoothing if we have enough data points
        if len(distances) > 5:
            # Savitzky-Golay filter for noise reduction
            y_smooth = savgol_filter(y_values, SAVGOL_WINDOW, SAVGOL_POLY_ORDER, 
                                     mode='nearest')
            
            if show_graphs:
                plt.plot(x_values, y_smooth, '-', linewidth=1,
                         label='Smoothed curve', color='black', alpha=0.6)
            
            # Detect peaks with prominence-based filtering
            # Parameters are tuned for musical segmentation
            all_peaks, properties = find_peaks(
                y_smooth,
                height=0.05,       # Minimum peak height
                distance=25,       # Minimum samples between peaks
                prominence=0.05,   # Minimum peak prominence
                width=1            # Minimum peak width
            )
            
            all_peaks = sorted(all_peaks.tolist())
            
            # Filter peaks away from edges and preserve prominence information
            filtered_peaks = []
            filtered_prominences = []
            
            for i, peak in enumerate(all_peaks):
                if EDGE_THRESHOLD < peak < total_points - EDGE_THRESHOLD:
                    filtered_peaks.append(peak)
                    filtered_prominences.append(properties['prominences'][i])
            
            peak_positions = filtered_peaks
            
            if show_graphs and len(filtered_peaks) > 0:
                # Plot detected peaks
                plt.plot(filtered_peaks, y_smooth[filtered_peaks], 'x',
                         markersize=8, markeredgewidth=1.5, color='black',
                         label=f'Peaks ({len(filtered_peaks)})')
        
        else:
            # Not enough points for smoothing - just connect them
            if show_graphs:
                plt.plot(x_values, y_values, '-', linewidth=1,
                         label='Connected points', color='blue')
        
        # Finalise and display plot
        if show_graphs:
            ax.set_title(f"{composer}: {title}", fontsize=12)
            ax.set_xlabel('Sequence Position', fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            ax.set_xlim(0, total_points - 1)
            
            # Generate sensible tick positions
            tick_interval = max(1, total_points // 10)
            ax.set_xticks(range(0, total_points, tick_interval))
            
            # Minimal axis styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
            ax.set_yticklabels([])
            
            plt.subplots_adjust(left=0, right=1, bottom=0.15)
            plt.tight_layout(pad=0.5, rect=[0, 0.05, 1, 0.95])
            plt.show()
        
        db_result.at[index, 'PeakPositions'] = peak_positions
    
    return db_result


def aggregate_chords_by_segments(db: pd.DataFrame) -> pd.DataFrame:
    """
    Sum pitch class vectors within segments defined by peak positions.
    
    This function aggregates the individual chord vectors between detected
    peak positions, creating one summed vector per segment. The sums are
    then log-transformed to reduce the influence of segment length.
    
    Parameters
    ----------
    db : pd.DataFrame
        DataFrame containing 'Chords' and 'PeakPositions' columns.
    
    Returns
    -------
    pd.DataFrame
        The input DataFrame with an added 'ChordSections' column containing
        lists of summed (and log-transformed) pitch class vectors.
    
    Notes
    -----
    The log transformation uses log₂(x + 1) to:
    - Handle zero values (adding 1 avoids log(0))
    - Compress the range of values
    - Reduce the influence of very long segments
    """
    db_result = db.copy()
    db_result['ChordSections'] = None
    
    for index, row in db_result.iterrows():
        # Handle missing data
        if 'Chords' not in row or 'PeakPositions' not in row:
            db_result.at[index, 'ChordSections'] = []
            continue
        
        if row['Chords'] is None or len(row['Chords']) == 0:
            db_result.at[index, 'ChordSections'] = []
            continue
        
        chords = np.array(row['Chords'])
        peak_positions = row['PeakPositions'] if row['PeakPositions'] is not None else []
        
        # Define segment boundaries
        if not peak_positions:
            # No peaks: entire piece is one segment
            boundaries = [0, len(chords)]
        else:
            # Filter invalid peak positions
            valid_peaks = [
                p for p in peak_positions
                if isinstance(p, (int, np.integer)) and 0 <= p < len(chords)
            ]
            boundaries = [0] + sorted(valid_peaks) + [len(chords)]
        
        # Sum chords within each segment
        chord_sections = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            if start == end:
                # Empty segment
                chord_sections.append([0] * 12)
                continue
            
            # Sum and apply log transformation
            section_sum = np.sum(chords[start:end], axis=0).tolist()
            section_sum = [np.log2(val + 1) for val in section_sum]
            chord_sections.append(section_sum)
        
        db_result.at[index, 'ChordSections'] = chord_sections
    
    return db_result


# =============================================================================
# VISUALISATION FUNCTIONS
# =============================================================================

def visualise_peak_based_qualia(db: pd.DataFrame) -> None:
    """
    Visualise the qualia progression using peak-based segmentation.
    
    Creates a horizontal bar chart for each piece showing the sequence of
    harmonic qualities (qualia) with colour-coded regions. Vertical dashed
    lines indicate detected peak positions (segment boundaries).
    
    Parameters
    ----------
    db : pd.DataFrame
        DataFrame containing 'Composer', 'Title', 'QualiaProgression',
        'PeakPositions', and 'Chords' columns.
    """
    for idx, row in db.iterrows():
        composer = row['Composer']
        title = row['Title']
        qualia_progression = row['QualiaProgression']
        peak_positions = row['PeakPositions'] if row['PeakPositions'] is not None else []
        total_points = len(row['Chords'])
        
        if not qualia_progression:
            print(f"Skipping {composer} - {title}: No qualia progression data")
            continue
        
        # Calculate section positions based on peaks
        section_positions = _calculate_peak_based_positions(
            qualia_progression, peak_positions, total_points
        )
        
        # Create the visualisation
        _create_peak_visualisation(
            section_positions=section_positions,
            peak_positions=peak_positions,
            total_points=total_points,
            title=f"Peak-based QualiaProgression for {composer} - {title}"
        )


def _calculate_peak_based_positions(qualia_progression: List[str],
                                    peak_positions: List[int],
                                    total_points: int) -> List[Tuple[int, int, str]]:
    """
    Calculate section positions for peak-based visualisation.
    
    Parameters
    ----------
    qualia_progression : List[str]
        The sequence of qualia labels.
    peak_positions : List[int]
        The indices where peaks were detected.
    total_points : int
        The total number of points in the piece.
    
    Returns
    -------
    List[Tuple[int, int, str]]
        A list of (start, end, qualia) tuples for each section.
    """
    if not peak_positions:
        # No peaks: single section spanning entire piece
        if qualia_progression:
            return [(0, total_points, qualia_progression[0])]
        return []
    
    section_positions = []
    
    # First section: from start to first peak
    if len(qualia_progression) > 0:
        section_positions.append((0, peak_positions[0], qualia_progression[0]))
    
    # Middle sections: between consecutive peaks
    for i in range(1, min(len(qualia_progression) - 1, len(peak_positions))):
        if i - 1 < len(peak_positions) and i < len(peak_positions):
            start = peak_positions[i - 1] + 1
            end = peak_positions[i]
            section_positions.append((start, end, qualia_progression[i]))
    
    # Final section: from last peak to end
    if len(qualia_progression) > 1 and len(peak_positions) > 0:
        last_peak_idx = len(peak_positions) - 1
        last_qualia_idx = len(qualia_progression) - 1
        
        if last_peak_idx >= 0 and last_qualia_idx > 0:
            start = peak_positions[last_peak_idx] + 1
            end = total_points
            section_positions.append((start, end, qualia_progression[last_qualia_idx]))
    
    # Validate positions
    validated = []
    for start, end, qualia in section_positions:
        start = max(0, min(start, total_points))
        end = max(start, min(end, total_points))
        validated.append((start, end, qualia))
    
    return validated


def _create_qualia_visualisation(positions: List[Tuple[int, int, str]],
                                 total_points: int,
                                 title: str) -> None:
    """
    Create a basic qualia progression visualisation.
    
    Parameters
    ----------
    positions : List[Tuple[int, int, str]]
        List of (start, end, qualia) tuples.
    total_points : int
        Total number of points for x-axis scaling.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Draw coloured regions
    for start, end, qualia in positions:
        ax.axvspan(start, end, color=QUALIA_COLOURS.get(qualia, 'gray'), alpha=0.5)
        ax.text((start + end) / 2, 0.5, qualia,
                horizontalalignment='center', verticalalignment='center')
    
    # Configure axes
    tick_interval = max(1, total_points // 10)
    ax.set_xticks(range(0, total_points, tick_interval))
    ax.set_xticklabels(range(0, total_points, tick_interval))
    ax.set_yticks([])
    ax.set_xlim(0, total_points)
    ax.set_title(title)
    
    plt.subplots_adjust(left=0, right=1, bottom=0.15)
    plt.tight_layout(pad=0.5, rect=[0, 0.05, 1, 0.95])
    plt.show()


def _create_peak_visualisation(section_positions: List[Tuple[int, int, str]],
                               peak_positions: List[int],
                               total_points: int,
                               title: str) -> None:
    """
    Create a peak-based qualia progression visualisation with boundary markers.
    
    Parameters
    ----------
    section_positions : List[Tuple[int, int, str]]
        List of (start, end, qualia) tuples.
    peak_positions : List[int]
        Indices where peaks were detected.
    total_points : int
        Total number of points for x-axis scaling.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Draw coloured regions with qualia labels
    prev_qualia = None
    for start, end, qualia in section_positions:
        ax.axvspan(start, end, color=QUALIA_COLOURS.get(qualia, 'gray'), alpha=0.5)
        # Only label if different from previous (avoid clutter)
        if qualia != prev_qualia:
            ax.text((start + end) / 2, 0.5, qualia,
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, fontweight='bold')
        prev_qualia = qualia
    
    # Draw vertical lines at peak positions
    for pos in sorted(peak_positions):
        ax.axvline(x=pos, color='black', linestyle='--', linewidth=1)
        ax.text(pos, 0.9, f"{pos}",
                horizontalalignment='center', verticalalignment='top',
                fontsize=8, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Configure axes
    tick_interval = max(1, total_points // 10)
    ax.set_xticks(range(0, total_points, tick_interval))
    ax.set_xticklabels(range(0, total_points, tick_interval))
    ax.set_yticks([])
    ax.set_xlim(0, total_points)
    ax.set_title(title, fontsize=14)
    
    plt.subplots_adjust(left=0, right=1, bottom=0.15)
    plt.tight_layout(pad=0.5, rect=[0, 0.05, 1, 0.95])
    plt.show()
