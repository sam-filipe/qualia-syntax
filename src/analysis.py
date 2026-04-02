"""
Analysis Utilities Module for Qualia Syntax Analysis
=====================================================

This module provides statistical and analytical tools for studying harmonic
progressions represented in the Fourier Qualia Space. It includes:

1. **Qualia Classification**: Assigns harmonic quality labels to FQS points
   based on their position in the RadViz space.

2. **Transition Matrices**: Computes first-order and higher-order Markov
   transition probabilities between qualia.

3. **Entropy Analysis**: Calculates information-theoretic measures of
   harmonic predictability.

4. **Hierarchical Clustering**: Groups qualia based on their transition
   patterns using agglomerative clustering.

5. **Zipf Analysis**: Examines the frequency distribution of transitions
   for evidence of statistical regularities.

Written for Python 3.11+
"""

import itertools as it
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from tqdm import tqdm


# =============================================================================
# CONSTANTS
# =============================================================================

# The seven qualia types representing distinct harmonic regions
ALL_QUALIA_TYPES = ['DT', 'T', 'O', 'WT', 'D', 'C', 'A']


# =============================================================================
# QUALIA CLASSIFICATION FUNCTIONS
# =============================================================================

def classify_point(x: float, y: float, 
                   ambiguity_radius: float,
                   regions: Dict[str, List[List[float]]]) -> str:
    """
    Classify a single RadViz point into a qualia category.
    
    Points within the central ambiguity circle are classified as 'A' (ambiguous).
    Points outside are classified based on which triangular region they fall into.
    
    Parameters
    ----------
    x : float
        The x-coordinate of the point in RadViz space.
    y : float
        The y-coordinate of the point in RadViz space.
    ambiguity_radius : float
        The radius of the central ambiguity region. Points closer to the
        centre than this are classified as ambiguous.
    regions : Dict[str, List[List[float]]]
        A dictionary mapping qualia names to lists of vertices defining
        their triangular regions.
    
    Returns
    -------
    str
        The qualia label ('C', 'DT', 'T', 'O', 'WT', 'D', or 'A').
    """
    # Check if point is within the ambiguity circle
    if x ** 2 + y ** 2 <= ambiguity_radius ** 2:
        return 'A'
    
    # Check which triangular region contains the point
    for region_name, region_vertices in regions.items():
        path = mplPath.Path(region_vertices)
        if path.contains_point([x, y]):
            return region_name
    
    # Fallback (should not normally occur)
    return 'A'


def compute_qualia_progression(points: List[List[float]], 
                               ambiguity_radius: float,
                               regions: Dict[str, List[List[float]]]) -> List[str]:
    """
    Classify a sequence of RadViz points into qualia categories.
    
    Parameters
    ----------
    points : List[List[float]]
        A list of [x, y] coordinate pairs in RadViz space.
    ambiguity_radius : float
        The radius of the central ambiguity region.
    regions : Dict[str, List[List[float]]]
        A dictionary mapping qualia names to region vertices.
    
    Returns
    -------
    List[str]
        A list of qualia labels corresponding to each input point.
    """
    return [
        classify_point(x, y, ambiguity_radius, regions)
        for x, y in points
    ]


def _compute_region_boundaries(anchors: List[List[float]], 
                               num_anchors: int) -> Dict[str, List[List[float]]]:
    """
    Calculate the triangular region boundaries for qualia classification.
    
    Each region is a triangle with vertices at:
    - The origin (0, 0)
    - The midpoint between two adjacent anchors
    - An anchor point
    - The midpoint between that anchor and the next anchor
    
    Parameters
    ----------
    anchors : List[List[float]]
        The RadViz anchor coordinates.
    num_anchors : int
        The number of anchors (should be 6 for standard FQS).
    
    Returns
    -------
    Dict[str, List[List[float]]]
        A dictionary mapping qualia names to lists of four vertices.
    """
    # Calculate midpoints between adjacent anchors
    mid_points = [
        [(anchors[i][j] + anchors[(i + 1) % num_anchors][j]) / 2 for j in range(2)]
        for i in range(num_anchors)
    ]
    
    # Define triangular regions for each qualia
    # Note: The order corresponds to the DFT coefficient ordering
    regions = {
        "C": [[0, 0], mid_points[5], anchors[0], mid_points[0]],
        "DT": [[0, 0], mid_points[0], anchors[1], mid_points[1]],
        "T": [[0, 0], mid_points[1], anchors[2], mid_points[2]],
        "O": [[0, 0], mid_points[2], anchors[3], mid_points[3]],
        "WT": [[0, 0], mid_points[3], anchors[4], mid_points[4]],
        "D": [[0, 0], mid_points[4], anchors[5], mid_points[5]]
    }
    
    return regions


# =============================================================================
# TRANSITION MATRIX FUNCTIONS
# =============================================================================

def compute_qualia_matrix(df: pd.DataFrame, 
                          anchors: List[List[float]],
                          num_anchors: int,
                          ambiguity_radius: float) -> Tuple:
    """
    Compute qualia progressions and first-order transition matrices.
    
    This function:
    1. Classifies each RadViz point into a qualia category
    2. Constructs a global transition probability matrix
    3. Constructs a conditional (row-normalised) transition matrix
    4. Summarises antecedent/consequent relationships
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'RadViz' column with point coordinates.
    anchors : List[List[float]]
        The RadViz anchor coordinates.
    num_anchors : int
        The number of anchors (6 for standard FQS).
    ambiguity_radius : float
        The radius of the central ambiguity region.
    
    Returns
    -------
    Tuple containing:
        - df: Updated DataFrame with 'QualiaProgression' column
        - global_matrix: DataFrame with global transition percentages
        - conditional_matrix: DataFrame with conditional probabilities
        - df_qualia: DataFrame summarising antecedent/consequent relationships
        - n_transitions: Total number of transitions
        - qualia_list: Flattened list of all qualia (with consecutive duplicates removed)
    """
    # Compute region boundaries
    regions = _compute_region_boundaries(anchors, num_anchors)
    
    print('\n\nQualia Matrix')
    print(' - Computing Qualia Progression')
    
    # Classify all points into qualia
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#fff3b0')
    df['QualiaProgression'] = df['RadViz'].progress_apply(
        lambda points: compute_qualia_progression(points, ambiguity_radius, regions)
    )
    
    # Flatten all progressions into a single list
    qualia_list = []
    for progression in df['QualiaProgression']:
        qualia_list.extend(progression)
    
    # Remove consecutive duplicates (we're interested in transitions between different qualia)
    qualia_list = [
        qualia_list[i] for i in range(len(qualia_list))
        if i == 0 or qualia_list[i] != qualia_list[i - 1]
    ]
    
    # Count transitions
    transitions = defaultdict(lambda: defaultdict(int))
    for i in range(len(qualia_list) - 1):
        transitions[qualia_list[i]][qualia_list[i + 1]] += 1
    
    # Compute global percentage matrix
    total_transitions = len(qualia_list) - 1
    global_percentages = defaultdict(lambda: defaultdict(float))
    for qualia, next_qualias in transitions.items():
        for next_qualia, count in next_qualias.items():
            global_percentages[qualia][next_qualia] = (count / total_transitions) * 100
    
    global_matrix = pd.DataFrame(global_percentages).T
    
    # Compute conditional probability matrix (row-normalised)
    conditional_probs = defaultdict(lambda: defaultdict(float))
    for qualia, next_qualias in transitions.items():
        total_from_qualia = sum(next_qualias.values())
        for next_qualia, count in next_qualias.items():
            conditional_probs[qualia][next_qualia] = (count / total_from_qualia) * 100
    
    conditional_matrix = pd.DataFrame(conditional_probs).T
    
    # Format matrices with consistent ordering
    global_matrix = _format_transition_matrix(global_matrix, ALL_QUALIA_TYPES)
    conditional_matrix = _format_transition_matrix(conditional_matrix, ALL_QUALIA_TYPES)
    
    # Summarise antecedent/consequent relationships
    df_qualia = _summarise_qualia_relationships(qualia_list)
    
    n_transitions = total_transitions
    
    return df, global_matrix, conditional_matrix, df_qualia, n_transitions, qualia_list


def _format_transition_matrix(matrix: pd.DataFrame, 
                              qualia_order: List[str]) -> pd.DataFrame:
    """
    Format a transition matrix with consistent row/column ordering.
    
    Parameters
    ----------
    matrix : pd.DataFrame
        The raw transition matrix.
    qualia_order : List[str]
        The desired order for rows and columns.
    
    Returns
    -------
    pd.DataFrame
        The formatted matrix with NaN for missing transitions.
    """
    # Add missing columns
    for col in qualia_order:
        if col not in matrix.columns:
            matrix[col] = np.nan
    
    # Reorder columns
    available_columns = [col for col in qualia_order if col in matrix.columns]
    matrix = matrix[available_columns]
    
    # Reorder rows
    available_rows = [row for row in qualia_order if row in matrix.index]
    matrix = matrix.reindex(available_rows)
    
    return matrix


def _summarise_qualia_relationships(qualia_list: List[str]) -> pd.DataFrame:
    """
    Summarise unique antecedent and consequent relationships for each qualia.
    
    Parameters
    ----------
    qualia_list : List[str]
        The sequence of qualia (consecutive duplicates already removed).
    
    Returns
    -------
    pd.DataFrame
        A DataFrame with 'antecedents' and 'consequents' columns showing
        the unique qualia that precede/follow each qualia type.
    """
    state_dict = {}
    
    for i, state in enumerate(qualia_list):
        if state not in state_dict:
            state_dict[state] = {"antecedents": set(), "consequents": set()}
        
        if i > 0:
            state_dict[state]["antecedents"].add(qualia_list[i - 1])
        if i < len(qualia_list) - 1:
            state_dict[state]["consequents"].add(qualia_list[i + 1])
    
    df_qualia = pd.DataFrame.from_dict(state_dict, orient='index')
    
    if df_qualia.empty:
        print("Warning: No qualia transitions found in dataset")
        return pd.DataFrame(columns=['antecedents', 'consequents'])
    
    # Format as "count, [list]"
    if 'antecedents' in df_qualia.columns:
        df_qualia['antecedents'] = df_qualia['antecedents'].apply(
            lambda x: f"{len(x)}, {list(x)}"
        )
    else:
        df_qualia['antecedents'] = [[]] * len(df_qualia)
    
    if 'consequents' in df_qualia.columns:
        df_qualia['consequents'] = df_qualia['consequents'].apply(
            lambda x: f"{len(x)}, {list(x)}"
        )
    else:
        df_qualia['consequents'] = [[]] * len(df_qualia)
    
    # Reorder rows
    available_qualia = [q for q in ALL_QUALIA_TYPES if q in df_qualia.index]
    if available_qualia:
        df_qualia = df_qualia.loc[available_qualia]
    
    return df_qualia


def compute_higher_order_matrix(df: pd.DataFrame, order: int) -> Tuple[pd.DataFrame, int]:
    """
    Compute a higher-order Markov transition matrix.
    
    A higher-order matrix considers sequences of qualia as antecedents,
    allowing analysis of dependencies beyond immediate neighbours.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'QualiaProgression' column.
    order : int
        The order of the transition matrix (number of antecedent qualia).
    
    Returns
    -------
    Tuple containing:
        - matrix: DataFrame with transition probabilities
        - n_transitions: Total number of transitions
    
    Example
    -------
    For order=2, transitions are of the form (q1, q2) -> q3
    """
    qualities = ALL_QUALIA_TYPES
    
    # Generate all possible antecedent sequences
    antecedents = [' '.join(x) for x in it.product(qualities, repeat=order)]
    
    # Initialise matrix
    matrix = pd.DataFrame(0, index=antecedents, columns=qualities)
    
    # Flatten and deduplicate qualia progression
    qualia_list = list(it.chain.from_iterable(df['QualiaProgression']))
    qualia_list = [
        qualia_list[i] for i in range(len(qualia_list))
        if i == 0 or qualia_list[i] != qualia_list[i - 1]
    ]
    
    # Count higher-order transitions
    for i in range(len(qualia_list) - order):
        antecedent = ' '.join(qualia_list[i:i + order])
        consequent = qualia_list[i + order]
        matrix.loc[antecedent, consequent] += 1
    
    # Convert to probabilities
    n_transitions = matrix.sum().sum()
    matrix = matrix / n_transitions * 100
    
    return matrix, int(n_transitions)


# =============================================================================
# HIERARCHICAL CLUSTERING FUNCTIONS
# =============================================================================

def compute_qualia_clustering(matrix: pd.DataFrame) -> np.ndarray:
    """
    Perform hierarchical clustering on qualia based on transition patterns.
    
    Uses Ward's method with Euclidean distance to cluster qualia that have
    similar transition probability profiles.
    
    Parameters
    ----------
    matrix : pd.DataFrame
        A transition matrix (either global or conditional).
    
    Returns
    -------
    np.ndarray
        The linkage matrix for use with scipy's dendrogram function.
    """
    # Handle missing values
    matrix = matrix.fillna(0)
    
    # Compute linkage using Ward's method
    linkage_matrix = linkage(matrix, method='ward', metric='euclidean')
    
    return linkage_matrix


def plot_qualia_dendrogram(linkage_matrix: np.ndarray, matrix: pd.DataFrame) -> None:
    """
    Plot a dendrogram showing qualia clustering.
    
    Parameters
    ----------
    linkage_matrix : np.ndarray
        The linkage matrix from compute_qualia_clustering().
    matrix : pd.DataFrame
        The original matrix (used for row labels).
    """
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, labels=matrix.index, leaf_rotation=90)
    plt.title('Qualia Agglomerative HCA Dendrogram')
    plt.ylabel('Euclidean distance')
    plt.xlabel('Qualia')
    plt.grid(axis='y')
    plt.show()


# =============================================================================
# ZIPF ANALYSIS FUNCTIONS
# =============================================================================

def compute_ranked_counts(matrix: pd.DataFrame, n_transitions: int) -> pd.DataFrame:
    """
    Compute ranked transition counts for Zipf analysis.
    
    Ranks all possible transitions by frequency and computes cumulative
    frequencies for power-law analysis.
    
    Parameters
    ----------
    matrix : pd.DataFrame
        A global transition probability matrix.
    n_transitions : int
        The total number of transitions.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing:
        - 'Rank': The frequency rank (1 = most common)
        - 'Transition': The transition (e.g., 'C --> DT')
        - 'Count': The raw count
        - 'Frequency': The percentage
        - 'CumFrequency': Cumulative percentage
        - 'LogRank': Natural log of rank
        - 'LogFrequency': Natural log of frequency
    """
    # Fill missing values
    matrix = matrix.fillna(0)
    
    # Reshape to long format
    ranked = matrix.reset_index().melt(
        id_vars='index', var_name='Consequent', value_name='Frequency'
    )
    
    # Create transition label
    ranked['Transition'] = ranked['index'] + ' --> ' + ranked['Consequent']
    ranked = ranked.drop(columns=['index', 'Consequent'])
    
    # Sort by frequency and assign ranks
    ranked = ranked.sort_values('Frequency', ascending=False)
    ranked['Rank'] = range(1, len(ranked) + 1)
    
    # Calculate count and cumulative frequency
    ranked['Count'] = ranked['Frequency'] * n_transitions / 100
    ranked['CumFrequency'] = ranked['Frequency'].cumsum()
    
    # Reorder columns
    ranked = ranked[['Rank', 'Transition', 'Count', 'Frequency', 'CumFrequency']]
    
    # Remove zero-frequency transitions for log calculations
    ranked = ranked[ranked['Frequency'] > 0]
    
    # Add log columns for Zipf analysis
    ranked['LogRank'] = np.log(ranked['Rank'])
    ranked['LogFrequency'] = np.log(ranked['Frequency'])
    
    return ranked


def compute_regression_coefficients(ranked_counts: pd.DataFrame) -> np.ndarray:
    """
    Fit a linear regression to the log-log rank-frequency data.
    
    A slope close to -1 indicates Zipf's law behaviour.
    
    Parameters
    ----------
    ranked_counts : pd.DataFrame
        Output from compute_ranked_counts().
    
    Returns
    -------
    np.ndarray
        Coefficients [slope, intercept] of the linear fit.
    """
    coefficients = np.polyfit(
        ranked_counts['LogRank'], 
        ranked_counts['LogFrequency'], 
        1
    )
    return coefficients


def plot_zipf_analysis(ranked_counts: pd.DataFrame, 
                       coefficients: np.ndarray) -> None:
    """
    Plot the log-log rank-frequency relationship with regression line.
    
    Parameters
    ----------
    ranked_counts : pd.DataFrame
        Output from compute_ranked_counts().
    coefficients : np.ndarray
        Output from compute_regression_coefficients().
    """
    # Generate regression line
    line_func = np.poly1d(coefficients)
    x_line = np.linspace(
        min(ranked_counts['LogRank']), 
        max(ranked_counts['LogRank']), 
        100
    )
    y_line = line_func(x_line)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.scatter(ranked_counts['LogRank'], ranked_counts['LogFrequency'], 
                color='blue', s=10, label='Data')
    plt.plot(x_line, y_line, color='red', linestyle='--', 
             label=f'Fit (slope={coefficients[0]:.3f})')
    plt.xlabel('Log(Rank)')
    plt.ylabel('Log(Frequency)')
    plt.title('Log-Log Plot of Rank vs Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_cumulative_frequency(ranked_counts: pd.DataFrame) -> None:
    """
    Plot the cumulative frequency distribution.
    
    Parameters
    ----------
    ranked_counts : pd.DataFrame
        Output from compute_ranked_counts().
    """
    plt.figure(figsize=(12, 8))
    plt.plot(ranked_counts['Rank'], ranked_counts['CumFrequency'], color='blue')
    plt.xlabel('Rank')
    plt.ylabel('Cumulative Frequency (%)')
    plt.title('Cumulative Frequency Plot')
    plt.grid(True)
    plt.show()


# =============================================================================
# ENTROPY ANALYSIS FUNCTIONS
# =============================================================================

def compute_first_order_entropy(sequence: List[str]) -> float:
    """
    Calculate the first-order (unigram) entropy of a sequence.
    
    First-order entropy measures the average information content per symbol,
    based only on individual symbol frequencies.
    
    Parameters
    ----------
    sequence : List[str]
        A sequence of qualia labels.
    
    Returns
    -------
    float
        The entropy in bits.
    """
    counts = Counter(sequence)
    total = len(sequence)
    probabilities = {k: v / total for k, v in counts.items()}
    return -sum(p * math.log2(p) for p in probabilities.values())


def compute_conditional_entropy(sequence: List[str]) -> float:
    """
    Calculate the conditional entropy H(Y|X) for bigrams.
    
    Measures the average uncertainty about the next symbol given the current
    symbol. Lower values indicate more predictable sequences.
    
    Parameters
    ----------
    sequence : List[str]
        A sequence of qualia labels.
    
    Returns
    -------
    float
        The conditional entropy in bits.
    """
    bigram_counts = Counter(zip(sequence, sequence[1:]))
    total_bigrams = len(sequence) - 1
    bigram_probs = {k: v / total_bigrams for k, v in bigram_counts.items()}
    
    conditional_probs = {}
    for (x, y), p_xy in bigram_probs.items():
        if x not in conditional_probs:
            conditional_probs[x] = {}
        # P(Y|X) = P(X,Y) / P(X)
        p_x = Counter(sequence)[x] / len(sequence)
        conditional_probs[x][y] = p_xy / p_x
    
    return -sum(
        p_xy * math.log2(conditional_probs[x][y])
        for (x, y), p_xy in bigram_probs.items()
    )


def compute_mutual_information(sequence: List[str]) -> float:
    """
    Calculate the mutual information I(X; Y) between adjacent positions.
    
    Measures how much information the current symbol provides about the next
    symbol. Higher values indicate stronger dependencies.
    
    Parameters
    ----------
    sequence : List[str]
        A sequence of qualia labels.
    
    Returns
    -------
    float
        The mutual information in bits.
    """
    h_x = compute_first_order_entropy(sequence)
    h_y = compute_first_order_entropy(sequence[1:])
    
    # H(X, Y) - joint entropy
    bigram_counts = Counter(zip(sequence, sequence[1:]))
    total_bigrams = len(sequence) - 1
    h_xy = -sum(
        (v / total_bigrams) * math.log2(v / total_bigrams)
        for v in bigram_counts.values()
    )
    
    # I(X; Y) = H(X) + H(Y) - H(X, Y)
    return h_x + h_y - h_xy


# =============================================================================
# N-GRAM ANALYSIS FUNCTIONS
# =============================================================================

def compute_ngram_statistics(qualia_list: List[str], 
                             n: int) -> Tuple[Dict[Tuple[str, ...], int], 
                                              Dict[Tuple[str, ...], float]]:
    """
    Compute n-gram counts and frequencies.
    
    Parameters
    ----------
    qualia_list : List[str]
        A sequence of qualia labels.
    n : int
        The n-gram order (e.g., 2 for bigrams, 3 for trigrams).
    
    Returns
    -------
    Tuple containing:
        - ngram_counts: Dictionary mapping n-grams to counts
        - ngram_frequencies: Dictionary mapping n-grams to relative frequencies
    """
    # Generate n-grams
    ngrams = [tuple(qualia_list[i:i + n]) for i in range(len(qualia_list) - n + 1)]
    
    # Count occurrences
    ngram_counts = Counter(ngrams)
    
    # Calculate frequencies
    total = sum(ngram_counts.values())
    ngram_frequencies = {ngram: count / total for ngram, count in ngram_counts.items()}
    
    return dict(ngram_counts), ngram_frequencies


def print_top_ngrams(ngram_counts: Dict[Tuple[str, ...], int],
                     ngram_frequencies: Dict[Tuple[str, ...], float],
                     top_n: int = 10) -> None:
    """
    Print the most frequent n-grams with their counts and frequencies.
    
    Parameters
    ----------
    ngram_counts : Dict[Tuple[str, ...], int]
        Dictionary of n-gram counts.
    ngram_frequencies : Dict[Tuple[str, ...], float]
        Dictionary of n-gram frequencies.
    top_n : int, optional
        Number of top n-grams to display (default: 10).
    """
    sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Top {top_n} n-grams:")
    print(f"{'N-gram':<15} {'Count':<10} {'Frequency':<10}")
    print("-" * 35)
    
    for ngram, count in sorted_ngrams[:top_n]:
        frequency = ngram_frequencies[ngram]
        ngram_str = ' '.join(ngram)
        print(f"{ngram_str:<15} {count:<10} {frequency:.4f}")


# =============================================================================
# REPORTING FUNCTIONS
# =============================================================================

def print_analysis_report(db: pd.DataFrame,
                          linkage_matrix: np.ndarray,
                          global_matrix: pd.DataFrame,
                          conditional_matrix: pd.DataFrame,
                          df_qualia: pd.DataFrame,
                          ranked_counts: pd.DataFrame,
                          coefficients: np.ndarray) -> None:
    """
    Print a comprehensive analysis report.
    
    Parameters
    ----------
    db : pd.DataFrame
        The main database with qualia progressions.
    linkage_matrix : np.ndarray
        Hierarchical clustering linkage matrix.
    global_matrix : pd.DataFrame
        Global transition probability matrix.
    conditional_matrix : pd.DataFrame
        Conditional transition probability matrix.
    df_qualia : pd.DataFrame
        Summary of antecedent/consequent relationships.
    ranked_counts : pd.DataFrame
        Ranked transition counts.
    coefficients : np.ndarray
        Zipf regression coefficients.
    """
    print('\n' + '=' * 60)
    print('DATA REPORT')
    print('=' * 60)
    
    print('\n- Standard database:')
    print(db)
    
    print('\n- Plotting qualia dendrogram:')
    plot_qualia_dendrogram(linkage_matrix, conditional_matrix)
    print('(...dendrogram plotted...)')
    
    print('\n' + '=' * 60)
    print('1ST ORDER RESULTS')
    print('=' * 60)
    
    print('\n- Qualia matrix (global percentages):')
    print('  (Each cell shows the percentage of all transitions)')
    print(global_matrix)
    
    print('\n- Qualia matrix (conditional probabilities):')
    print('  (Each row shows P(consequent | antecedent))')
    print(conditional_matrix)
    
    print('\n- Non-zero qualia counts:')
    print('  (Shows unique antecedents and consequents for each qualia)')
    print(df_qualia)
    
    print('\n- Transition counts ranked:')
    print(ranked_counts)
    
    print('\n- Zipf regression analysis:')
    print(f'  Slope = {coefficients[0]:.4f}')
    print(f'  (A slope close to -1 indicates Zipf\'s law behaviour)')
    print(f'  Y-intercept = {coefficients[1]:.4f}')
