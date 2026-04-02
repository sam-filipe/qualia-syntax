"""
Corpus Utilities Module for Qualia Syntax Analysis
===================================================

This module provides functionality for handling the Yale Classical Archives Corpus (YCAC)
and similar musical corpora. It includes utilities for:
    - User interaction and mode selection
    - Metadata file reading and processing
    - Composer validation and file management
    - Pitch class vector transformation from chord notation
    - Composition year estimation from date ranges

The module transforms raw corpus data into structured DataFrames suitable for
subsequent Fourier Qualia Space (FQS) analysis.

Written for Python 3.11+
"""

import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# CONSTANTS
# =============================================================================

# Mapping of note names (with accidentals) to pitch classes (0-11)
# This dictionary handles standard note names, sharps (#), double sharps (##),
# flats (-), and double flats (--) using enharmonic equivalence
NOTE_TO_PITCH_CLASS = {
    'C': 0, 'C#': 1, 'C##': 2, 'D-': 1, 'D--': 0,
    'D': 2, 'D#': 3, 'D##': 4, 'E-': 3, 'E--': 2,
    'E': 4, 'E#': 5, 'F-': 4,
    'F': 5, 'F#': 6, 'F##': 7, 'G-': 6, 'G--': 5,
    'G': 7, 'G#': 8, 'G##': 9, 'A-': 8, 'A--': 7,
    'A': 9, 'A#': 10, 'A##': 11, 'B-': 10, 'B--': 9,
    'B': 11, 'B#': 0, 'C-': 11
}

# Regular expression pattern to extract note names from chord strings
# Matches a letter A-G followed by 0-2 accidentals (# or -)
NOTE_PATTERN = re.compile(r'([A-G][#-]{0,2})')


# =============================================================================
# USER INTERACTION FUNCTIONS
# =============================================================================

def mode_definition() -> int:
    """
    Display mode options and prompt the user to select an analysis mode.
    
    Available modes:
        1 - Analyse specific composer(s)
        2 - Analyse entire corpus
    
    Returns
    -------
    int
        The selected mode (1 or 2).
    
    Raises
    ------
    SystemExit
        If the user provides invalid input.
    """
    print("\nMode options:\n 1 - composer(s)\n 2 - corpus")
    mode = input(" Please select a mode:")
    
    if mode.isdigit() and int(mode) in range(1, 3):
        return int(mode)
    else:
        print("\nInvalid input. Restart the program and please select a valid mode: 1 or 2.")
        quit()


def composer_definition() -> List[str]:
    """
    Prompt the user to enter composer names for analysis.
    
    The user should enter composer names separated by commas, with capitalised
    first letters (e.g., "Mozart, Bach, Satie, Messiaen").
    
    Returns
    -------
    List[str]
        A list of composer names stripped of leading/trailing whitespace.
    """
    print("\n\nPlease write the composers' names separated by commas with "
          "capitalized first letter:\n(...example: Mozart, Bach, Satie, Messiaen...)\n")
    names = input("Composers' names: ")
    composers = [name.strip() for name in names.split(',')]
    return composers


def col_width(width: int) -> None:
    """
    Set the maximum column width for pandas DataFrame display.
    
    Parameters
    ----------
    width : int
        The maximum number of characters to display in DataFrame columns.
    """
    pd.set_option('display.max_colwidth', width)


# =============================================================================
# METADATA AND FILE HANDLING FUNCTIONS
# =============================================================================

def read_metadata(metadata_filepath: str) -> pd.DataFrame:
    """
    Read and parse the corpus metadata CSV file.
    
    Parameters
    ----------
    metadata_filepath : str
        The system path to the metadata CSV file (typically '0_Metadata.csv').
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the columns: 'Title', 'Composer', 'Date',
        'Range', 'Filename', and 'Comments'.
    """
    metadata = pd.read_csv(
        metadata_filepath,
        usecols=['Title', 'Composer', 'Date', 'Range', 'Filename', 'Comments'],
        encoding='utf8',
        encoding_errors='replace'
    )
    # Remove quotation marks from filenames
    metadata['Filename'] = metadata['Filename'].str.replace('"', '')
    return metadata


def check_composers(metadata: pd.DataFrame, composers: List[str]) -> List[str]:
    """
    Validate that the requested composers exist in the corpus metadata.
    
    Parameters
    ----------
    metadata : pd.DataFrame
        The corpus metadata DataFrame containing a 'Composer' column.
    composers : List[str]
        The list of composer names to validate.
    
    Returns
    -------
    List[str]
        The filtered list containing only valid composer names.
    
    Raises
    ------
    SystemExit
        If none of the requested composers exist in the database.
    """
    # Identify composers not present in the metadata
    non_existing = [
        composer for composer in composers 
        if not metadata['Composer'].eq(composer).any()
    ]
    
    # Filter to keep only valid composers
    valid_composers = [c for c in composers if c not in non_existing]
    
    # Handle different validation outcomes
    if len(valid_composers) == 0:
        print("\nNone of the selected composers exist in the database. "
              "Please restart the program.\n")
        quit()
    elif len(non_existing) > 0:
        # Inform user about invalid composers
        if len(non_existing) == 1:
            invalid_names = non_existing[0]
        else:
            invalid_names = ", ".join(non_existing[:-1]) + " and " + non_existing[-1]
        print(f'\nSorry, but "{invalid_names}" is/are not valid or existing '
              f'composer(s) in the database.')
        print('Please check if the name is spelled correctly.\n')
    
    return valid_composers


def files_definition(corpus_directory: str, composers: List[str]) -> List[Optional[str]]:
    """
    Determine which corpus files to analyse based on the selected composers.
    
    The function searches the corpus directory for files matching each composer's
    name or initial letter.
    
    Parameters
    ----------
    corpus_directory : str
        The system path to the YCAC corpus directory.
    composers : List[str]
        The list of validated composer names.
    
    Returns
    -------
    List[Optional[str]]
        A list of filenames corresponding to each composer. May contain None
        if no matching file is found.
    """
    dir_files = os.listdir(corpus_directory)
    dir_files.remove('0_Metadata.csv')  # Exclude metadata file
    
    # Match each composer to a file (by full name prefix or initial letter)
    files = [
        next(
            (f for f in dir_files if f.startswith(c)),
            next(
                (f for f in dir_files if f.startswith(c[0]) and len(f) == 11),
                None
            )
        )
        for c in composers
    ]
    return files


# =============================================================================
# CHORD PROCESSING HELPER FUNCTIONS
# =============================================================================

def _parse_offset(offset_value: str) -> float:
    """
    Convert an offset value from string to float, handling fractional notation.
    
    Parameters
    ----------
    offset_value : str
        The offset value as a string, potentially containing fraction notation
        (e.g., '3/4').
    
    Returns
    -------
    float
        The numeric offset value.
    """
    if isinstance(offset_value, str) and '/' in offset_value:
        return eval(offset_value)
    return float(offset_value)


def _calculate_delta_values(df: pd.DataFrame, group_column: str = 'file') -> pd.DataFrame:
    """
    Calculate delta (time interval) values between consecutive chords.
    
    For each piece, this function computes the time difference between successive
    chord onsets. The final chord in each piece receives a delta value equal to
    twice the average delta for that piece.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'offset' and group_column columns.
    group_column : str, optional
        The column name used to group chords by piece (default: 'file').
    
    Returns
    -------
    pd.DataFrame
        The input DataFrame with an added 'delta' column.
    """
    df['delta'] = np.nan
    
    for filename, piece_group in df.groupby(group_column):
        piece_indices = piece_group.index
        offsets = piece_group['offset'].values
        
        if len(offsets) > 1:
            # Calculate time differences between consecutive chords
            deltas = np.diff(offsets)
            df.loc[piece_indices[:-1], 'delta'] = deltas
            # Final chord: use 2× average delta
            avg_delta = np.mean(deltas)
            df.loc[piece_indices[-1], 'delta'] = avg_delta * 2
        else:
            # Single chord piece: assign default value
            df.loc[piece_indices, 'delta'] = 2.0
    
    return df


def _clean_chord_string(chord_series: pd.Series) -> pd.Series:
    """
    Clean chord strings by removing music21 formatting and octave numbers.
    
    Parameters
    ----------
    chord_series : pd.Series
        A Series of chord strings in music21 format.
    
    Returns
    -------
    pd.Series
        Cleaned chord strings containing only note names with accidentals.
    """
    chord_series = chord_series.str.replace('<music21.chord.Chord ', '')
    chord_series = chord_series.str.replace('>', '')
    chord_series = chord_series.str.replace(r'\d', '', regex=True)
    return chord_series


def _chord_to_pitch_class_vector(chord_string: str) -> np.ndarray:
    """
    Convert a chord string to a 12-element pitch class vector.
    
    The vector contains counts of each pitch class (0-11) present in the chord.
    For example, 'C# C# A- E' becomes [0, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0].
    
    Parameters
    ----------
    chord_string : str
        A string containing note names separated by spaces.
    
    Returns
    -------
    np.ndarray
        A 12-element array where each index represents a pitch class (C=0, C#=1,
        ..., B=11) and each value is the count of that pitch class in the chord.
    """
    notes = NOTE_PATTERN.findall(chord_string)
    return np.array([
        sum(1 for note in notes if note in NOTE_TO_PITCH_CLASS 
            and NOTE_TO_PITCH_CLASS[note] == pc)
        for pc in range(12)
    ])


def _process_chord_dataframe(df: pd.DataFrame, composer_name: str) -> pd.DataFrame:
    """
    Process a raw chord DataFrame into pitch class vectors.
    
    This function performs the complete transformation pipeline:
    1. Parse offset values
    2. Calculate delta values
    3. Clean chord strings
    4. Convert to pitch class vectors
    5. Remove delta/offset columns
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with 'offset', 'Chord', 'file', and 'Composer' columns.
    composer_name : str
        The name of the composer being processed (for progress display).
    
    Returns
    -------
    pd.DataFrame
        Processed DataFrame with pitch class vectors in the 'Chord' column.
    """
    # Parse offset values (handle fractions like '3/4')
    df['offset'] = df['offset'].apply(_parse_offset)
    
    # Calculate time intervals between chords
    df = _calculate_delta_values(df)
    
    # Clean chord notation
    df['Chord'] = _clean_chord_string(df['Chord'])
    
    # Convert to pitch class vectors with progress bar
    print('')
    print(f' - {composer_name}: computing pitch lists:')
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#f7f0f5')
    df['Chord'] = df['Chord'].progress_apply(_chord_to_pitch_class_vector)
    
    # Remove intermediate columns
    df = df.drop(['delta', 'offset'], axis=1)
    
    return df


def _clean_filename(filename_series: pd.Series) -> pd.Series:
    """
    Clean filename strings by removing file extensions and special characters.
    
    Parameters
    ----------
    filename_series : pd.Series
        A Series of filename strings.
    
    Returns
    -------
    pd.Series
        Cleaned filename strings.
    """
    return filename_series.str.replace(
        r'\.mid|\.mxl|"|\\u00A0',
        lambda m: '_' if m.group(0) == '\u00A0' else '',
        regex=True
    )


# =============================================================================
# MAIN DATA LOADING FUNCTIONS
# =============================================================================

def dataLoop_m1(corpus_directory: str, composers: List[str], 
                metadata: pd.DataFrame, files: List[str]) -> pd.DataFrame:
    """
    Load and process corpus data for specific composers (Mode 1).
    
    This function iterates over the selected composer files, extracts chord data,
    transforms it into pitch class vectors, and combines it with metadata.
    
    Parameters
    ----------
    corpus_directory : str
        The system path to the YCAC corpus directory.
    composers : List[str]
        The list of validated composer names.
    metadata : pd.DataFrame
        The corpus metadata DataFrame.
    files : List[str]
        The list of filenames corresponding to each composer.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing:
        - 'Composer': Composer name
        - 'Title': Piece title/filename
        - 'Date': Composition date (if available)
        - 'Range': Composition date range
        - 'Chords': List of 12-element pitch class vectors for each chord
    """
    db = pd.DataFrame()
    
    for i in range(len(files)):
        file = files[i]
        filepath = corpus_directory + file
        
        # Load metadata for this composer
        meta = metadata[metadata['Composer'] == composers[i]]
        
        # Load chord data from CSV
        df = pd.read_csv(
            filepath,
            usecols=['offset', 'Chord', 'file', 'Composer'],
            dtype={'offset': str},
            encoding='utf-8-sig',
            encoding_errors='replace'
        )
        
        # Process chords into pitch class vectors
        df = _process_chord_dataframe(df, composers[i])
        
        # Filter to only the specified composer and remove anonymous entries
        df = df[df['Composer'] == composers[i]]
        df = df[df['Composer'] != 'Anonymous']
        
        # Clean up filename column
        df = df.rename(columns={'file': 'Filename'})
        df['Filename'] = _clean_filename(df['Filename'])
        
        # Aggregate chords by piece
        df = df.groupby('Filename').agg({
            'Chord': lambda x: x.tolist(),
            'Composer': 'first'
        }).reset_index()
        
        # Join with metadata
        df = pd.merge(df, meta[['Filename', 'Date', 'Range']], on='Filename', how='left')
        df = df.rename(columns={'Filename': 'Title', 'Chord': 'Chords'})
        df = df[['Composer', 'Title', 'Date', 'Range', 'Chords']]
        
        # Append to main database
        db = pd.concat([db, df], ignore_index=True)
        db = db.sort_values(by=['Composer'])
        db = db.reset_index(drop=True)
    
    return db


def dataLoop_m2(corpus_directory: str, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Load and process the entire corpus (Mode 2).
    
    This function iterates over all files in the corpus directory, extracts
    chord data, transforms it into pitch class vectors, and combines it
    with metadata.
    
    Parameters
    ----------
    corpus_directory : str
        The system path to the YCAC corpus directory.
    metadata : pd.DataFrame
        The corpus metadata DataFrame.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing:
        - 'Composer': Composer name
        - 'Title': Piece title/filename
        - 'Date': Composition date (if available)
        - 'Range': Composition date range
        - 'Chords': List of 12-element pitch class vectors for each chord
    """
    print('\n\n(...proceeding with processing; this task may take a few minutes...)\n\n')
    
    db = pd.DataFrame()
    
    # Get list of all corpus files (excluding metadata)
    dir_files = os.listdir(corpus_directory)
    dir_files.remove('0_Metadata.csv')
    
    for file in dir_files:
        filepath = corpus_directory + file
        
        # Load chord data from CSV
        df = pd.read_csv(
            filepath,
            usecols=['offset', 'Chord', 'file', 'Composer'],
            dtype={'offset': str},
            encoding='utf-8-sig',
            encoding_errors='replace'
        )
        
        # Process chords into pitch class vectors
        df = _process_chord_dataframe(df, str(file))
        
        # Remove anonymous composers
        df = df[df['Composer'] != 'Anonymous']
        
        # Clean up filename column
        df = df.rename(columns={'file': 'Filename'})
        df['Filename'] = _clean_filename(df['Filename'])
        
        # Aggregate chords by piece
        df = df.groupby('Filename').agg({
            'Chord': lambda x: x.tolist(),
            'Composer': 'first'
        }).reset_index()
        
        # Join with metadata
        df = pd.merge(df, metadata[['Filename', 'Date', 'Range']], on='Filename', how='left')
        df = df.rename(columns={'Filename': 'Title', 'Chord': 'Chords'})
        df = df[['Composer', 'Title', 'Date', 'Range', 'Chords']]
        
        # Append to main database
        db = pd.concat([db, df], ignore_index=True)
        db = db.sort_values(by=['Composer'])
        db = db.reset_index(drop=True)
    
    return db


def year_definition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate composition years for pieces with undefined dates.
    
    For pieces without an exact composition date, this function extracts the
    latest year from the date range (e.g., '1910-1920' yields 1920).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'Date' and 'Range' columns.
    
    Returns
    -------
    pd.DataFrame
        The input DataFrame with an added 'Year' column containing either:
        - The exact date (if available)
        - The end year of the range (if exact date unavailable)
        - 'Undetermined' (if neither is available)
    """
    print(' - determining undefined composition years:')
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#f7f0f5')
    
    # Extract the end year from the range (e.g., '1910-1920' -> 1920)
    df['range_mean'] = df['Range'].progress_apply(
        lambda x: pd.to_numeric(x.split('-'))[1] 
        if isinstance(x, str) and '-' in x else None
    )
    
    # Fill missing dates with range-derived values
    df['year_filled'] = df['Date'].fillna(df['range_mean'])
    df['year_filled'] = pd.to_numeric(df['year_filled'], errors='coerce')
    df['year_filled'] = df['year_filled'].fillna(0)
    
    # Convert to integer and mark undetermined years
    df['Year'] = df['year_filled'].astype(int)
    df['Year'] = df['Year'].apply(lambda x: 'Undetermined' if x == 0 else x)
    
    # Clean up intermediate columns
    df.drop(['range_mean', 'year_filled'], axis=1, inplace=True)
    
    return df
