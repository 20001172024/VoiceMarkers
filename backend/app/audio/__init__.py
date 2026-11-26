"""Audio loading, validation and Parselmouth (Praat) feature extraction.

Pipeline: raw bytes -> `load_audio` -> `validate_audio` -> `extract_features`.
Audio is only ever held in memory; nothing here writes to disk or logs samples.
"""
