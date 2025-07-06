# NeoCarto

## ECG Processing Script

The `ecg_CARTO.py` script processes ECG XML files from the `/carto` folder, extracting both metadata and waveform data.

### What it does:

- **Processes ECG XML files** from the carto directory using parallel processing
- **Extracts metadata** from XML files including:
  - Acquisition date and time
  - ECG interpretation text
  - Sinus rhythm detection (true/false)
- **Extracts waveform data** from all 12 ECG leads:
  - Resamples to 250 Hz sampling rate
  - Standardizes to 10-second duration
  - Pads shorter recordings with zeros
- **Saves outputs**:
  - `carto_ecg_metadata.csv` - Metadata with index references
  - `carto_ecg_waveforms.npy` - Waveform data (n_files × 2500 × 12)
  - `carto_ecg_sampling_rates.npy` - Original sampling rates

### Usage:

```bash
uv run python ecg_CARTO.py
```

The script uses parallel processing to handle large numbers of ECG files efficiently.
