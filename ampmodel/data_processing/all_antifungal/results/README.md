# Antifungal Peptide Data Analysis Results

## Overview

This directory contains the analysis results of the antifungal peptide data processing pipeline, including sequence count statistics, database source distribution, and visualization charts.

## Generated Files

### Data Files (CSV)
| File | Description |
|------|-------------|
| `sequence_count_statistics.csv` | Sequence count at each processing step |
| `database_source_statistics.csv` | Database source distribution summary |
| `database_region_statistics.csv` | Detailed statistics for each database combination region |
| `sequence_database_detail.csv` | Full sequence list with database source information |

### Visualization Charts (PNG)
| File | Description |
|------|-------------|
| `database_upset_plot.png` | UpSet plot showing database combination distribution |
| `database_pie_chart.png` | Pie chart showing database contribution ratio |
| `database_stacked_bar.png` | Stacked bar chart showing unique vs shared sequences |
| `database_venn_schematic.png` | 5-way Venn diagram schematic with region counts |

### Documentation
| File | Description |
|------|-------------|
| `data_processing_flowchart.md` | Complete data processing pipeline flowchart |
| `README.md` | This file |

## Python Scripts

### `analyze_data.py`
Main analysis script that:
- Counts sequences in each processing step
- Analyzes database source distribution
- Generates CSV statistics files

**Usage:**
```bash
python analyze_data.py
```

### `plot_venn.py`
Visualization script that generates:
- UpSet plot
- Pie chart
- Stacked bar chart
- Venn diagram schematic

**Usage:**
```bash
python plot_venn.py
```

### `update_stats.py`
Updates sequence count statistics to include MMSeqs clustering results.

**Usage:**
```bash
python update_stats.py
```

## Data Processing Pipeline Summary

| Step | Description | Sequence Count |
|------|-------------|----------------|
| 1 | Original data (5 files) | 16,811 |
| 2 | After merge | 7,920 |
| 3 | After standardization | 6,921 |
| 4 | After 2+ files filter | 5,306 |
| 5 | After length filter (≤60 aa) | 4,758 |
| 6 | After MMSeqs clustering | 2,255 |

## Database Source Distribution

| Database | Unique Sequences |
|----------|-----------------|
| peptides_merged.csv | 6,005 |
| antifungal.csv | 5,696 |
| antifungal_peptides.csv | 1,813 |
| Antifungal_amps.csv | 1,811 |
| merged_data.csv | 559 |

## Re-run Analysis

To regenerate all analysis results:

1. Run sequence count analysis:
   ```bash
   python analyze_data.py
   ```

2. Update statistics with MMSeqs results:
   ```bash
   python update_stats.py
   ```

3. Generate visualization charts:
   ```bash
   python plot_venn.py
   ```

All output files will be saved to this directory, preserving intermediate data for future re-analysis.