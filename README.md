```markdown:README.md
# EMD and Fairness Analysis

This repository contains tools for analyzing fairness in machine learning models using Earth Mover's Distance (EMD) and statistical significance testing. The project includes two main components:

1. **EMD Calculator**: Computes the EMD between protected groups and overall distributions, with support for permutation tests to assess statistical significance.
2. **Fairness Analyzer**: Evaluates fairness metrics across multiple machine learning models, including traditional classifiers and deep learning models like ResNet50.

## Features

### EMD Calculator (`EMD_calculator.py`)
- Computes the minimal EMD between protected groups and overall distributions
- Supports categorical and numerical target variables
- Includes a permutation test for statistical significance
- Interactive command-line interface for dataset analysis

### Fairness Analyzer (`fairness_analyzer.py`)
- Evaluates fairness metrics across multiple machine learning models:
  - Support Vector Machine (SVM)
  - Random Forest (RF)
  - Decision Tree (DT)
  - Logistic Regression (LR)
  - K-Nearest Neighbors (KNN)
  - Naive Bayes (NB)
  - ResNet50 (Deep Learning)
- Supports both UCI and Kaggle datasets
- Computes fairness metrics including True Positive Rate (TPR), False Positive Rate (FPR), and Type II Error Ratio
- Includes cross-validation and statistical significance testing

## Usage

### EMD Calculator
```bash
python EMD_calculator.py
```

### Fairness Analyzer
```python
from fairness_analyzer import FairnessAnalyzer

# Initialize with selected algorithms
selected_algorithms = ['SVM', 'RF', 'DT', 'LR', 'KNN', 'NB', 'ResNet50']
analyzer = FairnessAnalyzer('fairness_results', algorithms=selected_algorithms)

# Process dataset list from Excel file
analyzer.process_dataset_list('/path/to/Datasets_with_EMD_and_p-value.xlsx')
```

## Requirements
- Python 3.7+
- Required packages:
  - pandas
  - numpy
  - scipy
  - scikit-learn
  - cuml (for GPU-accelerated models)
  - torch (for ResNet50)
  - ucimlrepo (for UCI datasets)
  - tqdm (for progress bars)

## File Structure
```
.
├── EMD_calculator.py          # EMD calculation and permutation testing
├── fairness_analyzer.py       # Fairness analysis across multiple models
├── README.md                  # This file
└── Datasets_with_EMD_and_p-value.xlsx  # Dataset list (example)
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
