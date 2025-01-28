import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from itertools import permutations
from ucimlrepo import fetch_ucirepo
from tqdm import tqdm  # For progress bar in the permutation test

def find_best_ordering(values1, values2):
    """
    Finds the ordering (permutation) of categories that minimizes
    the Wasserstein distance (EMD) between two discrete distributions.

    Parameters
    ----------
    values1 : list or np.array
        Probability distribution for the protected group.
    values2 : list or np.array
        Probability distribution for the overall group.

    Returns
    -------
    min_emd : float
        The minimum EMD found.
    best_order : tuple
        The permutation of indices that yields the minimum EMD.
    best_values : tuple
        (best_values1, best_values2) after applying 'best_order'.
    """
    n = len(values1)
    base_indices = list(range(n))
    min_emd = float('inf')
    best_order = None
    best_values = (None, None)
    
    for perm in permutations(base_indices):
        reordered_values1 = [values1[i] for i in perm]
        reordered_values2 = [values2[i] for i in perm]
        
        emd = wasserstein_distance(base_indices, base_indices, 
                                   reordered_values1, reordered_values2)
        if emd < min_emd:
            min_emd = emd
            best_order = perm
            best_values = (reordered_values1, reordered_values2)
    
    return min_emd, best_order, best_values

def analyze_distribution(df, protected_col, protected_val, target_col, categories_map=None):
    """
    Computes the minimal EMD between a specified protected attribute value
    and the overall target distribution.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe.
    protected_col : str
        Protected attribute column name (e.g., 'Gender').
    protected_val : str or int
        A specific value of the protected attribute (e.g., 'Male').
    target_col : str
        The name of the target attribute column.
    categories_map : dict, optional
        If the target column is categorical, this dictionary maps each category to an integer.

    Returns
    -------
    min_emd : float
        The minimum EMD.
    best_order : tuple
        The permutation of indices that yields the minimum EMD.
    best_values1 : list
        Reordered distribution for the protected group.
    best_values2 : list
        Reordered distribution for the overall group.
    categories : list
        The sorted list of categories (or numeric values).
    """
    df_analyzed = df.copy()
    
    # If target_col is categorical, apply mapping to integers
    if categories_map is not None:
        mapped_col = target_col + "_num"
        df_analyzed[mapped_col] = df_analyzed[target_col].map(categories_map)
        categories = sorted(categories_map.keys(), key=lambda x: categories_map[x])
    else:
        mapped_col = target_col
        categories = sorted(df_analyzed[mapped_col].unique())

    # Compute protected_val distribution vs. overall distribution
    protected_dist_series = (df_analyzed[df_analyzed[protected_col] == protected_val][mapped_col]
                             .value_counts(normalize=True)
                             .sort_index())
    overall_dist_series = (df_analyzed[mapped_col]
                           .value_counts(normalize=True)
                           .sort_index())

    # Align indices, fill missing as 0
    protected_dist_series = protected_dist_series.reindex(range(len(categories))).fillna(0)
    overall_dist_series = overall_dist_series.reindex(range(len(categories))).fillna(0)

    # Find best ordering that minimizes EMD
    min_emd, best_order, (best_values1, best_values2) = find_best_ordering(
        protected_dist_series.values, overall_dist_series.values
    )
    
    return min_emd, best_order, best_values1, best_values2, categories

def compute_emd(df, protected_col, protected_val, target_col, categories_map=None):
    """
    A helper function to compute the minimal EMD, primarily used for permutations
    during the randomization test.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe.
    protected_col : str
        Protected attribute column.
    protected_val : str or int
        A specific value of the protected attribute.
    target_col : str
        The target column name.
    categories_map : dict, optional
        Mapping of category -> integer if the target column is categorical.

    Returns
    -------
    min_emd : float
        The minimal EMD (best ordering is computed internally but not returned).
    """
    df_temp = df.copy()
    
    # Map categorical target to integer if needed
    if categories_map is not None:
        mapped_col = target_col + "_num"
        df_temp[mapped_col] = df_temp[target_col].map(categories_map)
        categories = sorted(categories_map.keys(), key=lambda x: categories_map[x])
    else:
        mapped_col = target_col
        categories = sorted(df_temp[mapped_col].unique())

    # Probability distributions
    protected_dist_series = (df_temp[df_temp[protected_col] == protected_val][mapped_col]
                             .value_counts(normalize=True)
                             .sort_index())
    overall_dist_series = (df_temp[mapped_col]
                           .value_counts(normalize=True)
                           .sort_index())

    protected_dist_series = protected_dist_series.reindex(range(len(categories))).fillna(0)
    overall_dist_series = overall_dist_series.reindex(range(len(categories))).fillna(0)

    # Compute minimal EMD by finding best ordering
    min_emd, _, _ = find_best_ordering(protected_dist_series.values,
                                       overall_dist_series.values)
    return min_emd

def permutation_test_emd(df, protected_col, protected_val, target_col,
                         categories_map=None, n_permutations=1000, random_seed=42):
    """
    Performs a permutation test to assess the statistical significance of the EMD.

    1. Compute the observed EMD with the original data.
    2. Shuffle the protected attribute (or the target) multiple times,
       recompute EMD each time to build a null distribution.
    3. Estimate the p-value based on how many permuted EMDs are >= the observed EMD.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe.
    protected_col : str
        Protected attribute column.
    protected_val : str or int
        The specific value of the protected attribute (e.g., 'Male').
    target_col : str
        Target column name.
    categories_map : dict, optional
        Mapping from category -> integer if the target column is categorical.
    n_permutations : int
        Number of permutations (larger = more accurate, more time).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    p_value : float
        Approximate p-value for a one-sided test (observed EMD being unusually large).
    emd_observed : float
        The EMD computed on the original, unshuffled data.
    perm_emd_values : list
        EMD values from the shuffled data permutations.
    """
    # 1. Compute observed EMD
    emd_observed = compute_emd(df, protected_col, protected_val, target_col, categories_map)
    
    # 2. Build the null distribution of EMDs via shuffling
    perm_emd_values = []
    rng = np.random.default_rng(seed=random_seed)
    
    # Using tqdm to show a progress bar
    # from tqdm import tqdm
    for _ in tqdm(range(n_permutations), desc="Permutation test in progress"):
        shuffled_df = df.copy()
        # Shuffle the protected attribute column
        original_values = shuffled_df[protected_col].values
        rng.shuffle(original_values)  # In-place shuffle
        shuffled_df[protected_col] = original_values
        
        # Recompute the EMD on the shuffled data
        emd_perm = compute_emd(shuffled_df, protected_col, protected_val, target_col, categories_map)
        perm_emd_values.append(emd_perm)
    
    # 3. Compute a one-sided p-value
    count_greater_or_equal = sum(1 for e in perm_emd_values if e >= emd_observed)
    p_value = count_greater_or_equal / n_permutations
    
    return p_value, emd_observed, perm_emd_values

def main():
    # ------------------------------------------------------------------
    # 1. Ask user whether it's a UCI dataset or a CSV file
    # ------------------------------------------------------------------
    is_uci = input("Is this a UCI dataset? (yes/no): ").strip().lower()
    while is_uci not in ['yes', 'no']:
        is_uci = input("Invalid input. Please enter 'yes' or 'no': ").strip().lower()

    if is_uci == 'yes':
        uci_id = input("Please enter the UCI dataset ID: ").strip()
        dataset = fetch_ucirepo(id=int(uci_id))
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    else:
        csv_path = input("Please enter the CSV file path or URL: ").strip()
        df = pd.read_csv(csv_path)

    # ------------------------------------------------------------------
    # 2. Identify the protected attribute (attempt "Gender"/"Sex")
    # ------------------------------------------------------------------
    protected_candidates = ["Gender", "gender", "Sex", "sex"]
    protected_col = None
    for col_name in df.columns:
        if col_name.lower() in [c.lower() for c in protected_candidates]:
            protected_col = col_name
            break

    if protected_col is None:
        print("No common gender/sex column found automatically.")
        print("Available columns:", df.columns.tolist())
        protected_col = input("Please specify the protected attribute column name: ").strip()
        while protected_col not in df.columns:
            print("That column does not exist. Try again.")
            protected_col = input("Please specify the protected attribute column name: ").strip()
    else:
        print(f"Detected protected attribute: {protected_col}")

    # ------------------------------------------------------------------
    # 3. Identify the target column (default is the last column)
    # ------------------------------------------------------------------
    default_target_col = df.columns[-1]
    print(f"The default target column is: {default_target_col}")
    user_confirm = input("Use this as the target? (yes/no): ").strip().lower()
    if user_confirm == 'no':
        print("Available columns:", df.columns.tolist())
        target_col = input("Please specify the target column name: ").strip()
        while target_col not in df.columns:
            print("That column does not exist. Try again.")
            target_col = input("Please specify the target column name: ").strip()
    else:
        target_col = default_target_col

    # ------------------------------------------------------------------
    # 4. If target column is categorical, build a category->int map
    # ------------------------------------------------------------------
    if df[target_col].dtype == 'object' or str(df[target_col].dtype).startswith('category'):
        categories = df[target_col].unique().tolist()
        categories_map = {cat: i for i, cat in enumerate(categories)}
    else:
        categories_map = None

    # ------------------------------------------------------------------
    # 5. For each unique protected value, compute EMD & (optionally) do a permutation test
    # ------------------------------------------------------------------
    unique_protected_values = df[protected_col].unique().tolist()
    print("\n========================================================")
    print("Calculating EMD for each protected attribute value vs. overall distribution...\n")

    do_perm_test = input("Do you want to run a permutation test for significance? (yes/no): ").strip().lower()
    if do_perm_test not in ['yes', 'no']:
        do_perm_test = 'no'

    for val in unique_protected_values:
        # Basic EMD analysis
        min_emd, best_order, best_values1, best_values2, categories = analyze_distribution(
            df, protected_col, val, target_col, categories_map
        )

        print(f"\nProtectedCol={protected_col}, Value={val} vs Overall:")
        print(f"Minimum EMD: {min_emd:.6f}")
        print("Best ordering (original index -> permuted index):", best_order)

        print("\nBest ordering category mapping:")
        for i, new_pos in enumerate(best_order):
            print(f"Index {i} (category {categories[i]}) -> "
                  f"Index {new_pos} (category {categories[new_pos]})")

        print("\nDistribution under the best ordering:")
        for i, (p_val, o_val) in enumerate(zip(best_values1, best_values2)):
            cat_name = categories[best_order[i]]
            print(f"Category '{cat_name}': {protected_col}={val} prob = {p_val:.6f}, Overall prob = {o_val:.6f}")

        # Verification EMD
        verify_emd = wasserstein_distance(
            range(len(best_values1)),
            range(len(best_values2)),
            best_values1,
            best_values2
        )
        print(f"Verification EMD: {verify_emd:.6f}")

        # Permutation test for significance (if desired)
        if do_perm_test == 'yes':
            p_value, emd_observed, perm_emd_values = permutation_test_emd(
                df, protected_col, val, target_col,
                categories_map=categories_map,
                n_permutations=1000,  # Adjust as needed
                random_seed=42
            )
            print("\n--- Permutation Test Results ---")
            print(f"Observed EMD: {emd_observed:.10f}")
            print(f"One-sided p-value: {p_value:.10f}")
            if p_value < 0.05:
                print("=> Statistically significant at the 5% level.")
            else:
                print("=> Not statistically significant at the 5% level.")
            # print(perm_emd_values)

        print("\n" + "="*80)

if __name__ == "__main__":
    main()