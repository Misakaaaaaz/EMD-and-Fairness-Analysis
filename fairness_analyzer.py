import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from cuml.svm import SVC
from cuml.ensemble import RandomForestClassifier
from cuml.neighbors import KNeighborsClassifier
from cuml.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.stats import ttest_ind
from ucimlrepo import fetch_ucirepo
import os
import glob
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

class ResNet50Classifier:
    def __init__(self, input_dim, num_classes, random_state=42):
        """
        Initialize the ResNet50 classifier.

        Parameters
        ----------
        input_dim : int
            The input dimension of the data.
        num_classes : int
            The number of output classes for classification.
        random_state : int, optional
            Random seed for reproducibility (default is 42).
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet50(pretrained=True)
        self.num_classes = num_classes
        print(f"Initializing ResNet50 with {num_classes} classes")
        
        # Modify the first layer to accept the correct input dimensions
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the final layer for classification
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.epochs = 10
        self.batch_size = 2048
        torch.manual_seed(random_state)
        
    def reshape_data(self, X):
        """
        Reshape the input data to match ResNet's expected input shape.

        Parameters
        ----------
        X : np.ndarray
            The input data to reshape.

        Returns
        -------
        np.ndarray
            The reshaped input data.
        """
        side_length = int(np.ceil(np.sqrt(X.shape[1])))
        pad_size = side_length * side_length - X.shape[1]
        if pad_size > 0:
            X = np.pad(X, ((0, 0), (0, pad_size)))
        return X.reshape(-1, 1, side_length, side_length)
        
    def fit(self, X, y):
        """
        Train the ResNet50 model on the provided data.

        Parameters
        ----------
        X : np.ndarray
            The input features for training.
        y : np.ndarray
            The target labels for training.
        """
        # Verify the target labels
        unique_labels = np.unique(y)
        if not all(label >= 0 and label < self.num_classes for label in unique_labels):
            raise ValueError(f"Invalid target labels found. Labels should be in range [0, {self.num_classes-1}], "
                             f"but got {sorted(unique_labels)}")
            
        X = self.reshape_data(X)
        X = torch.FloatTensor(X).to(self.device)
        y = torch.LongTensor(y).to(self.device)
        
        # Verify the data
        print(f"Training data shape: {X.shape}")
        print(f"Target labels range: [{y.min().item()}, {y.max().item()}]")
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        # Add progress bar for epochs
        for epoch in tqdm(range(self.epochs), desc='Training ResNet50', unit='epoch'):
            # Add progress bar for batches within each epoch
            batch_progress = tqdm(dataloader, 
                                  desc=f'Epoch {epoch+1}/{self.epochs}',
                                  leave=False,
                                  unit='batch')
            for batch_X, batch_y in batch_progress:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                # Update batch progress bar with loss
                batch_progress.set_postfix({'loss': f'{loss.item():.4f}'})
            
            batch_progress.close()
                
    def predict(self, X):
        """
        Make predictions using the trained ResNet50 model.

        Parameters
        ----------
        X : np.ndarray
            The input features for prediction.

        Returns
        -------
        np.ndarray
            The predicted labels.
        """
        X = self.reshape_data(X)
        X = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        
        # Add progress bar for prediction
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

class FairnessAnalyzer:
    def __init__(self, output_dir='fairness_results', algorithms=None):
        """
        Initialize the FairnessAnalyzer.

        Parameters
        ----------
        output_dir : str, optional
            The directory to save results (default is 'fairness_results').
        algorithms : list, optional
            List of algorithms to use (default is all available algorithms).
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Default models dictionary
        self.all_models = {
            'SVM': lambda: SVC(random_state=42),
            'RF': lambda: RandomForestClassifier(random_state=42),
            'DT': lambda: DecisionTreeClassifier(random_state=42),
            'LR': lambda: LogisticRegression(random_state=42),
            'KNN': lambda: KNeighborsClassifier(),
            'NB': lambda: GaussianNB(),
            'ResNet50': lambda num_classes: ResNet50Classifier(input_dim=None, num_classes=num_classes, random_state=42)
        }
        
        # Use specified algorithms or all available ones
        self.selected_algorithms = algorithms if algorithms else list(self.all_models.keys())
        
        # Check existing results and raise an error if the results file is not found
        self.results_file = os.path.join(output_dir, "final_results.csv")
        if not os.path.exists(self.results_file):
            raise Exception("final_results.csv not found!")
            
        # Read existing results to get dataset IDs
        self.existing_results = pd.read_csv(self.results_file)
        self.existing_dataset_ids = self.existing_results['Dataset_ID'].unique()
        
        # Check for algorithm overwrites
        existing_algorithms = set(self.existing_results.columns) - {'Dataset_ID'}
        for algo in self.selected_algorithms:
            if algo in existing_algorithms:
                response = input(f"Algorithm {algo} already exists in results. Overwrite? (y/n): ")
                if response.lower() != 'y':
                    self.selected_algorithms.remove(algo)
        
        # Initialize models based on selected algorithms
        self.models = {}
        
        self.gender_terms = {
            'sex', 'gender', 'male', 'female', 
            'Sex', 'Gender', 'Male', 'Female',
            'SEX', 'GENDER', 'MALE', 'FEMALE'
        }
        
        self.gender_mapping = {
            # Traditional 0/1 encoding
            'm': 1, 'male': 1, 'man': 1, '1': 1, 'Male': 1, 'MALE': 1, 1: 1,
            'f': 0, 'female': 0, 'woman': 0, '0': 0, 'Female': 0, 'FEMALE': 0, 0: 0,
            # 1/2 encoding
            '2': 0, 2: 0,  # 2 typically represents female
        }

        # Define tree-based and non-tree-based algorithms
        self.tree_based = {'DT', 'RF'}
        self.non_tree_based = set(self.all_models.keys()) - self.tree_based
        
        # New columns to add
        self.new_columns = ['k-value', 'tree-based-def1', 'non-tree-based-def1', 
                            'tree-based-def2', 'non-tree-based-def2']
        
        # Check if new columns exist and ask for confirmation to overwrite
        existing_df = pd.read_csv(self.results_file)
        for col in self.new_columns:
            if col in existing_df.columns:
                response = input(f"Column {col} already exists in results. Overwrite? (y/n): ")
                if response.lower() != 'y':
                    self.new_columns.remove(col)
        
        # Add new columns if they don't exist
        for col in self.new_columns:
            if col not in existing_df.columns:
                existing_df[col] = None
        existing_df.to_csv(self.results_file, index=False)

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate fairness metrics for multi-class problems.
        Uses a one-vs-rest strategy to compute metrics for each class.

        Parameters
        ----------
        y_true : np.ndarray
            The true labels.
        y_pred : np.ndarray
            The predicted labels.

        Returns
        -------
        tuple
            A tuple containing TPR, FPR, ratio, TP, TN, FP, FN.
        """
        unique_classes = np.unique(y_true)
        n_classes = len(unique_classes)
        
        if n_classes <= 2:  # Binary classification case
            cm = confusion_matrix(y_true, y_pred)
            TP = cm[1, 1]
            TN = cm[0, 0]
            FP = cm[0, 1]
            FN = cm[1, 0]
            
            TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
            FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
            ratio = FN / FP if FP > 0 else 0
            
            return TPR, FPR, ratio, TP, TN, FP, FN
        
        else:  # Multi-class case
            # Initialize accumulators for metrics
            macro_TPR = 0
            macro_FPR = 0
            macro_ratio = 0
            total_TP = 0
            total_TN = 0
            total_FP = 0
            total_FN = 0
            
            # Calculate metrics for each class
            for class_label in unique_classes:
                # Treat the current class as positive and others as negative
                y_true_binary = (y_true == class_label).astype(int)
                y_pred_binary = (y_pred == class_label).astype(int)
                
                cm = confusion_matrix(y_true_binary, y_pred_binary)
                TP = cm[1, 1]
                TN = cm[0, 0]
                FP = cm[0, 1]
                FN = cm[1, 0]
                
                # Accumulate basic metrics
                total_TP += TP
                total_TN += TN
                total_FP += FP
                total_FN += FN
                
                # Calculate current class's ratio metrics
                class_TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
                class_FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
                class_ratio = FN / FP if FP > 0 else 0
                
                # Accumulate ratio metrics
                macro_TPR += class_TPR
                macro_FPR += class_FPR
                macro_ratio += class_ratio
            
            # Calculate macro averages
            macro_TPR /= n_classes
            macro_FPR /= n_classes
            macro_ratio /= n_classes
            
            print(f"\nMulti-class metrics:")
            print(f"Number of classes: {n_classes}")
            print(f"Per-class metrics:")
            for class_label in unique_classes:
                y_true_binary = (y_true == class_label).astype(int)
                y_pred_binary = (y_pred == class_label).astype(int)
                cm = confusion_matrix(y_true_binary, y_pred_binary)
                print(f"\nClass {class_label}:")
                print(f"Confusion Matrix:\n{cm}")
            
            return macro_TPR, macro_FPR, macro_ratio, total_TP, total_TN, total_FP, total_FN

    def detect_protected_attribute(self, df):
        """
        Detect potential protected attributes in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to search for protected attributes.

        Returns
        -------
        str or None
            The name of the detected protected attribute column, or None if not found.
        """
        for col in df.columns:
            if any(term in str(col) for term in self.gender_terms):
                unique_values = set(str(x).lower() for x in df[col].unique())
                print(f"\nFound potential protected attribute: {col}")
                print(f"Unique values: {df[col].unique()}")
                if any(str(val) in self.gender_mapping or val in self.gender_mapping for val in df[col].unique()):
                    return col
        return None

    def map_gender_values(self, series):
        """
        Map gender values in the series to numerical representations.

        Parameters
        ----------
        series : pd.Series
            The series containing gender values to map.

        Returns
        -------
        pd.Series
            The series with mapped numerical values.
        """
        mapped_values = series.copy()
        print(f"\nOriginal gender value distribution:\n{series.value_counts()}")
        
        for idx, val in series.items():
            # Handle both numeric and string types
            if val in self.gender_mapping:
                mapped_values[idx] = self.gender_mapping[val]
            else:
                val_str = str(val).lower().strip()
                if val_str in self.gender_mapping:
                    mapped_values[idx] = self.gender_mapping[val_str]
        
        print(f"\nMapped gender value distribution:\n{mapped_values.value_counts()}")
        return mapped_values

    def evaluate_fairness(self, X_protected, X_unprotected, y_protected, y_unprotected, dataset_id):
        """
        Evaluate fairness metrics between protected and unprotected groups.

        Parameters
        ----------
        X_protected : np.ndarray
            The feature data for the protected group.
        X_unprotected : np.ndarray
            The feature data for the unprotected group.
        y_protected : np.ndarray
            The target labels for the protected group.
        y_unprotected : np.ndarray
            The target labels for the unprotected group.
        dataset_id : str
            The identifier for the dataset being processed.

        Returns
        -------
        dict
            A dictionary containing the fairness evaluation results for each model.
        """
        print(f"\nDataset {dataset_id} sizes:")
        print(f"Protected group: {len(X_protected)} samples")
        print(f"Unprotected group: {len(X_unprotected)} samples")
        
        # Return nan results if either group has no samples
        if len(X_protected) == 0 or len(X_unprotected) == 0:
            print(f"Warning: Empty group detected in dataset {dataset_id}")
            return {name: "(nan, nan) nan" for name in self.models.keys()}
        
        # Determine the number of folds based on the minimum group size
        min_samples = min(len(X_protected), len(X_unprotected))
        if min_samples <= 500:
            k = min(5, min_samples)
        elif min_samples <= 1000:
            k = min(10, min_samples)
        else:
            k = min(20, min_samples)
        
        print(f"Using {k} folds based on minimum group size of {min_samples}")
        
        # Ensure k is valid
        if k < 2:
            print(f"Warning: Not enough samples for cross-validation in dataset {dataset_id}")
            return {name: "(nan, nan) nan" for name in self.models.keys()}
        
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        final_results = {}
        
        # Create DataFrame for intermediate results
        intermediate_columns = ['Dataset_ID', 'Model', 'Fold', 'Group',
                                'TPR', 'FPR', 'TE_ratio', 'TP', 'TN', 'FP', 'FN']
        intermediate_results = []

        for name, model in self.models.items():
            print(f"\nProcessing {dataset_id} with {name}")
            protected_metrics = []
            unprotected_metrics = []

            # Generate fold indices
            protected_folds = list(kf.split(X_protected))
            unprotected_folds = list(kf.split(X_unprotected))

            for fold, ((train_idx_p, test_idx_p), (train_idx_u, test_idx_u)) in enumerate(zip(protected_folds, unprotected_folds), 1):
                print(f"Processing {dataset_id} - {name} - Fold {fold}/{k}")
                
                try:
                    # Protected group
                    X_train_p = X_protected[train_idx_p]
                    X_test_p = X_protected[test_idx_p]
                    y_train_p = y_protected[train_idx_p]
                    y_test_p = y_protected[test_idx_p]
                    
                    model.fit(X_train_p, y_train_p)
                    y_pred_p = model.predict(X_test_p)
                    TPR_p, FPR_p, ratio_p, TP_p, TN_p, FP_p, FN_p = self.calculate_metrics(y_test_p, y_pred_p)
                    protected_metrics.append([TPR_p, FPR_p, ratio_p])
                    
                    # Save intermediate results for protected group
                    intermediate_results.append({
                        'Dataset_ID': dataset_id,
                        'Model': name,
                        'Fold': fold,
                        'Group': 'Protected',
                        'TPR': TPR_p,
                        'FPR': FPR_p,
                        'TE_ratio': ratio_p,
                        'TP': TP_p,
                        'TN': TN_p,
                        'FP': FP_p,
                        'FN': FN_p
                    })

                    # Unprotected group
                    if len(X_unprotected) > 0:
                        X_train_u = X_unprotected[train_idx_u]
                        X_test_u = X_unprotected[test_idx_u]
                        y_train_u = y_unprotected[train_idx_u]
                        y_test_u = y_unprotected[test_idx_u]
                        
                        model.fit(X_train_u, y_train_u)
                        y_pred_u = model.predict(X_test_u)
                        TPR_u, FPR_u, ratio_u, TP_u, TN_u, FP_u, FN_u = self.calculate_metrics(y_test_u, y_pred_u)
                        unprotected_metrics.append([TPR_u, FPR_u, ratio_u])
                        
                        # Save intermediate results for unprotected group
                        intermediate_results.append({
                            'Dataset_ID': dataset_id,
                            'Model': name,
                            'Fold': fold,
                            'Group': 'Unprotected',
                            'TPR': TPR_u,
                            'FPR': FPR_u,
                            'TE_ratio': ratio_u,
                            'TP': TP_u,
                            'TN': TN_u,
                            'FP': FP_u,
                            'FN': FN_u
                        })

                except Exception as e:
                    print(f"Error in fold {fold} for {dataset_id} with {name}: {str(e)}")
                    continue

            if len(protected_metrics) > 0 and len(unprotected_metrics) > 0:
                protected_metrics = np.array(protected_metrics)
                unprotected_metrics = np.array(unprotected_metrics)

                try:
                    tpr_pvalue = ttest_ind(protected_metrics[:, 0], unprotected_metrics[:, 0]).pvalue
                    fpr_pvalue = ttest_ind(protected_metrics[:, 1], unprotected_metrics[:, 1]).pvalue
                    te_pvalue = ttest_ind(protected_metrics[:, 2], unprotected_metrics[:, 2]).pvalue
                    
                    # Format the results string with p-values
                    final_results[name] = f"({tpr_pvalue:.4f}, {fpr_pvalue:.4f}) {te_pvalue:.4f}"
                    
                except Exception as e:
                    print(f"Error calculating p-values for {dataset_id} with {name}: {str(e)}")
                    final_results[name] = "(nan, nan) nan"
            else:
                final_results[name] = "(nan, nan) nan"

            # Save intermediate results after each model
            intermediate_df = pd.DataFrame(intermediate_results)
            intermediate_df.to_csv(os.path.join(self.output_dir, f"{dataset_id}_{name}_intermediate_results.csv"), index=False)
            intermediate_results = []  # Clear intermediate results
        
        return final_results

    def process_kaggle_dataset(self, dataset_id, url, target_attr):
        """
        Process a Kaggle dataset given its ID and URL.

        Parameters
        ----------
        dataset_id : str
            The ID of the Kaggle dataset.
        url : str
            The URL to fetch the dataset.
        target_attr : str
            The name of the target attribute column.

        Returns
        -------
        bool
            True if processing was successful, False otherwise.
        """
        try:
            base_directory = "kaggle_datasets"
            dataset_path = "/".join(url.split("/")[-2:])
            dataset_directory = os.path.join(base_directory, dataset_path.replace("/", "_"))
            
            csv_files = glob.glob(os.path.join(dataset_directory, "*.csv"))
            if not csv_files:
                print(f"No CSV file found in {dataset_directory}")
                manual_path = input("Please enter the path to your dataset CSV file: ")
                if not os.path.exists(manual_path):
                    raise Exception(f"File not found: {manual_path}")
                df = pd.read_csv(manual_path)
            else:
                df = pd.read_csv(csv_files[0])
            
            # Remove rows with missing values
            df = df.dropna()
            
            # Display column names and first 5 rows
            print("\nAvailable columns:")
            for i, col in enumerate(df.columns):
                print(f"{i}: {col}")
            print("\nFirst 5 rows of the dataset:")
            print(df.head())
            
            protected_attr = self.detect_protected_attribute(df)
            if protected_attr is None:
                print("\nNo protected attribute automatically detected.")
                protected_attr = input("Please enter the name of the protected attribute column: ")
                if protected_attr not in df.columns:
                    raise Exception(f"Column {protected_attr} not found in dataset")
                
            if target_attr not in df.columns:
                print(f"\nTarget attribute '{target_attr}' not found in dataset.")
                target_attr = input("Please enter the name of the target attribute column: ")
                if target_attr not in df.columns:
                    raise Exception(f"Column {target_attr} not found in dataset")
                
            print(f"Using protected attribute: {protected_attr}")
            print(f"Using target attribute: {target_attr}")
            return self.process_dataset(df, dataset_id, protected_attr, target_attr)
            
        except Exception as e:
            print(f"Error processing Kaggle dataset {dataset_id}: {str(e)}")
            return None

    def process_uci_dataset(self, dataset_id, url, target_attr):
        """
        Process a UCI dataset given its ID and URL.

        Parameters
        ----------
        dataset_id : str
            The ID of the UCI dataset.
        url : str
            The URL to fetch the dataset.
        target_attr : str
            The name of the target attribute column.

        Returns
        -------
        bool
            True if processing was successful, False otherwise.
        """
        try:
            dataset_id_num = int(url.split('/')[-2])
            try:
                dataset = fetch_ucirepo(id=dataset_id_num)
                df = dataset.data.features
                df[target_attr] = dataset.data.targets
            except:
                print(f"Could not fetch UCI dataset with ID {dataset_id_num}")
                manual_path = input("Please enter the path to your dataset CSV file: ")
                if not os.path.exists(manual_path):
                    raise Exception(f"File not found: {manual_path}")
                df = pd.read_csv(manual_path)
            
            # Remove rows with missing values
            df = df.dropna()
            
            # Display column names and first 5 rows
            print("\nAvailable columns:")
            for i, col in enumerate(df.columns):
                print(f"{i}: {col}")
            print("\nFirst 5 rows of the dataset:")
            print(df.head())
            
            protected_attr = self.detect_protected_attribute(df)
            if protected_attr is None:
                print("\nNo protected attribute automatically detected.")
                protected_attr = input("Please enter the name of the protected attribute column: ")
                if protected_attr not in df.columns:
                    raise Exception(f"Column {protected_attr} not found in dataset")
                
            if target_attr not in df.columns:
                print(f"\nTarget attribute '{target_attr}' not found in dataset.")
                target_attr = input("Please enter the name of the target attribute column: ")
                if target_attr not in df.columns:
                    raise Exception(f"Column {target_attr} not found in dataset")
                
            print(f"Using protected attribute: {protected_attr}")
            print(f"Using target attribute: {target_attr}")
            return self.process_dataset(df, dataset_id, protected_attr, target_attr)
            
        except Exception as e:
            print(f"Error processing UCI dataset {dataset_id}: {str(e)}")
            return None

    def process_dataset(self, df, dataset_id, protected_attr, target_attr):
        """
        Process the dataset and evaluate fairness metrics.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the dataset.
        dataset_id : str
            The identifier for the dataset being processed.
        protected_attr : str
            The name of the protected attribute column.
        target_attr : str
            The name of the target attribute column.

        Returns
        -------
        bool
            True if processing was successful, False otherwise.
        """
        try:
            print(f"\nProcessing dataset {dataset_id}")
            print(f"Protected attribute: {protected_attr}")
            print(f"Target attribute: {target_attr}")
            
            # Check if the dataset is empty
            if df.empty:
                print(f"Error: Empty dataset for {dataset_id}")
                return False
            
            # Print basic dataset information
            print("\nDataset info:")
            print(f"Total rows before processing: {len(df)}")
            
            # Only remove rows with missing values in protected_attr and target_attr
            df = df.dropna(subset=[protected_attr, target_attr])
            print(f"Total rows after removing rows with missing protected/target attributes: {len(df)}")
            
            if df.empty:
                print("Error: No valid data after removing missing values in critical columns")
                return False
            
            # Handle missing values in other columns
            # Fill numeric columns with median
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
            
            # Fill categorical columns with mode
            categorical_columns = df.select_dtypes(exclude=[np.number]).columns
            for col in categorical_columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'missing')
            
            print("\nMissing value counts after imputation:")
            print(df.isnull().sum().sum())
            
            # Print distribution of protected attribute values
            print(f"\nProtected attribute ({protected_attr}) distribution before mapping:")
            print(df[protected_attr].value_counts())
            
            # Map gender values
            df[protected_attr] = self.map_gender_values(df[protected_attr])
            
            # Ensure target variable is encoded starting from 0
            le = LabelEncoder()
            y = le.fit_transform(df[target_attr])
            num_classes = len(np.unique(y))
            
            print(f"\nTarget attribute ({target_attr}) information:")
            print(f"Number of classes: {num_classes}")
            print(f"Original classes: {sorted(df[target_attr].unique())}")
            print(f"Encoded classes: {sorted(np.unique(y))}")
            
            if num_classes == 0:
                print("Error: No classes found in target attribute")
                return False
            
            # Prepare feature data
            X = df.drop([protected_attr, target_attr], axis=1)
            X = pd.get_dummies(X)
            
            if X.empty:
                print("Error: No features available after preprocessing")
                return False
            
            print(f"\nFeature matrix shape: {X.shape}")
            
            # Standardize the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split data into protected and unprotected groups
            X_protected = X_scaled[df[protected_attr] == 0]
            X_unprotected = X_scaled[df[protected_attr] == 1]
            y_protected = y[df[protected_attr] == 0]
            y_unprotected = y[df[protected_attr] == 1]
            
            print("\nSplit information:")
            print(f"Protected group size: {len(X_protected)}")
            print(f"Unprotected group size: {len(X_unprotected)}")
            
            # Initialize models with the correct number of classes for ResNet50
            self.models = {
                name: (self.all_models[name](num_classes) if name == 'ResNet50' 
                      else self.all_models[name]())
                for name in self.selected_algorithms
            }
            
            results = self.evaluate_fairness(X_protected, X_unprotected, 
                                              y_protected, y_unprotected,
                                              dataset_id)
            
            # Read existing results
            existing_results = pd.read_csv(self.results_file)
            
            # Update the row for this dataset
            dataset_mask = existing_results['Dataset_ID'] == dataset_id
            for algo in self.selected_algorithms:
                if algo in results:
                    existing_results.loc[dataset_mask, algo] = results[algo]
            
            # Update new columns for this dataset
            if 'k-value' in self.new_columns:
                total_samples = len(X_protected) + len(X_unprotected)
                k_value = 5 if total_samples <= 500 else (10 if total_samples <= 1000 else 20)
                existing_results.loc[dataset_mask, 'k-value'] = k_value
            
            def check_significance(value_str, definition=1):
                """
                Check if the significance of the given value string is below the threshold.

                Parameters
                ----------
                value_str : str
                    The string representation of the significance values.
                definition : int, optional
                    The definition of significance to check against (default is 1).

                Returns
                -------
                bool
                    True if significant, False otherwise.
                """
                if pd.isna(value_str) or value_str == "(nan, nan) nan":
                    return False
                try:
                    v1, v2 = map(float, value_str.split(')')[0].strip('(').split(','))
                    v3 = float(value_str.split(')')[1].strip())
                    if definition == 1:
                        return v1 < 0.05 and v2 < 0.05
                    else:
                        return v3 < 0.05
                except:
                    return False
            
            def get_significant_algorithms(row, algo_set, definition):
                """
                Get significant algorithms based on the defined significance level.

                Parameters
                ----------
                row : pd.Series
                    The row of results to check for significance.
                algo_set : set
                    The set of algorithms to check.
                definition : int
                    The definition of significance to check against.

                Returns
                -------
                str
                    A string of significant algorithms or 'okay (non-significant)'.
                """
                significant_algos = []
                for algo in algo_set:
                    if algo in row and check_significance(row[algo], definition):
                        significant_algos.append(algo)
                return ', '.join(significant_algos) if significant_algos else 'okay (non-significant)'
            
            # Update definition columns for this dataset
            for def_num in [1, 2]:
                tree_col = f'tree-based-def{def_num}'
                non_tree_col = f'non-tree-based-def{def_num}'
                
                if tree_col in self.new_columns:
                    existing_results.loc[dataset_mask, tree_col] = get_significant_algorithms(
                        existing_results.loc[dataset_mask].iloc[0], 
                        self.tree_based, 
                        def_num
                    )
                
                if non_tree_col in self.new_columns:
                    existing_results.loc[dataset_mask, non_tree_col] = get_significant_algorithms(
                        existing_results.loc[dataset_mask].iloc[0], 
                        self.non_tree_based, 
                        def_num
                    )
            
            # Save updated results
            existing_results.to_csv(self.results_file, index=False)
            print(f"Successfully processed dataset {dataset_id}")
            return True

        except Exception as e:
            print(f"Error processing dataset {dataset_id}: {str(e)}")
            import traceback
            traceback.print_exc()  # Print the full error stack
            return False

    def process_dataset_list(self, excel_path):
        """
        Process a list of datasets from an Excel file.

        Parameters
        ----------
        excel_path : str
            The path to the Excel file containing the dataset list.

        Returns
        -------
        None
        """
        try:
            # Read the full dataset list
            dataset_list = pd.read_excel(excel_path)
            
            # Filter to only include datasets that exist in final_results.csv
            dataset_list = dataset_list[dataset_list['ID'].isin(self.existing_dataset_ids)]
            
            print(f"Found {len(dataset_list)} datasets in final_results.csv")
            
            for _, row in dataset_list.iterrows():
                dataset_id = row['ID']
                url = row['URL']
                target_attr = row['Outcome_var_#_categories'].split('/')[0].strip()
                
                print(f"\nProcessing dataset: {dataset_id}")
                print(f"Target attribute: {target_attr}")
                
                if dataset_id.startswith('K'):
                    self.process_kaggle_dataset(dataset_id, url, target_attr)
                elif dataset_id.startswith('U'):
                    self.process_uci_dataset(dataset_id, url, target_attr)
                else:
                    print(f"Unknown dataset type for {dataset_id}")
            
        except Exception as e:
            print(f"Error processing dataset list: {str(e)}")

# Example usage with selected algorithms
selected_algorithms = ['SVM', 'RF', 'DT', 'LR', 'KNN', 'NB', 'ResNet50']
analyzer = FairnessAnalyzer('fairness_results', algorithms=selected_algorithms)
analyzer.process_dataset_list('/home/haolan/VRI/Datasets with EMD and p-value (v06).xlsx')