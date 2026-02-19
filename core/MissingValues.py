# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 14:14:37 2025

"""
import pandas as pd
import numpy as np
from collections import defaultdict
import random

random.seed(42)
np.random.seed(42)

class MissingValues:
    """
    Handles missing values at different stages of decision tree processing:
    
    - During split evaluation (mv_split)
    - During data distribution after a split (mv_distrib)
    - During classification/prediction (mv_classif)
    
    The strategy for each stage is controlled via string parameters, which
    are typically evolved as genes in an evolutionary algorithm.
    """
    def __init__(self, mv_split='ignore_all', mv_distrib='ignore_all', mv_classif='explore_all', split_criterion=None):
        """
        Initializes the missing value handler with chosen strategies.
        
        Parameters:
            mv_split (str): Strategy to handle missing values during split evaluation.
            mv_distrib (str): Strategy to handle missing values during data distribution after a split.
            mv_classif (str): Strategy to handle missing values during classification.
            split_criterion: Optional reference to the split criterion object for use in 'weight_split'.
        """
        self.mv_split = mv_split
        self.mv_distrib = mv_distrib
        self.mv_classif = mv_classif
        self.split_criteria = split_criterion
    
    # HANDLING MISSING VALUES DURING SPLIT CRITERION EVALUATION
    # core/MissingValues.py
    def handle_split(self, X, y, feature):
        feature = self._standardize_feature_name(feature)

        if self.mv_split == "ignore_all":
            return self.ignore_all_split(X, y, feature)

        elif self.mv_split == "impute_mv":
            return self.impute_mv_split(X, y, feature)

        elif self.mv_split == "impute_mv_class":
            return self.impute_mv_class_split(X, y, feature)

        elif self.mv_split == "weight_split":
            # IMPORTANT: no threshold yet → behave like ignore_all
            return self.ignore_all_split(X, y, feature)

        else:
            raise ValueError(f"Unsupported criterion: {self.mv_split}")

#    def _standardize_feature_name(self, feature):
#        if isinstance(feature, str) and feature.isdigit():
#            return int(feature)
#        return feature

    def _standardize_feature_name(self, feature):
        if isinstance(feature, str) and feature.isdigit():
            return int(feature)
        return feature  # Ensure non-digit strings are returned unchanged


    # Gene 0: Ignore All instances with missing values
    def ignore_all_split(self, X, y, feature):
        # Ensure consistent indexing between X and y
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        # Build mask on synchronized DataFrame
        not_missing_mask = X[feature].notna()

        # Apply mask safely
        X_filtered = X.loc[not_missing_mask].reset_index(drop=True)
        y_filtered = y.loc[not_missing_mask].reset_index(drop=True)

        return X_filtered, y_filtered, X_filtered[feature]

    
    # Gene 1: Impute Missing Values with mode or mean
    def impute_mv_split(self, X, y, feature):
        """ 
        Impute missing values with mode or mean regardless of their class.
        """
        feature = self._standardize_feature_name(feature)
        X_copy = X.copy()
        # Check if values are numerical or categorical
        if pd.api.types.is_numeric_dtype(X_copy[feature]):
            # Mean if numerical
            fill_value = X_copy[feature].mean()
        else:
            # Mode if categorical
            fill_value = X_copy[feature].mode().iloc[0]  
        X_copy[feature] = X_copy[feature].fillna(fill_value)
        return X_copy, y, X_copy[feature]

    # Gene 2: Weight Splitting Criterion     
    def weight_split(self, X, y, feature, threshold):
        feature = self._standardize_feature_name(feature)
        if isinstance(feature, str) and feature.isdigit():
            feature = int(feature)
    
        proportion_missing = X[feature].isnull().mean()
    
        # Filter nan values in y to calculate split criterion
        mask_valid = ~X[feature].isnull()
        X_temp = X[mask_valid]
        y_temp = pd.Series(y)[mask_valid]
    
        # Erase nan values in y
        final_mask = ~y_temp.isnull()
        X_clean = X_temp[final_mask]
        y_clean = y_temp[final_mask]
        # print(" y_clean dans weight_split:", y_clean)
        
        X_column = X_clean[feature]

        gain = self.split_criteria.calculate(y_clean.to_numpy(), X_column.to_numpy(), threshold)
        adjusted_gain = gain * (1 - proportion_missing)
    
        return adjusted_gain

    # Gene 3: Impute Missing Values with mode/mean of instances of same class
    def impute_mv_class_split(self, X, y, feature):
        """ 
        Impute missing values with mode or mean of isntances of same class.
        """
        feature = self._standardize_feature_name(feature)
        X_copy = X.copy()
        
        # Loop through each instance with missing value
        for idx in X_copy[X_copy[feature].isna()].index:
            label = y[idx]
            # Select instances of same class without missing value
            same_class_mask = (y == label) & X_copy[feature].notna()
            same_class_values = X_copy.loc[same_class_mask, feature]
            
            # Handle case where same_class_mask is empty
            if len(same_class_values)==0:
                # Check if values are numerical or categorical
                if pd.api.types.is_numeric_dtype(X_copy[feature]):
                    fill_value = X_copy[feature].mean()
                else:
                    fill_value = X_copy[feature].mode().iloc[0]
            else :
                # Check if values are numerical or categorical
                if pd.api.types.is_numeric_dtype(X_copy[feature]):
                    fill_value = same_class_values.mean()
                else:
                    fill_value = same_class_values.mode().iloc[0]
            # fill missing values
            X_copy.at[idx, feature] = fill_value
        
        return X_copy, y, X_copy[feature]
    
    # HANDLING MISSING VALUES DURING DISTRIBUTION
    # How the distribution is done after we find the best split 
    def apply_split_with_mv(self, X, y, feature, split_value):
        if self.mv_distrib == 'ignore_all':
            return self.ignore_all_dis(X, y, feature, split_value)
        elif self.mv_distrib == 'most_common':
            return self.impute_mv_dis(X, y, feature, split_value)
        elif self.mv_distrib == 'class_specific_common':
            return self.impute_mv_class_dis(X, y, feature, split_value)
        elif self.mv_distrib == 'assign_all':
            return self.assign_all_dis(X, y, feature, split_value)
        elif self.mv_distrib == 'largest_partition':
            return self.largest_part_dis(X, y, feature, split_value)
        elif self.mv_distrib == 'weight_dis':
            return self.weight_dis(X, y, feature, split_value)
        elif self.mv_distrib == 'most_probable_partition':
            return self.most_probable_partition(X, y, feature, split_value)
        else:
            raise ValueError("Invalid distribution strategy specified.")
    
    # Gene 0: Ignoring all
    def ignore_all_dis(self, X, y, feature, split_value):
        """
        Distribute samples ignoring missing values entirely.
        Keeps only rows where feature is not null.
        """
        # ✅ Ensure X and y have aligned indices
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        # Build mask for non-missing values
        mask = X[feature].notnull()

        # Apply mask safely with .loc and reindex
        X_filtered = X.loc[mask].reset_index(drop=True)
        y_filtered = y.loc[mask].reset_index(drop=True)

        # Compute left/right masks and apply them
        left_mask = X_filtered[feature] <= split_value
        right_mask = X_filtered[feature] > split_value

        X_left = X_filtered.loc[left_mask].reset_index(drop=True)
        y_left = y_filtered.loc[left_mask].reset_index(drop=True)

        X_right = X_filtered.loc[right_mask].reset_index(drop=True)
        y_right = y_filtered.loc[right_mask].reset_index(drop=True)

        return (X_left, y_left), (X_right, y_right)

    
    # Gene 1: Impute Missing Values with mode or mean
    def impute_mv_dis(self, X, y, feature, split_value):
        """
        Impute missing values for a feature with its mode (if categorical)
        or mean (if numerical), regardless of class. Ensures X and y are aligned.
        """
        # ✅ Ensure X and y have aligned indices
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        # Standardize feature name for consistency
        feature = self._standardize_feature_name(feature)

        # ✅ Determine the most common value for the feature
        if X[feature].dtype == 'object':
            most_common = X[feature].mode()[0] if not X[feature].mode().empty else None
        else:
            most_common = X[feature].mean()

        # ✅ Fill missing values safely
        X_filled = X.copy()
        if most_common is not None:
            X_filled[feature] = X_filled[feature].fillna(most_common)

        # ✅ Perform split using aligned masks
        left_mask = X_filled[feature] <= split_value
        right_mask = X_filled[feature] > split_value

        X_left = X_filled.loc[left_mask].reset_index(drop=True)
        y_left = y.loc[left_mask].reset_index(drop=True)

        X_right = X_filled.loc[right_mask].reset_index(drop=True)
        y_right = y.loc[right_mask].reset_index(drop=True)

        return (X_left, y_left), (X_right, y_right)

    # Gene 2: Impute Missing Values with mode/mean of instances of same class
    def impute_mv_class_dis(self, X, y, feature, split_value):
        """
        Impute missing values with the mode (for categorical) or mean (for numeric)
        of instances belonging to the same class. Ensures alignment between X and y.
        """
        # ✅ Ensure X and y are aligned
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        feature = self._standardize_feature_name(feature)
        X_filled = X.copy()

        # ✅ Compute class-wise imputation values
        if X[feature].dtype == 'object':  # Categorical
            most_common_by_class = (
                X.groupby(y)[feature]
                .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
            )
        else:  # Numerical
            most_common_by_class = X.groupby(y)[feature].mean()

        # ✅ Fill missing values per class
        for class_label in most_common_by_class.index:
            mask = (y == class_label) & X_filled[feature].isnull()
            fill_value = most_common_by_class[class_label]
            if pd.notna(fill_value):
                X_filled.loc[mask, feature] = fill_value

        # ✅ Split instances into left/right partitions
        left_mask = X_filled[feature] <= split_value
        right_mask = X_filled[feature] > split_value

        X_left = X_filled.loc[left_mask].reset_index(drop=True)
        y_left = y.loc[left_mask].reset_index(drop=True)

        X_right = X_filled.loc[right_mask].reset_index(drop=True)
        y_right = y.loc[right_mask].reset_index(drop=True)

        return (X_left, y_left), (X_right, y_right)


    # Gene 3: Assign to all nodes
    def assign_all_dis(self, X, y, feature, split_value):
        """
        Distribute instances with missing values to both left and right partitions.
        Ensures X and y are index-aligned and uses .loc for safe masking.
        """
        # ✅ Ensure X and y have aligned integer indices
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        feature = self._standardize_feature_name(feature)

        # ✅ Compute masks
        left_mask = X[feature] <= split_value
        right_mask = X[feature] > split_value
        missing_mask = X[feature].isnull()

        # ✅ Create aligned splits
        X_left = X.loc[left_mask].reset_index(drop=True)
        y_left = y.loc[left_mask].reset_index(drop=True)

        X_right = X.loc[right_mask].reset_index(drop=True)
        y_right = y.loc[right_mask].reset_index(drop=True)

        # ✅ Handle missing-value instances
        X_missing = X.loc[missing_mask].reset_index(drop=True)
        y_missing = y.loc[missing_mask].reset_index(drop=True)

        # ✅ Assign missing rows to both partitions
        X_left = pd.concat([X_left, X_missing], ignore_index=True)
        y_left = pd.concat([y_left, y_missing], ignore_index=True)

        X_right = pd.concat([X_right, X_missing], ignore_index=True)
        y_right = pd.concat([y_right, y_missing], ignore_index=True)

        return (X_left, y_left), (X_right, y_right)


    # Gene 4: Assign to largest partition
    def largest_part_dis(self, X, y, feature, split_value):
        """
        Assign instances with missing values to the partition (left or right)
        that has the most samples. Ensures X and y are fully index-aligned.
        """
        # ✅ Ensure index alignment
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        feature = self._standardize_feature_name(feature)

        # ✅ Compute base masks for splitting
        left_mask = X[feature] <= split_value
        right_mask = X[feature] > split_value
        missing_mask = X[feature].isnull()

        # ✅ Create left/right splits safely
        X_left = X.loc[left_mask].reset_index(drop=True)
        y_left = y.loc[left_mask].reset_index(drop=True)

        X_right = X.loc[right_mask].reset_index(drop=True)
        y_right = y.loc[right_mask].reset_index(drop=True)

        # ✅ Determine which partition is larger
        if len(X_left) >= len(X_right):
            largest_X, largest_y = X_left, y_left
        else:
            largest_X, largest_y = X_right, y_right

        # ✅ Handle missing-value instances
        X_missing = X.loc[missing_mask].reset_index(drop=True)
        y_missing = y.loc[missing_mask].reset_index(drop=True)

        # ✅ Assign missing instances to the largest partition
        largest_X = pd.concat([largest_X, X_missing], ignore_index=True)
        largest_y = pd.concat([largest_y, y_missing], ignore_index=True)

        # ✅ Return updated splits
        if len(X_left) >= len(X_right):
            left_split = (largest_X.reset_index(drop=True), largest_y.reset_index(drop=True))
            right_split = (X_right.reset_index(drop=True), y_right.reset_index(drop=True))
        else:
            left_split = (X_left.reset_index(drop=True), y_left.reset_index(drop=True))
            right_split = (largest_X.reset_index(drop=True), largest_y.reset_index(drop=True))

        return left_split, right_split



    # Gene 5: assign based on probability of each partition
    def weight_dis(self, X, y, feature, split_value):
        """
        Distribute instances with missing values based on the probability of each partition.
        Ensures X and y are aligned and masks are applied safely.
        """
        # ✅ Ensure X and y have aligned indices
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        feature = self._standardize_feature_name(feature)

        # ✅ Create left/right base partitions
        left_mask = X[feature] <= split_value
        right_mask = X[feature] > split_value

        X_left = X.loc[left_mask].reset_index(drop=True)
        y_left = y.loc[left_mask].reset_index(drop=True)

        X_right = X.loc[right_mask].reset_index(drop=True)
        y_right = y.loc[right_mask].reset_index(drop=True)

        # ✅ Calculate partition probabilities
        total = len(X)
        left_prob = len(X_left) / total if total > 0 else 0
        right_prob = len(X_right) / total if total > 0 else 0

        # ✅ Identify missing-value instances
        missing_mask = X[feature].isnull()
        X_missing = X.loc[missing_mask].reset_index(drop=True)
        y_missing = y.loc[missing_mask].reset_index(drop=True)

        # ✅ Randomly assign missing instances according to probabilities
        rng = np.random.default_rng(seed=42)  # Reproducible random sampling

        # Calculate number of samples for each branch
        n_missing = len(X_missing)
        n_left = int(np.round(left_prob * n_missing))
        n_right = n_missing - n_left

        if n_missing > 0:
            if n_left > 0:
                idx_left = rng.choice(X_missing.index, size=n_left, replace=False)
                X_left = pd.concat([X_left, X_missing.loc[idx_left]], ignore_index=True)
                y_left = pd.concat([y_left, y_missing.loc[idx_left]], ignore_index=True)

            if n_right > 0:
                idx_right = [i for i in X_missing.index if i not in idx_left]
                X_right = pd.concat([X_right, X_missing.loc[idx_right]], ignore_index=True)
                y_right = pd.concat([y_right, y_missing.loc[idx_right]], ignore_index=True)

        # ✅ Reset indices after concatenation
        X_left = X_left.reset_index(drop=True)
        y_left = y_left.reset_index(drop=True)
        X_right = X_right.reset_index(drop=True)
        y_right = y_right.reset_index(drop=True)

        return (X_left, y_left), (X_right, y_right)

    
    # Gene 6: Assign to most probable partition
    def most_probable_partition(self, X, y, feature, split_value):
        """
        Assign instances with missing values to the partition that is most probable
        considering class distribution. Ensures full X/y index alignment.
        """
        # ✅ Ensure X and y are aligned
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        feature = self._standardize_feature_name(feature)

        # ✅ Create left/right masks and splits
        left_mask = X[feature] <= split_value
        right_mask = X[feature] > split_value
        missing_mask = X[feature].isnull()

        X_left = X.loc[left_mask].reset_index(drop=True)
        y_left = y.loc[left_mask].reset_index(drop=True)

        X_right = X.loc[right_mask].reset_index(drop=True)
        y_right = y.loc[right_mask].reset_index(drop=True)

        X_missing = X.loc[missing_mask].reset_index(drop=True)
        y_missing = y.loc[missing_mask].reset_index(drop=True)

        # ✅ Compute class-based probability of belonging to each side
        # (Here we approximate "probable partition" as the side with more samples per class)
        if len(X_left) == 0 and len(X_right) == 0:
            # Nothing to split
            return (X, y), (pd.DataFrame(), pd.Series())

        # Count class frequencies on each side
        left_class_counts = y_left.value_counts(normalize=True)
        right_class_counts = y_right.value_counts(normalize=True)

        # Decide which partition is "most probable" for each missing instance
        left_votes, right_votes = [], []
        for i in range(len(X_missing)):
            cls = y_missing.iloc[i]
            left_prob = left_class_counts.get(cls, 0)
            right_prob = right_class_counts.get(cls, 0)
            if right_prob > left_prob:
                right_votes.append(i)
            else:
                left_votes.append(i)

        # ✅ Assign missing instances based on class probabilities
        X_left_final = pd.concat(
            [X_left, X_missing.iloc[left_votes]], ignore_index=True
        ).reset_index(drop=True)
        y_left_final = pd.concat(
            [y_left, y_missing.iloc[left_votes]], ignore_index=True
        ).reset_index(drop=True)

        X_right_final = pd.concat(
            [X_right, X_missing.iloc[right_votes]], ignore_index=True
        ).reset_index(drop=True)
        y_right_final = pd.concat(
            [y_right, y_missing.iloc[right_votes]], ignore_index=True
        ).reset_index(drop=True)

        # ✅ Return both partitions safely
        return (X_left_final, y_left_final), (X_right_final, y_right_final)



    # Gene 1: Most probable path
    def _predict_most_probable(self, inputs, node):
        """
        Take the rout to the most probable partition (largest subset)
        """
        # While not a leaf
        while node.left:
            val = inputs[node.feat_idx]
            # If there is missing value
            if np.isnan(val):
                # Follow largest branch
                if node.left_weight >= node.right_weight:
                    node = node.left
                else:
                    node = node.right
            elif val <= node.threshold:
                node = node.left
            else:
                node = node.right
        
        return node.leaf_value if node.is_leaf else node.majority_class
    


    def predict_explore_all(self, inputs, node, weight=1.0):
        from collections import defaultdict
        import numpy as np

        if node.is_leaf:
            return {node.leaf_value: weight}

        if pd.isna(inputs[node.feature]):
            left_ratio = getattr(node, "left_weight", 0.5)
            right_ratio = getattr(node, "right_weight", 0.5)

            left_votes = self.predict_explore_all(
                inputs, node.left, weight * left_ratio
            )
            right_votes = self.predict_explore_all(
                inputs, node.right, weight * right_ratio
            )

            combined_votes = defaultdict(float)
            for cls, w in left_votes.items():
                combined_votes[cls] += w
            for cls, w in right_votes.items():
                combined_votes[cls] += w

            return combined_votes

        elif inputs[node.feature] <= node.threshold:
            return self.predict_explore_all(inputs, node.left, weight)
        else:
            return self.predict_explore_all(inputs, node.right, weight)
    