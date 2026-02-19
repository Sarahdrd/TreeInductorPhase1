# -*- coding: utf-8 -*-
"""
Created on Thu May  8 11:26:10 2025

@author: Aurora
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pandas as pd

le = LabelEncoder()

src_path = os.path.dirname(__file__)


def extract_data(dataset):
    
    # Bank of 15 datasets
    
    X = None
    y = None

    if dataset == "outlook":
        path = os.path.join(src_path, "datasets", "Play(Sheet1).csv")
        #path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\Play(Sheet1).csv"
        df = pd.read_csv(path, sep=";")
        df["Weather"] = le.fit_transform(df["Weather"])
        df["Temp"] = le.fit_transform(df["Temp"])
        df["Play"] = df["Play"].str.strip()
        df["Play_binary"] = df["Play"].map({"No":0, "Yes":1})

        # Split into features and target
        X = df[["Weather", "Temp"]].to_numpy()
        y = df["Play_binary"].to_numpy().astype(int)
        
    elif dataset == "abalone":
        path = os.path.join(src_path, "datasets", "abalone.data")
        #path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\abalone.data"
        df = pd.read_csv(path, sep=",",  header=None)
        
        # Name columns
        df.columns = ["Sex", "Length", "Diameter", "Height", "Whole_weight", 
                      "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]
        # Encode categorial values
        df["Sex_encoded"] = le.fit_transform(df["Sex"])
        
        # Split into features and target
        X = df[["Sex_encoded", "Length", "Diameter", "Height", 
                "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight"]].to_numpy()
        y = df["Rings"].to_numpy().astype(int)
        
    elif dataset == "anneal":
        path = os.path.join(src_path, "datasets", "anneal.data")
        #path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\anneal.data"
        df = pd.read_csv(path, sep=",")
        
        # Name columns
        df.columns = ["Family", "Product Type", "Steel", "Carbon", "Hardness", #ok
                      "Temper rolling", "Condition", "Formability", "Strenght", #ok
                      "Non-ageing", "Surface-finish","Surface-quality", #ok
                      "Enamelability", "Bc","Bf", "Bt", "Bw/me","Bl","M","Chrom", #ok
                      "Phos", "Cbond", "Marvi","Exptl", "Ferro", "Corr", #ok
                      "Blue/Bright/Varn/Clean", "Lustre", "Jurofm", "S", "P", #ok
                      "Shape","Thick","Width","Len","Oil","Bore","Packing",
                      "Classes"] 
        # Encode categorial values
        categorical_cols = [
            "Family", "Product Type", "Steel", "Temper rolling", "Condition", "Formability",
            "Non-ageing", "Surface-finish", "Surface-quality", "Enamelability", "Bc", "Bf",
            "Bt", "Bw/me", "Bl", "M", "Chrom", "Phos", "Cbond", "Marvi", "Exptl", "Ferro",
            "Corr", "Blue/Bright/Varn/Clean", "Lustre", "Jurofm", "S", "P", "Shape", "Oil",
            "Bore", "Packing", "Classes"
        ]
        for col in categorical_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        
        # Remove rare classes (classes with only 1 sample)
        y_labels = df["Classes"]
        class_counts = y_labels.value_counts()
        rare_classes = class_counts[class_counts < 2].index.tolist()
        df = df[~df["Classes"].isin(rare_classes)]
        
        # Split into features and target
        X = df.drop(columns=["Classes"]).to_numpy().astype(np.float64)
        y = df["Classes"].to_numpy().astype(int)
        
    elif dataset == "arrhythmia":
        path = os.path.join(src_path, "datasets", "arrhythmia.data")
        #path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\arrhythmia.data"
        df = pd.read_csv(path, sep=",", header=None)
        
        # Name columns
        columns = [
            "Age", "Sex", "Height", "Weight", "QRS duration", "P-R interval", "Q-T interval",
            "T interval", "P interval",
            "QRS angle", "T angle", "P angle", "QRST angle", "J angle",
            "Heart rate",
        ]
        # to generate repetitive names of columns
        def wave_cols(prefix, start):
            labels = ["Q", "R", "S", "R'", "S'", "Nb deflections",
                      "R ragged", "R diphasic", "P ragged", "P diphasic", "T ragged", "T diphasic"]
            return [f"{prefix} {label}" for label in labels]
        
        def amplitude_cols(prefix):
            labels = ["JJ", "Q", "R", "S", "R'", "S'", "P", "T", "QRSA", "QRSTA"]
            return [f"{prefix} {label} amplitude" for label in labels]
        
        channels = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        
        for ch in channels:
            columns.extend(wave_cols(ch, start=len(columns)))
        
        for ch in channels:
            columns.extend(amplitude_cols(ch))
        columns.append("Classes")
        df.columns = columns
        
        # Encode categorial values
        non_numeric_cols = ['T angle', 'P angle', 'QRST angle', 'J angle', 'Heart rate']
        
        for col in non_numeric_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('[^\d.-]', '', regex=True), errors='coerce')

        # Split into features and target
        X = df.drop(columns=["Classes"]).to_numpy()
        y = df["Classes"].to_numpy().astype(int)
        counts = pd.Series(y).value_counts()
        valid_classes = counts[counts >= 2].index
        mask = np.isin(y, valid_classes)
        
        X = X[mask]
        y = y[mask]
         
    elif dataset == "audiology":
        path = os.path.join(src_path, "datasets", "audiology_standardized.data")
        # Load the dataset
        #path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\audiology_standardized.data"
        df = pd.read_csv(path, sep=",")
        
        # Define column names
        df.columns = [
            "age_gt_60", "air", "airBoneGap", "ar_c", "ar_u", "bone", "boneAbnormal", "bser",
            "history_buzzing", "history_dizziness", "history_fluctuating", "history_fullness",
            "history_heredity", "history_nausea", "history_noise", "history_recruitment",
            "history_ringing", "history_roaring", "history_vomiting", "late_wave_poor",
            "m_at_2k", "m_cond_lt_1k", "m_gt_1k", "m_m_gt_2k", "m_m_sn", "m_m_sn_gt_1k",
            "m_m_sn_gt_2k", "m_m_sn_gt_500", "m_p_sn_gt_2k", "m_s_gt_500", "m_s_sn",
            "m_s_sn_gt_1k", "m_s_sn_gt_2k", "m_s_sn_gt_3k", "m_s_sn_gt_4k", "m_sn_2_3k",
            "m_sn_gt_1k", "m_sn_gt_2k", "m_sn_gt_3k", "m_sn_gt_4k", "m_sn_gt_500",
            "m_sn_gt_6k", "m_sn_lt_1k", "m_sn_lt_2k", "m_sn_lt_3k", "middle_wave_poor",
            "mod_gt_4k", "mod_mixed", "mod_s_mixed", "mod_s_sn_gt_500", "mod_sn", "mod_sn_gt_1k",
            "mod_sn_gt_2k", "mod_sn_gt_3k", "mod_sn_gt_4k", "mod_sn_gt_500", "notch_4k",
            "notch_at_4k", "o_ar_c", "o_ar_u", "s_sn_gt_1k", "s_sn_gt_2k", "s_sn_gt_4k",
            "speech", "static_normal", "tymp", "viith_nerve_signs", "wave_V_delayed",
            "waveform_ItoV_prolonged", "identifier", "classes"
        ]
        
        df = df.drop(columns=["identifier"])
        
        # Replace "?" with np.nan to handle missing values
        df.replace("?", np.nan, inplace=True)
        
        categorical_cols = df.columns.tolist()
        # Define the order for columns with specific categories
        ordered_categories = {
            "air": ["mild", "moderate", "severe", "normal", "profound"],
            "ar_c": ["normal", "elevated", "absent"],
            "ar_u": ["normal", "elevated", "absent"],
            "bone": ["mild", "moderate", "normal", "unmeasured"],
            "bser": ["normal", "degraded"],
            "o_ar_c": ["normal", "elevated", "absent"],
            "o_ar_u": ["normal", "elevated", "absent"],
            "speech": ["normal", "good", "very_good", "poor", "very_poor", "unmeasured"],
            "tymp": ["a", "as", "b", "ad", "c"],
        }
        special_order_cols = list(ordered_categories.keys())
        
        # Encode columns with a specific category order
        encoder_special = OrdinalEncoder(
            categories=[ordered_categories[col] for col in special_order_cols],
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan
        )
        df[special_order_cols] = encoder_special.fit_transform(df[special_order_cols])
        
        # Encode boolean columns ('f' -> 0, 't' -> 1)
        bool_cols = [col for col in categorical_cols if col not in special_order_cols and col not in ["identifier", "classes"]]
        for col in bool_cols:
            df[col] = df[col].map({'f': 0, 't': 1})
        
        # Remove rare classes (classes with only 1 sample)
        y_labels = df["classes"]
        class_counts = y_labels.value_counts()
        rare_classes = class_counts[class_counts < 2].index.tolist()
        df = df[~df["classes"].isin(rare_classes)]
        
        X = df.drop(columns=["classes"]).to_numpy().astype(np.float64)
        y = le.fit_transform(df["classes"])
        
    elif dataset == "car":
        path = os.path.join(src_path, "datasets", "car.data")
        #path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\car.data"
        df = pd.read_csv(path, sep=",")
        
        # Name columns
        df.columns = ["Buying", "Maint", "Doors", "Persons", "Lug_Boot", 
                      "Safety", "classes"] 
        
        # Encode categorial values
        categorical_cols = df.columns.tolist()
        
        ordered_categories = {
            "Buying": ["vhigh", "high", "med", "low"],
            "Maint": ["vhigh", "high", "med", "low"],
            "Doors": ["2", "3", "4", "5more"],
            "Persons": ["2", "4", "more"],
            "Lug_Boot": ["small", "med", "big"],
            "Safety": ["high", "med", "low"],
 
        }
        special_order_cols = list(ordered_categories.keys())
        
        # Encode columns with a specific category order
        encoder_special = OrdinalEncoder(
            categories=[ordered_categories[col] for col in special_order_cols],
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan
        )
        df[special_order_cols] = encoder_special.fit_transform(df[special_order_cols])
        
        
        # Encode classes manually
        class_order = ["unacc", "acc", "good", "vgood"]
        class_mapping = {label: idx for idx, label in enumerate(class_order)}
        df["classes"] = df["classes"].map(class_mapping)
        
        # Remove rare classes (if needed, but for 'car' dataset normally all classes have enough samples)
        y_labels = df["classes"]
        class_counts = y_labels.value_counts()
        rare_classes = class_counts[class_counts < 2].index.tolist()
        df = df[~df["classes"].isin(rare_classes)]
        
        # Prepare X and y
        X = df.drop(columns=["classes"]).to_numpy().astype(np.float64)
        y = df["classes"].to_numpy().astype(int)
    
    elif dataset == "dermatology":
        path = os.path.join(src_path, "datasets", "dermatology.data")
        #path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\dermatology.data"
        df = pd.read_csv(path, sep=",")
        
        # Replace "?" with np.nan to handle missing values
        df.replace("?", np.nan, inplace=True)
        
        # Name columns
        df.columns = ["erythema", "scaling", "definite borders", "itching", "koebner phenomenon", 
                      "polygonal papules", "follicular papules", "oral mucosal involvement", 
                      "knee and elbow involvement", "scalp involvement", "family history", 
                      "melanin incontinence", "eosinophils in the infiltrate", "PNL infiltrate",
                      "fibrosis of the papillary dermis", "exocytosis", "acanthosis", "hyperkeratosis",
                      "parakeratosis", "clubbing of the rete ridges", "elongation of the rete ridges",
                      "thinning of the suprapapillary epidermis", "spongiform pustule", 
                      "munro microabcess", "focal hypergranulosis", "disappearance of the granular layer",
                      "vacuolisation and damage of basal layer", "spongiosis", 
                      "saw-tooth appearance of retes", "follicular horn plug", "perifollicular parakeratosis",
                      "inflammatory monoluclear inflitrate", "band-like infiltrate", "age", "classes"]
    
        # Prepare X and y
        X = df.drop(columns=["classes"]).to_numpy().astype(np.float64)
        y = df["classes"].to_numpy().astype(int)

    elif dataset == "ecoli":
        path = os.path.join(src_path, "datasets", "ecoli.data")
        #path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\ecoli.data"
        df = pd.read_csv(path, delim_whitespace=True, header=None, on_bad_lines='skip')
        
        # Replace "?" with np.nan to handle missing values
        df.replace("?", np.nan, inplace=True)

        # Name columns
        df.columns = ["Sequence Name", "mcg", "gvh", "lip", "chg", "aac", "alm1",
                      "alm2", "classes"]
        
        # Encode
        df["classes"] = LabelEncoder().fit_transform(df["classes"].astype(str))
        df["Sequence Name"] = LabelEncoder().fit_transform(df["Sequence Name"].astype(str))
    
        # Prepare X and y
        X = df.drop(columns=["classes"]).to_numpy().astype(np.float64)
        y = df["classes"].to_numpy().astype(int)
        
    elif dataset == "glass":
        path = os.path.join(src_path, "datasets", "glass.data")
        #path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\glass.data"
        df = pd.read_csv(path, sep=",")
        
        # Name columns
        df.columns = ["identifier", "RI", "Na", "Mg", "AI", "Si", "K", "Ca", "Ba", "Fe", "classes"] 
        df = df.drop(columns=["identifier"])
        
        # Prepare X and y
        X = df.drop(columns=["classes"]).to_numpy().astype(np.float64)
        y = df["classes"].to_numpy().astype(int)
        
    elif dataset == "hepatitis":
        path = os.path.join(src_path, "datasets", "hepatitis.data")
        #path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\hepatitis.data"
        df = pd.read_csv(path, sep=",")
        
        # Name columns
        df.columns = ["AGE", "SEX", "STEROID", "ANTIVIRALS", "FATIGUE",
                      "MALAISE", "ANOREXIA", "LIVER BIG", "LIVER FIRM",
                      "SPLEEN PALPABLE", "SPIDERS", "ASCITES", "VARICES", 
                      "BILIRUBIN", "ALK PHOSPHATE", "SGOT", "ALBUMIN", "PROTIME",
                      "HISTOLOGY", "classes"]
        
        # Replace "?" with np.nan to handle missing values
        df.replace("?", np.nan, inplace=True)
        
        X = df.drop(columns=["classes"]).to_numpy().astype(np.float64)
        y = df["classes"].to_numpy().astype(int)
    
    elif dataset == "iris":
        path = os.path.join(src_path, "datasets", "iris.data")
        #path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\iris.data"
        df = pd.read_csv(path, sep=",")
        
        # Name columns
        df.columns = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width",
                      "classes"]
        
        df["classes"] = LabelEncoder().fit_transform(df["classes"].astype(str))
        
        X = df.drop(columns=["classes"]).to_numpy().astype(np.float64)
        y = df["classes"].to_numpy().astype(int)
    
    elif dataset == "segment":
        path = os.path.join(src_path, "datasets", "segment.data")
        #path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\segment.data"
        df = pd.read_csv(path, sep=",")
        
        # Name columns
        df.columns = ["classes", "region-centroid-col", "region-centroid-row", 
                      "region-pixel-count", "short-line-density-5", "short-line-density-2",
                      "vedge-mean", "vegde-sd", "hedge-mean", "hedge-sd", "intensity-mean",
                      "rawred-mean", "rawblue-mean", "rawgreen-mean", "exred-mean",
                      "exblue-mean", "exgreen-mean", "value-mean", "saturatoin-mean",
                      "hue-mean"]
        
        X = df.drop(columns=["classes"]).to_numpy().astype(np.float64)
        y = le.fit_transform(df["classes"])
    
    elif dataset == "semeion":
        path = os.path.join(src_path, "datasets", "semeion.data")
        #path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\semeion.data"
        df = pd.read_csv(path, sep=" ")
        df = df.dropna(axis=1, how='all')
        
        X = df.iloc[:, :256].values  
        y_onehot = df.iloc[:, 256:].values  
        
        y = np.argmax(y_onehot, axis=1)
        
    elif dataset == "shuttle_landing":
        path = os.path.join(src_path, "datasets", "shuttle_landing.data")
        #path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\shuttle_landing.data"
        df = pd.read_csv(path, sep=",")
        
        # Name columns
        df.columns = ["Stability", "Error", "Sign", "Wind", "Magnitude", 
                      "Visibility", "classes"]
        # Replace "*" with np.nan to handle missing values
        df.replace("*", np.nan, inplace=True)
        
        df["classes"] = LabelEncoder().fit_transform(df["classes"].astype(str))
        
        X = df.drop(columns=["classes"]).to_numpy().astype(np.float64)
        y = df["classes"].to_numpy().astype(int)
    
    elif dataset == "vowel":
        path = os.path.join(src_path, "datasets", "vowel.data")
        #path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\vowel.data"
        df = pd.read_csv(path, sep=",")
        
        df.columns = ["classes", "x-box", "y-box", "width", "high", 
                      "onpix", "x-bar", "y-bar", "x2bar", "y2bar",
                      "xybar", "x2ybr", "xy2br", "X-ege", "xegvy",
                      "y-ege", "yegvx"]
        
        df["classes"] = LabelEncoder().fit_transform(df["classes"].astype(str))
        
        X = df.drop(columns=["classes"]).to_numpy().astype(np.float64)
        y = df["classes"].to_numpy().astype(int)
        
    elif dataset == "winequality-red":
        path = os.path.join(src_path, "datasets", "winequality-red.csv")
        #path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\winequality-red.csv"
        df = pd.read_csv(path, sep=";")
        
        # Split into features and target
        X = df[["fixed acidity", "volatile acidity", "citric acid", 
                "residual sugar", "chlorides", "free sulfur dioxide",
                "total sulfur dioxide", "density", "pH", "sulphates",
                "alcohol"]].to_numpy()
        y = df["quality"].to_numpy().astype(int)
        
    elif dataset == "winequality-white":
        path = os.path.join(src_path, "datasets", "winequality-white.csv")
        #path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\winequality-white.csv"
        df = pd.read_csv(path, sep=";")

        # Split into features and target
        X = df[["fixed acidity", "volatile acidity", "citric acid", 
                "residual sugar", "chlorides", "free sulfur dioxide",
                "total sulfur dioxide", "density", "pH", "sulphates",
                "alcohol"]].to_numpy()
        y = df["quality"].to_numpy().astype(int)
        
        
        
        
        
        
        
    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # After split, keep only classes that have at least 2 samples in y_test
    y_test_counts = pd.Series(y_test).value_counts()
    valid_classes = y_test_counts[y_test_counts >= 2].index
    mask_test = np.isin(y_test, valid_classes)
    
    X_test = X_test[mask_test]
    y_test = y_test[mask_test]
    
    # Filter training set to only keep classes present in the new test set
    mask_train = np.isin(y_train, valid_classes)
    
    X_train = X_train[mask_train]
    y_train = y_train[mask_train]
        
    return X_train, X_test, y_train, y_test