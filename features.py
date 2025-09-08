import pandas as pd
import numpy as np
import logging
from ingestion_db import create_connection
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import os
import sys

# Creating logs directory if it doesn't exist.
os.makedirs("logs", exist_ok=True)

# Create a module-specific logger
logger = logging.getLogger('features')
logger.setLevel(logging.DEBUG)

# Clear any existing handlers
logger.handlers.clear()

# Create file handler specifically for features
file_handler = logging.FileHandler("logs/features.log", mode="a")
file_handler.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Prevent propagation to root logger to avoid conflicts
logger.propagate = False
# Global variable for feature names
FEATURE_NAMES = []


def create_features(df):
    '''Create new engineering features.'''
    try:
        logging.info("Starting feature engineering")
        df_copy = df.copy()

        # Tenure based features
        if 'tenure' in df_copy.columns:
            df_copy['tenure_year'] = df_copy['tenure'] / 12
            df_copy['tenure_segment'] = pd.cut(df_copy['tenure'], 
                                               bins=[0, 6, 12, 24, 48, 100], 
                                               labels=['0-6m', '6m-1yr', '1-2yr', '2-4yr', '4+yr'])
            logging.info("Created tenure-based features")
        else:
            logging.warning("'tenure' column not found, skipping tenure features")
        
        # Contract and service features
        if 'contract' in df_copy.columns:
            df_copy['is_month_to_month'] = (df_copy['contract'].str.lower().str.replace(' ', '-') == 'month-to-month').astype('int64')
            df_copy['has_long_contract'] = (df_copy['contract'].isin(['One year', 'Two year'])).astype('int64')
            logging.info("Created contract-based features")
        else:
            logging.warning("'contract' column not found, skipping contract features")
        
        # Service bundle score
        service_cols = ['onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport']
        available_service_cols = [col for col in service_cols if col in df_copy.columns]
        if available_service_cols:
            df_copy['service_bundle_score'] = df_copy[available_service_cols].sum(axis=1)
            logging.info(f"Created service bundle score using {len(available_service_cols)} columns")
        else:
            logging.warning("No service columns found for bundle score")

        # Payment and charges features
        if 'totalcharges' in df_copy.columns and 'tenure' in df_copy.columns:
            df_copy['charges_per_tenure'] = df_copy['totalcharges'] / (df_copy['tenure'] + 1)
            logging.info("Created charges per tenure feature")
        
        if 'monthlycharges' in df_copy.columns:
            df_copy['high_monthly_charges'] = (df_copy['monthlycharges'] > df_copy['monthlycharges'].median()).astype(int)
            logging.info("Created high monthly charges feature")
        
        if 'paymentmethod' in df_copy.columns:
            df_copy['high_risk_payment'] = (df_copy['paymentmethod'] == 'Electronic check').astype('int64')
            logging.info("Created high risk payment feature")

        # Customer profile features
        if 'partner' in df_copy.columns and 'dependents' in df_copy.columns:
            df_copy['single_customer'] = ((df_copy['partner'] == 0) & (df_copy['dependents'] == 0)).astype('int64')
            df_copy['family_customer'] = ((df_copy['partner'] == 1) | (df_copy['dependents'] == 1)).astype('int64')
            logging.info("Created customer profile features")

        logging.info(f"Feature engineering completed. Shape: {df_copy.shape}")
        return df_copy
        
    except Exception as e:
        logging.error(f"Error in feature engineering: {str(e)}")
        raise


def preprocess_features(df, target_col='churn', is_training=True):
    '''Preprocess and clean features.'''
    try:
        logging.info(f"Starting feature preprocessing. Input shape: {df.shape}")
        global FEATURE_NAMES
        pre_df = df.copy()

        # Clean column names first
        pre_df.columns = pre_df.columns.str.lower().str.strip().str.replace(' ', '_')
        target_col = target_col.lower()
        
        # Handle missing values
        initial_rows = len(pre_df)
        pre_df = pre_df.dropna()
        logging.info(f"Dropped {initial_rows - len(pre_df)} rows with missing values")

        # Handle totalcharges column
        if 'totalcharges' in pre_df.columns:
            pre_df['totalcharges'] = pd.to_numeric(pre_df['totalcharges'], errors='coerce')
            missing_charges = pre_df['totalcharges'].isna().sum()
            if missing_charges > 0:
                median_charges = pre_df['totalcharges'].median()
                pre_df['totalcharges'] = pre_df['totalcharges'].fillna(median_charges)
                logging.info(f"Filled {missing_charges} missing totalcharges with median: {median_charges}")
        
        # Process object columns
        object_cols = pre_df.select_dtypes(include=['object']).columns
        for col in object_cols:
            if col == target_col:
                continue
            try:
                uniq_val = pre_df[col].dropna().unique()
                
                # Handle gender columns
                if len(uniq_val) <= 2 and set(str(v).lower() for v in uniq_val) <= {'male', 'female'}:
                    gender_map = {}
                    for val in uniq_val:
                        if str(val).lower() == 'male':
                            gender_map[val] = 1
                        elif str(val).lower() == 'female':
                            gender_map[val] = 0
                    pre_df[col] = pre_df[col].map(gender_map).astype('int64')
                    logging.info(f"Mapped gender column '{col}' to binary values")

                # Handle yes/no columns
                elif len(uniq_val) <= 2 and set(str(v).lower() for v in uniq_val) <= {'yes', 'no'}:
                    yesno_map = {}
                    for val in uniq_val:
                        if str(val).lower() == 'yes':
                            yesno_map[val] = 1
                        elif str(val).lower() == 'no':
                            yesno_map[val] = 0
                    pre_df[col] = pre_df[col].map(yesno_map).astype('int64')
                    logging.info(f"Mapped yes/no column '{col}' to binary values")
                
                # Handle numeric string columns
                elif all(str(val).replace('.', '').isdigit() for val in uniq_val if pd.notna(val)):
                    pre_df[col] = pre_df[col].astype('int64')
                    logging.info(f"Converted column '{col}' to integer")
                    
            except Exception as e:
                logging.warning(f"Error processing column '{col}': {str(e)}")
        
        # Apply feature engineering
        df_features = create_features(pre_df)

        # Convert categorical columns to numeric
        for col in df_features.select_dtypes(include=['category']).columns:
            df_features[col] = pd.factorize(df_features[col])[0]
            logging.info(f"Converted category column '{col}' to numeric")

        # One-hot encode categorical columns with more than 2 categories
        obj_col = ['contract', 'internetservice', 'paymentmethod']
        obj_col = [col for col in obj_col if col in df_features.columns]
        
        encoded_col = []
        for col in obj_col:
            try:
                ohe = OneHotEncoder(sparse_output=False, drop='first')
                encoded = ohe.fit_transform(df_features[[col]])
                feature_names = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]
                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df_features.index)
                encoded_col.append(encoded_df)
                logging.info(f"One-hot encoded column '{col}' into {len(feature_names)} features")
            except Exception as e:
                logging.error(f"Error one-hot encoding column '{col}': {str(e)}")

        # Drop original categorical columns
        df_features = df_features.drop(columns=obj_col)

        # Drop unwanted columns
        drop_col = ['customerid', 'tenure', 'partner', 'dependents']
        drop_col = [col for col in drop_col if col in df_features.columns]
        df_features = df_features.drop(columns=drop_col)
        logging.info(f"Dropped columns: {drop_col}")
        
        # Concatenate all dataframes
        clean_df = pd.concat([df_features] + encoded_col, axis=1)

        # Clean column names
        clean_df.columns = clean_df.columns.str.replace(' ', '_')

        # Store feature names for importance analysis
        if is_training and target_col in clean_df.columns:
            FEATURE_NAMES = [col for col in clean_df.columns if col != target_col]
            logging.info(f"Stored {len(FEATURE_NAMES)} feature names for training")
        
        logging.info(f"Feature preprocessing completed. Output shape: {clean_df.shape}")
        return clean_df
        
    except Exception as e:
        logging.error(f"Error in feature preprocessing: {str(e)}")
        raise


def select_features(df, target_col='churn', k=10):
    '''Select top k features based on statistical tests.'''
    try:
        logging.info(f"Starting feature selection with k={k}")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        if X.empty:
            raise ValueError("No features available for selection")
        
        # Select top k features using f_classif
        k_actual = min(k, X.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k_actual)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Create dataframe with selected features
        selected_df = pd.DataFrame(X_selected, columns=selected_features, index=df.index)
        selected_df[target_col] = y.values
        
        logging.info(f"Selected {len(selected_features)} features: {selected_features}")
        # print(f"Selected {len(selected_features)} features: {selected_features}")
        
        return selected_df, selected_features
        
    except Exception as e:
        logging.error(f"Error in feature selection: {str(e)}")
        raise


def get_feature_importance_names(df, target_col='churn', top_n=10):
    '''Get feature importance using Random Forest and return top feature names.'''
    try:
        logging.info(f"Calculating feature importance for top {top_n} features")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        if X.empty:
            raise ValueError("No features available for importance calculation")
        
        # Train Random Forest to get feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        logging.info("Top Feature Importances calculated successfully")
        print("Top Feature Importances:")
        for idx, row in top_features.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
        
        return top_features['feature'].tolist(), importance_df
        
    except Exception as e:
        logging.error(f"Error calculating feature importance: {str(e)}")
        raise


def load_and_split_data(test_size=0.2, random_state=42):
    '''Load, clean, transform the Data and split into train/test sets.'''
    try:
        logging.info("Loading data from database")
        
        engine = create_connection()
        df = pd.read_sql_query("SELECT * FROM telecom_data", engine)
        df.columns = df.columns.str.lower().str.strip().str.replace(' ','_')
        
        logging.info(f"Loaded data with shape: {df.shape}")
        
        # Process target column
        if 'churn' in df.columns:
            #df['churn'] = df['churn'].astype(str).str.lower().str.strip().map
            df['churn'] = (df['churn'].str.lower().str.strip().map({'yes':1,'no':0})).astype('int64')

        # Split the data into train/test set
        stratify_col = df['churn'] if 'churn' in df.columns else None
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )

        logging.info(f"Data split completed - Train: {len(train_df)}, Test: {len(test_df)}")
        print(f"Data split completed - Train: {len(train_df)}, Test: {len(test_df)}")

        return train_df, test_df
        
    except Exception as e:
        logging.error(f"Error loading and splitting data: {str(e)}")
        raise
    
    finally:
          # Clean up
          if 'engine' in locals():
               engine.dispose()
               logger.info("Database connection closed")


def main():
    '''Main function to demonstrate the features module functionality.'''
    try:
        logging.info("=== Starting Features Module Demo ===")
        
        # Load and split data
        train_df, test_df = load_and_split_data()
        
        # Preprocess features
        train_processed = preprocess_features(train_df, is_training=True)
        test_processed = preprocess_features(test_df, is_training=False)
        
        print(f"\nProcessed data shapes:")
        print(f"Train: {train_processed.shape}")
        print(f"Test: {test_processed.shape}")
        
        # Feature selection
        if 'churn' in train_processed.columns:
            selected_df, selected_features = select_features(train_processed, k=15)
            print(f"\nSelected features shape: {selected_df.shape}")
            
            # Feature importance
            top_features, importance_df = get_feature_importance_names(train_processed, top_n=10)
            print(f"\nTop 10 important features identified")
            
        logging.info("=== Features Module Demo Completed Successfully ===")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        print(f"Error occurred: {str(e)}")
        raise


if __name__ == '__main__':
    main()