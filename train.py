import os
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')
from logging_setup import setup_logger
logger = setup_logger("train")
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, make_scorer
)
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from features import (
    load_and_split_data, 
    preprocess_features, 
    select_features,
    get_feature_importance_names
)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RANDOM_STATE = 42

def setup_models():
    """Setup model configurations with hyperparameters"""
    logger.info("Setting up model configurations")
    try:
        models = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=RANDOM_STATE),
                'params': {
                    'n_estimators': [300, 359, 400],
                    'max_depth': [4, 5, 6],
                    'min_samples_split': [11, 14, 17],
                    'min_samples_leaf': [8, 10, 12],
                    'bootstrap': [False, True],
                    'max_features': ['sqrt'],
                    'random_state': [42],
                    'class_weight': ['balanced','balanced_subsample']
                }
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(
                    use_label_encoder=False, 
                    eval_metric='logloss', 
                    random_state=RANDOM_STATE
                ),
                'params': {
                    'n_estimators': [350,400,450],
                    'max_depth': [12,16],
                    'learning_rate': [0.1, 0.15],
                    'min_child_weight': [2, 4],
                    'subsample': [0.9,1.0],
                    'colsample_bytree': [0.8, 1.0],
                    'scale_pos_weight': [2, 3],
                    'reg_lambda': [1.5,2],
                    'reg_alpha': [2, 2.5]
                }
            },
            'LightGBM': {
                'model': lgb.LGBMClassifier(
                        random_state=RANDOM_STATE,
                        verbosity=-1
                ),
                'params': {
                    'n_estimators': [250,300,350],
                    'max_depth': [9,15,20],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [30,40,50],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0],
                    'reg_lambda': [1,1.5],
                    'min_child_samples': [15,20],
                    'objective': ['binary'],
                    'scale_pos_weight': [5,8,11],
                    'boosting_type': ['gbdt'],
                    #'class_weight': ['balanced']
                }
            }
        }
        logger.info(f"Successfully configured {len(models)} models")
        return models
    except Exception as e:
        logger.error(f"Error setting up models: {str(e)}", exc_info=True)
        raise

def apply_smote(X_train, y_train):
    """Apply SMOTE to balance the training dataset"""
    logger.info("Applying SMOTE to balance training data")
    try:
        original_shape = X_train.shape
        original_class_dist = y_train.value_counts().to_dict()
        logger.debug(f"Original training data shape: {original_shape}")
        logger.debug(f"Original class distribution: {original_class_dist}")
        
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        new_shape = X_train_res.shape
        new_class_dist = y_train_res.value_counts().to_dict()
        logger.info(f"SMOTE applied successfully. New shape: {new_shape}")
        logger.debug(f"New class distribution: {new_class_dist}")
        
        return X_train_res, y_train_res
    except Exception as e:
        logger.error(f"Error applying SMOTE: {str(e)}", exc_info=True)
        logger.warning("Continuing without SMOTE - using original training data")
        return X_train, y_train

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test data"""
    logger.debug("Evaluating model performance")
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        metrics['classification_report'] = classification_report(y_test, y_pred)
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)

        logger.debug(f"Model evaluation completed. ROC AUC: {metrics['roc_auc']:.4f}")
        return metrics
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}", exc_info=True)
        raise

def train_single_model(model_name, model_config, X_train, y_train, X_test, y_test, use_grid_search=True):
    """Train a single model with optional hyperparameter tuning"""
    logger.info(f"Training {model_name} model")
    try:
        churn_recall_scorer = make_scorer(recall_score, pos_label=1)
        churn_f1_scorer = make_scorer(f1_score, pos_label=1)
        base_model = model_config['model']
        
        if use_grid_search and len(model_config['params']) > 0:
            logger.info(f"Performing hyperparameter tuning for {model_name} using GridSearchCV")

            cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            grid_search = RandomizedSearchCV(
                base_model,
                model_config['params'],
                cv = cv_folds,
                scoring={
                    'churn_recall': churn_recall_scorer,
                    'churn_f1': churn_f1_scorer,
                    'f1_weighted': "f1_weighted",
                    'precision_macro': "precision_macro",
                },
                refit='churn_recall',
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            logger.info(f"{model_name} - Best CV Score: {grid_search.best_score_:.4f}")
            logger.debug(f"{model_name} - Best Parameters: {grid_search.best_params_}")
            print(f"Best CV Score: {grid_search.best_score_:.4f}")
            print(f"Best Parameters: {grid_search.best_params_}")
        else:
            logger.info(f"Training {model_name} with default parameters (no grid search)")
            best_model = base_model
            best_model.fit(X_train, y_train)

        metrics = evaluate_model(best_model, X_test, y_test)

        results = {
            'model': best_model,
            'metrics': metrics,
            'model_name': model_name
        }

        logger.info(f"{model_name} training completed successfully")
        logger.info(f"{model_name} Results - Accuracy: {metrics['accuracy']:.4f}, "
                   f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
                   f"F1: {metrics['f1_score']:.4f}, ROC AUC: {metrics['roc_auc']:.4f}")

        print(f"\n{model_name} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")

        return results
    except Exception as e:
        logger.error(f"Error training {model_name}: {str(e)}", exc_info=True)
        raise

def train_all_models(models, X_train, y_train, X_test, y_test, use_smote=True):
    """Train all configured models"""
    logger.info(f"Starting training for {len(models)} models")
    trained_models = {}
    model_scores = {}

    try:
        if use_smote:
            X_train_res, y_train_res = apply_smote(X_train, y_train)
        else:
            logger.info("Skipping SMOTE - using original training data")
            X_train_res, y_train_res = X_train, y_train

        successful_models = 0
        for model_name, model_config in models.items():
            try:
                results = train_single_model(
                    model_name,
                    model_config,
                    X_train_res,
                    y_train_res,
                    X_test,
                    y_test
                )

                trained_models[model_name] = results['model']
                model_scores[model_name] = results['metrics']
                successful_models += 1
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                logger.warning(f"Continuing with other models...")
                continue

        logger.info(f"Successfully trained {successful_models}/{len(models)} models")
        return trained_models, model_scores
    except Exception as e:
        logger.error(f"Error in train_all_models: {str(e)}", exc_info=True)
        raise

def find_best_model(trained_models, model_scores):
    """Find the best performing model based on ROC AUC score"""
    logger.info("Finding best performing model")
    try:
        if not model_scores:
            logger.warning("No model scores available")
            return None
        
        best_score = 0
        best_name = None

        for name, metrics in model_scores.items():
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_name = name

        if best_name:
            best_model = {
                'name': best_name,
                'model': trained_models[best_name],
                'score': best_score
            }
            logger.info(f"Best model identified: {best_name} with ROC AUC: {best_score:.4f}")
            return best_model
        
        logger.warning("No best model could be determined")
        return None
    except Exception as e:
        logger.error(f"Error finding best model: {str(e)}", exc_info=True)
        return None

def save_models(trained_models, model_scores, best_model, feature_names, use_smote=True):
    """Save trained models and metadata"""
    logger.info("Saving trained models and metadata")
    try:
        if not trained_models:
            logger.warning("No trained models to save")
            return []
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        saved_paths = []

        # Save individual models
        for model_name, model in trained_models.items():
            try:
                filename = f"models/{model_name}_model_{TIMESTAMP}.joblib"
                joblib.dump(model, filename)
                saved_paths.append(filename)
                logger.debug(f"Saved {model_name} to {filename}")
                print(f"Saved {model_name} to {filename}")
            except Exception as e:
                logger.error(f"Failed to save {model_name}: {str(e)}")

        # Save best model separately
        if best_model:
            try:
                best_filename = f"models/best_model_{TIMESTAMP}.joblib"
                joblib.dump(best_model['model'], best_filename)
                saved_paths.append(best_filename)
                logger.info(f"Saved best model ({best_model['name']}) to {best_filename}")
            except Exception as e:
                logger.error(f"Failed to save best model: {str(e)}")

        # Save metadata
        try:
            metadata = {
                'timestamp': TIMESTAMP,
                'use_smote': use_smote,
                'model_scores': model_scores,
                'best_model_name': best_model['name'] if best_model else None,
                'feature_names': feature_names,
                'random_state': RANDOM_STATE
            }
            
            metadata_filename = f"models/model_metadata_{TIMESTAMP}.joblib"
            joblib.dump(metadata, metadata_filename)
            saved_paths.append(metadata_filename)
            logger.info(f"Saved metadata to {metadata_filename}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")
            
        logger.info(f"Successfully saved {len(saved_paths)} files")
        print(f"\nModels saved successfully:")
        for path in saved_paths:
            print(f"  - {path}")
            
        print(f"All models and metadata saved successfully")
        return saved_paths
    except Exception as e:
        logger.error(f"Error saving models: {str(e)}", exc_info=True)
        return []

def generate_training_report(model_scores, best_model, use_smote=True):
    """Generate detailed training report"""
    logger.info("Generating training report")
    try:
        # Create reports directory if it doesn't exist
        os.makedirs("reports", exist_ok=True)
        
        report_filename = f"reports/training_report_{TIMESTAMP}.txt"
        with open(report_filename, 'w') as f:
            f.write('-'*80 + '\n')
            f.write("CHURN PREDICTION MODEL TRAINING REPORT\n")
            f.write('-'*80 + '\n')
            f.write(f"Training Timestamp: {TIMESTAMP}\n")
            f.write(f"SMOTE Applied: {use_smote}\n")
            f.write(f"Random State: {RANDOM_STATE}\n\n")

            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write('-'*50 + '\n')

            for model_name, metrics in model_scores.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall:    {metrics['recall']:.4f}\n")
                f.write(f"  F1 Score:  {metrics['f1_score']:.4f}\n")
                f.write(f"  ROC AUC:   {metrics['roc_auc']:.4f}\n")

            if best_model:
                f.write(f"\nBEST MODEL: {best_model['name']}\n")
                f.write(f"ROC AUC Score: {best_model['score']:.4f}\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("DETAILED CLASSIFICATION REPORTS\n")
                f.write("="*80 + "\n")
                
                for model_name, metrics in model_scores.items():
                    f.write(f"\n{model_name} Classification Report:\n")
                    f.write("-"*40 + "\n")
                    f.write(metrics['classification_report'])
                    f.write("\n")
                    f.write(f"Confusion Matrix:\n")
                    f.write(str(metrics['confusion_matrix']))
                    f.write("\n\n")
        
        logger.info(f"Training report saved to: {report_filename}")
        print(f"\nTraining report saved to: {report_filename}")
        return report_filename
    except Exception as e:
        logger.error(f"Error generating training report: {str(e)}", exc_info=True)
        return None

def load_saved_model(model_path):
    """Load a saved model from disk"""
    logger.info(f"Loading model from {model_path}")
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}", exc_info=True)
        raise

def predict_with_model(model, X_data):
    """Make predictions using a trained model"""
    logger.debug("Making predictions with trained model")
    try:
        predictions = model.predict(X_data)
        probabilities = model.predict_proba(X_data)[:, 1]
        
        logger.debug(f"Generated {len(predictions)} predictions")
        return predictions, probabilities
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}", exc_info=True)
        raise

def main():
    """Main training pipeline"""
    logger.info("="*60)
    logger.info("CHURN PREDICTION MODEL TRAINING PIPELINE STARTED")
    logger.info("="*60)
    
    print("="*60)
    print("CHURN PREDICTION MODEL TRAINING PIPELINE")
    print("="*60)
    
    try:
        # Configuration
        USE_SMOTE = True
        FEATURE_SELECTION_K = 15
        
        logger.info(f"Configuration - USE_SMOTE: {USE_SMOTE}, FEATURE_SELECTION_K: {FEATURE_SELECTION_K}")
        
        # Load and split data
        logger.info("Loading and splitting data...")
        print("Loading and splitting data...")
        train_df, test_df = load_and_split_data(test_size=0.2, random_state=RANDOM_STATE)
        logger.info(f"Data loaded successfully. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        
        # Preprocess features
        logger.info("Preprocessing features...")
        print("Preprocessing features...")
        train_processed = preprocess_features(train_df, target_col='churn', is_training=True)
        test_processed = preprocess_features(test_df, target_col='churn', is_training=False)
        
        logger.info(f"Features preprocessed. Train: {train_processed.shape}, Test: {test_processed.shape}")
        print(f"Processed data shapes - Train: {train_processed.shape}, Test: {test_processed.shape}")
        
        # Feature selection
        logger.info(f"Performing feature selection (k={FEATURE_SELECTION_K})...")
        print(f"Performing feature selection (k={FEATURE_SELECTION_K})...")
        
        if 'churn' not in train_processed.columns:
            logger.error("Target column 'churn' not found in processed data")
            raise ValueError("Target column 'churn' not found in processed data")
        
        selected_train, selected_features = select_features(
            train_processed, 
            target_col='churn', 
            k=FEATURE_SELECTION_K
        )
        
        # Apply same feature selection to test data
        test_selected = test_processed[selected_features + ['churn']]
        
        logger.info(f"Selected {len(selected_features)} features")
        print(f"Selected {len(selected_features)} features")
        
        # Prepare training data
        X_train = selected_train.drop('churn', axis=1)
        y_train = selected_train['churn']
        X_test = test_selected.drop('churn', axis=1)
        y_test = test_selected['churn']
        
        feature_names = X_train.columns.tolist()
        logger.debug(f"Feature names: {feature_names}")

        # Setup models
        logger.info("Setting up models...")
        models = setup_models()
        
        # Train all models
        logger.info("Starting model training...")
        trained_models, model_scores = train_all_models(
            models, X_train, y_train, X_test, y_test, use_smote=USE_SMOTE
        )
        
        if not trained_models:
            logger.error("No models were successfully trained")
            raise RuntimeError("No models were successfully trained")
        
        # Find best model
        best_model = find_best_model(trained_models, model_scores)
        
        # Save models
        logger.info("Saving models...")
        saved_paths = save_models(trained_models, model_scores, best_model, feature_names, USE_SMOTE)
        
        # Generate report
        logger.info("Generating training report...")
        report_path = generate_training_report(model_scores, best_model, USE_SMOTE)
        
        # Summary
        summary_info = {
            'models_trained': len(trained_models),
            'best_model': best_model['name'] if best_model else 'None',
            'best_roc_auc': best_model['score'] if best_model else 0,
            'files_saved': len(saved_paths),
            'report_path': report_path
        }
        
        logger.info("Training pipeline completed successfully")
        logger.info(f"Summary: {summary_info}")
        
        print(f"\nSummary:")
        print(f"- Models trained: {len(trained_models)}")
        print(f"- Best model: {best_model['name'] if best_model else 'None'}")
        print(f"- Best ROC AUC: {best_model['score']:.4f}" if best_model else "")
        print(f"- Models saved: {len(saved_paths)} files")
        print(f"- Report saved: {report_path}")

        return {
            'trained_models': trained_models,
            'model_scores': model_scores,
            'best_model': best_model,
            'feature_names': feature_names,
            'saved_paths': saved_paths,
            'report_path': report_path
        }
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
        print(f"Training failed: {str(e)}")


if __name__ == '__main__':
    try:
        result = main()
        logger.info("Training completed successfully")
        print("Training completed successfully")
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}", exc_info=True)
        print(f"Script failed: {str(e)}")