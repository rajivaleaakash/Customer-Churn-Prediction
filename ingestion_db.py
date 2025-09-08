from sqlalchemy import create_engine,text
import pandas as pd
import os
import logging
import time

# Creating logs directory if it doesn't exist.
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="logs/ingestion_db.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"
)

logger = logging.getLogger(__name__)


def create_connection():
    '''Create and return database engine'''
    try:
        server = r'AAKASH-LAPTOP\SQLEXPRESS'
        database = r'db_churn'
        connection_string = f"mssql+pyodbc://@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
        engine = create_engine(connection_string)

        # Test Connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection established successfully")
        return engine
    except Exception as e:
        logger.error(f"Failed to create database connection: {e}")
        raise

def ingest_db(df, table_name, engine):
        """This function will ingest the dataframe into database table"""
        try:
            clean_table_name = table_name.replace(' ','_').replace('-','_').replace('.','_')
            logger.info(f"Starting ingestion for table: {clean_table_name}")
            logger.info(f"DataFrame shape: {df.shape}")
            df.to_sql(
                table_name,
                con = engine,
                if_exists='replace',
                index=False
                #chunksize = 1000,
                #method='multi'
            )
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM [{clean_table_name}] limit"))
                count = result.fetchone()[0]
                logger.info(f"Successfully ingested {count} records into table '{clean_table_name}'")

            return True
        
        except Exception as e:
            logger.error(f"Failed to ingest data into table '{table_name}: {e}")
            return False


def load_raw_data():
     """This function will load CSV data into dataframe and ingest into db"""
     data_directory = r"d:\Elevated_Lab\Customer Churn Analysis for Telecom Industry"

     if not os.path.exists(data_directory):
          logger.error(f"directory not found: {data_directory}")
          return
     
     try:
          engine = create_connection()
     except Exception:
          logger.error("Could not establish database connection. Exiting.")

     start_time = time.time()
     successful_files = 0
     failed_files = 0
    
     logger.info("=" * 50)
     logger.info("Starting CSV ingestion process")
     logger.info(f"Source directory: {data_directory}")
     logger.info("=" * 50)

     try:
          csv_files = [f for f in os.listdir(data_directory) if f.lower().endswith('.csv')]

          if not csv_files:
               logger.warning(f"No CSV files found in directory: {data_directory}")
               return
          
          logger.info(f"Found {len(csv_files)} CSV file to process")

          for file in csv_files:
               file_path = os.path.join(data_directory, file)
               table_name = file[:-4]

               try:
                    logger.info(f"Processing file: {file}")

                    # read csv file.
                    df = pd.read_csv(file_path)
                    logger.info(f"Loaded CSV with shape: {df.shape}")
                    logger.info(f"Columns: {list(df.columns)}")

                    # Check for empty dataframe
                    if df.empty:
                         logger.warning(f"File {file} is empty. Skipping.")
                         continue
                    
                    # Ingest into Database
                    if ingest_db(df, table_name, engine):
                         successful_files += 1
                         logger.info(f"Successfully processed: {file}")
                    else:
                         failed_files += 1
                         logger.error(f"Failed to process: {file}")
                
               except pd.errors.EmptyDataError:
                    logger.error(f"File {file} is empty or corrupted")
                    failed_files += 1
               except pd.errors.ParserError as e:
                    logger.error(f"Error parsing file {file}: {e}")
                    failed_files += 1
               except Exception as e:
                    logger.error(f"Unexpected error processing file {file}: {e}")
                    failed_files += 1
                
     except Exception as e:
          logger.error(f"Error accessing directory {data_directory}: {e}")
          return
     
     finally:
          # Clean up
          if 'engine' in locals():
               engine.dispose()
               logger.info("Database connection closed")
     
     # Calculate and log summary
     end_time = time.time()
     total_time = (end_time - start_time)/60

     logger.info("="*50)
     logger.info("INGESTION COMPLETE")
     logger.info(f"Total files processed: {successful_files + failed_files}")
     logger.info(f"Successful: {successful_files}")
     logger.info(f"Failed: {failed_files}")
     logger.info(f"Total time taken: {total_time:.2f} minutes")
     logger.info("=" * 50)


def main():
    """Main execution function"""
    try:
        load_raw_data()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {e}")
        raise

if __name__ == '__main__':
    main()

