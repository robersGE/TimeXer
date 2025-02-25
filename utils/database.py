from sqlalchemy import create_engine
import logging
import os
import yaml 
import urllib


def connectionSQL(data_base: str, secrets_path: str = "secrets.yaml") -> create_engine:
    """
    Establish a connection to the SQL Server database using credentials from secrets.yaml.

    Args:
        data_base (str): The name of the database.
        secrets_path (str): Path to the YAML file containing secrets.

    Returns:
        sqlalchemy.engine.Engine: SQLAlchemy engine for database connection.
    """
    secrets = load_secrets(secrets_path)

    if data_base not in secrets['database_credentials']:
        raise ValueError(f"Database '{data_base}' not found in secrets.")

    credentials = secrets['database_credentials'][data_base]
    username = credentials['username']
    password = credentials['password']
    server = "svsql001\\promart"
    database = data_base

    connection_string = f"mssql+pymssql://{username}:{urllib.parse.quote_plus(password)}@{server}/{database}"

    try:
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        logging.exception("SQL connection could not be established")
        raise e
    
def load_secrets(secrets_path: str = "secrets.yaml") -> dict:
    """
    Load secrets from a YAML file.

    Args:
        secrets_path (str): Path to the YAML file containing secrets.

    Returns:
        dict: Dictionary containing secrets.
    """
    if not os.path.exists(secrets_path):
        raise FileNotFoundError(f"Secrets file '{secrets_path}' not found.")

    with open(secrets_path, 'r') as file:
        secrets = yaml.safe_load(file)
    return secrets

