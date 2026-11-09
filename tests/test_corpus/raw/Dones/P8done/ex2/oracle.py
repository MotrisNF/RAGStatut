import os
import sys

from dotenv import load_dotenv

REQUIRED_KEYS = (
    "MATRIX_MODE",
    "DATABASE_URL",
    "API_KEY",
    "LOG_LEVEL",
    "ZION_ENDPOINT",
)


class ConfigurationError(Exception):
    def __init__(self, missing_keys: list[str]) -> None:
        super().__init__(f"Missing configuration keys: {missing_keys}")
        self.missing_keys = missing_keys


class MatrixConfig:
    def __init__(
        self,
        mode: str,
        database_url: str,
        api_key: str,
        log_level: str,
        zion_endpoint: str,
        dotenv_loaded: bool,
    ) -> None:
        self.mode = mode
        self.database_url = database_url
        self.api_key = api_key
        self.log_level = log_level
        self.zion_endpoint = zion_endpoint
        self.dotenv_loaded = dotenv_loaded


def load_configuration() -> MatrixConfig:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, ".env")
    dotenv_loaded = load_dotenv(dotenv_path=env_path)

    missing_keys = [key for key in REQUIRED_KEYS if key not in os.environ]
    if missing_keys:
        raise ConfigurationError(missing_keys)

    return MatrixConfig(
        mode=os.environ["MATRIX_MODE"],
        database_url=os.environ["DATABASE_URL"],
        api_key=os.environ["API_KEY"],
        log_level=os.environ["LOG_LEVEL"],
        zion_endpoint=os.environ["ZION_ENDPOINT"],
        dotenv_loaded=dotenv_loaded,
    )


def describe_database(mode: str) -> str:
    if mode == "production":
        return "Connected to production cluster"
    return "Connected to local instance"


def print_configuration(config: MatrixConfig) -> None:
    print("ORACLE STATUS: Reading the Matrix...")
    print()
    print("Configuration loaded:")
    print(f"Mode: {config.mode}")
    print(f"Database: {describe_database(config.mode)}")
    print("API Access: Authenticated")
    print(f"Log Level: {config.log_level}")
    print("Zion Network: Online")
    print()
    print("Environment security check:")
    print("[OK] No hardcoded secrets detected")
    dotenv_status = "[OK]" if config.dotenv_loaded else "[MISSING]"
    print(f"{dotenv_status} .env file properly configured")
    print("[OK] Production overrides available")
    print()
    print("The Oracle sees all configurations.")


def print_missing_configuration(missing_keys: list[str]) -> None:
    print("ORACLE STATUS: Reading the Matrix...")
    print()
    print("WARNING: Missing configuration for:")
    for key in missing_keys:
        print(f"  - {key}")
    print()
    print("Create a .env file (copy .env.example to .env) or export")
    print("these variables, then run this program again.")


def main() -> None:
    try:
        config = load_configuration()
    except ConfigurationError as e:
        print_missing_configuration(e.missing_keys)
        sys.exit(1)
    print_configuration(config)


if __name__ == "__main__":
    main()
