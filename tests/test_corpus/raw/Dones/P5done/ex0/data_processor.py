from abc import ABC, abstractmethod
from typing import Any


class DataProcesor(ABC):
    def __init__(self) -> None:
        self.lst_data: list[str] = []
        self.entry: int = 0

    @abstractmethod
    def validate(self, data: Any) -> bool:
        pass

    @abstractmethod
    def ingest(self, data: Any) -> None:
        pass

    def output(self) -> tuple[int, str]:
        if len(self.lst_data) > 0:
            actual: int = self.entry
            self.entry += 1
            return (actual, self.lst_data.pop(0))

        return (-1, "None")


class NumericProcesor(DataProcesor):
    def validate(self, data: Any) -> bool:
        if isinstance(data, int | float) and not isinstance(data, bool):
            return True
        elif isinstance(data, list) and len(data) > 0:
            for num in data:
                if not isinstance(num, int | float) or isinstance(num, bool):
                    return False
            return True
        else:
            return False

    def ingest(self, data: Any) -> None:
        if not self.validate(data):
            raise Exception("Improper numeric data")

        if isinstance(data, int | float):
            self.lst_data.append(str(data))
        else:
            new_tmp_list: list[str] = [
                str(num) for num in data if isinstance(num, int | float)
                ]
            self.lst_data = self.lst_data + new_tmp_list


class TextProcesor(DataProcesor):
    def validate(self, data: Any) -> bool:
        if isinstance(data, str):
            return True
        elif isinstance(data, list) and len(data) > 0:
            for word in data:
                if not isinstance(word, str):
                    return False
            return True
        else:
            return False

    def ingest(self, data: Any) -> None:
        if not self.validate(data):
            raise Exception("Improper text data")

        if isinstance(data, str):
            self.lst_data.append(data)
        else:
            new_tmp_list: list[str] = [
                str(word) for word in data if isinstance(word, str)
                ]

            self.lst_data = self.lst_data + new_tmp_list


class LogProcesor(DataProcesor):
    def validate(self, data: Any) -> bool:
        if isinstance(data, dict) and len(data) > 0:
            return True
        elif isinstance(data, list) and len(data) > 0:
            for dictionary in data:
                if not isinstance(dictionary, dict):
                    return False
            return True
        else:
            return False

    def ingest(self, data: Any) -> None:
        if not self.validate(data):
            raise Exception("Improper log data")

        if isinstance(data, dict) and len(data) > 0:
            for key, value in data.items():
                self.lst_data.append(str(key) + ": " + str(value))
        else:
            for index in data:
                for key, value in index.items():
                    self.lst_data.append(str(key) + ": " + str(value))


def test_numeric() -> None:
    num_procesor = NumericProcesor()
    number: int | str = 42

    print(f"Trying to validate input '{number}': ", end="")
    print(f"{num_procesor.validate(number)}")

    number = "Hello"
    print(f"Trying to validate input '{number}': ", end="")
    print(f"{num_procesor.validate(number)}")

    number = "foo"
    print("Test invalid ingestion of string ", end="")
    print(f"'{number}' without prior validation:")
    try:
        num_procesor.ingest("foo")

    except Exception as e:
        print(f"Got exception: {e}")

    list_num_good: list[Any] = [1.1, 2, 3, 4, 5.5]
    print(f"Procesing data: {list_num_good}")
    try:
        num_procesor.ingest(list_num_good)
    except Exception as e:
        print(f"Got exception: {e}")

    values: int = 3
    print(f"Extracting {values} values...")
    for _ in range(0, values):
        try:
            tuple_test: tuple[int, str] = num_procesor.output()
        except Exception as e:
            print(f"Got exception: {e}")
        print(f"Numeric value {tuple_test[0]}: {tuple_test[1]}")


def test_str() -> None:
    text_procesor = TextProcesor()
    word: int | str = 42

    print(f"Trying to validate input '{word}': ", end="")
    print(f"{text_procesor.validate(word)}")

    word = "Hello"
    print(f"Trying to validate input '{word}': ", end="")
    print(f"{text_procesor.validate(word)}")

    word = 42
    print("Test invalid ingestion of string ", end="")
    print(f"'{word}' without prior validation:")
    try:
        text_procesor.ingest(42)

    except Exception as e:
        print(f"Got exception: {e}")

    list_words_good: list[str] = ["Hello", "Nexsus", "World"]
    print(f"Procesing data: {list_words_good}")
    text_procesor.ingest(list_words_good)

    values: int = 1
    print(f"Extracting {values} values...")
    for _ in range(0, values):
        tuple_test: tuple[int, str] = text_procesor.output()
        print(f"Text value {tuple_test[0]}: {tuple_test[1]}")


def test_log() -> None:
    log_procesor = LogProcesor()
    dictionary: Any = "Hello"

    print(f"Trying to validate input '{dictionary}': ", end="")
    print(f"{log_procesor.validate(dictionary)}")

    dictionary = {"log_level": "NOTICE", "log_mensage": "Conection to server"}
    print(f"Trying to validate input '{dictionary}': ", end="")
    print(f"{log_procesor.validate(dictionary)}")

    dictionary = 42
    print("Test invalid ingestion of string ", end="")
    print(f"'{dictionary}' without prior validation:")
    try:
        log_procesor.ingest(42)

    except Exception as e:
        print(f"Got exception: {e}")

    dictionary1: dict[str, str] = {"NOTICE": "Connection to server"}
    dictionary2: dict[str, str] = {"ERROR": "Unauthorized access!!"}
    dictionary3: dict[str, str] = {"DEBBUG": "Cleaned the cache data"}
    list_dict_good: list[Any] = [dictionary1, dictionary2, dictionary3]
    print(f"Procesing data: {list_dict_good}")
    try:
        log_procesor.ingest(list_dict_good)
    except Exception as e:
        print(e)

    values: int = 2
    print(f"Extracting {values} values...")
    for _ in range(0, values):
        tuple_test: tuple[int, str] = log_procesor.output()
        print(f"Log entry {tuple_test[0]}: {tuple_test[1]}")


def main() -> None:
    print("Testing Numeric Processor...")
    test_numeric()
    print()
    print("Testing Text Processor...")
    test_str()
    print()
    print("Testing Log Processor...")
    test_log()


if __name__ == "__main__":
    print("=== Code Nexus - Data Processor ===\n")
    main()
