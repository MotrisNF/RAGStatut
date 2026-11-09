from abc import ABC, abstractmethod
import typing


class DataProcessor(ABC):
    def __init__(self) -> None:
        self.lst_data: list[str] = []
        self.entry: int = 0
        self.consumed: int = 0

    @abstractmethod
    def validate(self, data: typing.Any) -> bool:
        pass

    @abstractmethod
    def ingest(self, data: typing.Any) -> None:
        pass

    def output(self) -> tuple[int, str]:
        if len(self.lst_data) > 0:
            actual: int = self.entry
            self.entry += 1
            return (actual, self.lst_data.pop(0))

        return (-1, "None")


class NumericProcesor(DataProcessor):
    def validate(self, data: typing.Any) -> bool:
        if isinstance(data, int | float) and not isinstance(data, bool):
            return True
        elif isinstance(data, list) and len(data) > 0:
            for num in data:
                if not isinstance(num, int | float) or isinstance(num, bool):
                    return False
            return True
        else:
            return False

    def ingest(self, data: typing.Any) -> None:
        if not self.validate(data):
            raise Exception("Improper numeric data")

        if isinstance(data, int | float):
            self.lst_data.append(str(data))
            self.consumed += 1
        else:
            new_tmp_list: list[str] = [
                str(num) for num in data if isinstance(num, int | float)
                ]
            self.consumed += len(new_tmp_list)
            self.lst_data = self.lst_data + new_tmp_list

    def __str__(self) -> str:
        return "NumericProcesor"


class TextProcesor(DataProcessor):
    def validate(self, data: typing.Any) -> bool:
        if isinstance(data, str):
            return True
        elif isinstance(data, list) and len(data) > 0:
            for word in data:
                if not isinstance(word, str):
                    return False
            return True
        else:
            return False

    def ingest(self, data: typing.Any) -> None:
        if not self.validate(data):
            raise Exception("Improper text data")

        if isinstance(data, str):
            self.lst_data.append(data)
            self.consumed += 1
        else:
            new_tmp_list: list[str] = [
                str(word) for word in data if isinstance(word, str)
                ]

            self.consumed += len(new_tmp_list)
            self.lst_data = self.lst_data + new_tmp_list

    def __str__(self) -> str:
        return "TextProcesor"


class LogProcesor(DataProcessor):
    def validate(self, data: typing.Any) -> bool:
        if isinstance(data, dict) and len(data) > 0:
            return True
        elif isinstance(data, list) and len(data) > 0:
            for dictionary in data:
                if not isinstance(dictionary, dict):
                    return False
            return True
        else:
            return False

    def ingest(self, data: typing.Any) -> None:
        if not self.validate(data):
            raise Exception("Improper log data")

        if isinstance(data, dict) and len(data) > 0:
            self.consumed += 1
            self.lst_data.append(": ".join(str(v) for v in data.values()))
        else:
            for dictionary in data:
                self.consumed += 1
                self.lst_data.append(
                    ": ".join(str(v) for v in dictionary.values())
                )

    def __str__(self) -> str:
        return "LogProcesor"


class CSV:
    def process_output(self, data: list[tuple[int, str]]) -> None:
        print("CSV output")
        for value in data:
            if value is not data[len(data) - 1]:
                print(value[1], end=",")
            else:
                print(value[1])


class JSON:
    def process_output(self, data: list[tuple[int, str]]) -> None:
        print("JSON Output:")
        pairs: str = ", ".join(
            f'"item_{rank}": "{value}"' for rank, value in data
            )
        print("{" + pairs + "}")


class ExportPlugin(typing.Protocol):
    def process_output(self, data: list[tuple[int, str]]) -> None:
        pass


class DataStream:
    def __init__(self) -> None:
        self.constructors: list[DataProcessor] = []

    def register_procesor(self, procesor: DataProcessor) -> None:
        for procesors in self.constructors:
            if type(procesor) is type(procesors):
                print(f"The {procesor} alreadys exists")
                return

        self.constructors.append(procesor)
        print(f"Added a {procesor}")

    def process_stream(self, stream: list[typing.Any]) -> None:
        for element in stream:
            processed = False
            for processor in self.constructors:
                if processor.validate(element):
                    processor.ingest(element)
                    processed = True
                    break
            if not processed:
                print("DataStream Error", end=" ")
                print(f"- Can't process element in stream: {element}")

    def print_procesor_stats(self) -> None:
        if len(self.constructors) == 0:
            print("No processor found, no data")
            return
        for procesor in self.constructors:
            print(f"{procesor}: total {procesor.consumed}", end=" ")
            print("item processed,", end=" ")
            print(f"remaining {len(procesor.lst_data)} on procesor")

    def output_pipeline(self, nb: int, plugin: ExportPlugin) -> None:
        for processor in self.constructors:
            list_tuple: list[tuple[int, str]] = []
            for _ in range(0, nb):
                temp: tuple[int, str] = processor.output()
                if temp[0] != -1:
                    list_tuple.append(temp)
            plugin.process_output(list_tuple)


def main() -> None:
    print("=== Code Nexus - Data Pipeline ===\n")

    print("Initialize Data Stream...")
    stream: DataStream = DataStream()

    print("== DataStream statistics ==")
    stream.print_procesor_stats()

    print("\nRegistering Processors\n")
    stream.register_procesor(NumericProcesor())
    stream.register_procesor(TextProcesor())
    stream.register_procesor(LogProcesor())

    test_list: list[typing.Any] = [
        "Hello world",
        [3.14, -1, 2.71],
        [{"log_level": "WARNING",
          "log_mensage": "Telnet access! Use ssh instead"},
            {"log_level": "INFO",
             "log message": "User wil is connected"}],
        42,
        ["Hi", "five"],
    ]
    print("Send first batch of data on stream:")
    print(test_list)
    stream.process_stream(test_list)

    print("== DataStream statistics ==")
    stream.print_procesor_stats()

    print("Send 3 processed data from each processor to a CSV pluguin")
    stream.output_pipeline(3, CSV())
    print("\n== DataStream statistics ==")
    stream.print_procesor_stats()

    test_list2: list[typing.Any] = [
        21,
        ["I love AI",
         "LLMs are wonderful",
         "Stay healthy"
         ],
        [{"log_level": "ERROR", "log_message": "500 server crash"},
         {"log_level": "NOTICE",
          "log_message": "Certificate exired in 10 days"}],
        [32, 42, 64, 84, 128, 168],
        "World hello"
    ]
    print(f"Send another bach of data: {test_list2}\n")
    stream.process_stream(test_list2)
    print("== DataStream statistics ==")
    stream.print_procesor_stats()
    print("Send 5 processed data from each processor to a JSON pluguin")
    stream.output_pipeline(5, JSON())
    print("\n== DataStream statistics ==")
    stream.print_procesor_stats()


if __name__ == "__main__":
    main()
