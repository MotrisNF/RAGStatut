#!/usr/bin/env python3

def secure_archive(
        file: str,
        action: str = "r",
        new: str = ""
        ) -> tuple[bool, str]:
    try:
        if action.lower() == "r" or action.lower() == "read":
            with open(file, "rt") as archive:
                string: str = archive.read()
        elif action.lower() == "w" or action.lower() == "write":
            if new != "":
                with open(file, "rt") as archive:
                    with open(new, "wt") as new_archive:
                        new_archive.write(archive.read())
                        string = "Content successfully written to file"
        else:
            raise Exception("Unknown action")

    except Exception as e:
        return (False, str(e))
    return (True, string)


if __name__ == "__main__":
    result: tuple[bool, str] = secure_archive("test2.txt", "w", "test4.txt")
    print(result)
