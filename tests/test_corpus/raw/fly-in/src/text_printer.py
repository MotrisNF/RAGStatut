"""Terminal output helpers for the letter-by-letter animated prompts."""

import io
import os
import sys
import termios
import time


class Printer:
    """Prints text to the terminal with a typewriter animation."""

    @staticmethod
    def print_by_letter(
                        text: str,
                        speed: float,
                        time_to_sleep: float,
                        color: str = "\033[0m",
                        testing: bool = False
                        ) -> None:
        """Print text one letter at a time, hiding cursor and echo.

        While the animation runs, the terminal's echo is disabled (on
        a real tty) so keystrokes typed by the user don't get printed
        on top of the animated text; any keys pressed during that
        window are discarded once the animation finishes.

        Args:
            text: The text to print letter by letter.
            speed: Seconds to sleep between each printed letter.
            time_to_sleep: Seconds to pause after the full text is
                shown, before returning.
        """
        if testing:
            return
        try:
            fd = sys.stdin.fileno()
            is_tty = os.isatty(fd)
        except (OSError, ValueError, io.UnsupportedOperation):
            # stdin has no real file descriptor at all to wait on or
            # restore afterwards -- e.g. pytest's captured stdin, or
            # any other fully detached stream. There's no terminal to
            # animate for, so print plainly instead of both crashing
            # on it and paying for every letter's sleep.
            print(f"{color}{text}\033[0m")
            return
        old_settings = None
        if is_tty:
            old_settings = termios.tcgetattr(fd)
            no_echo_settings = termios.tcgetattr(fd)
            no_echo_settings[3] = no_echo_settings[3] & ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSANOW, no_echo_settings)
        try:
            temp_text: str = ""
            print("\033[?25l", end="")
            for leter in text:
                temp_text = temp_text + leter
                print(color + temp_text, end="\r")
                time.sleep(speed)
            print(temp_text, end="")
            print("\033[?25h\033[0m", end="")
            time.sleep(time_to_sleep)

            print()
        finally:
            if is_tty:
                assert old_settings is not None
                termios.tcflush(fd, termios.TCIFLUSH)
                termios.tcsetattr(fd, termios.TCSANOW, old_settings)

    @staticmethod
    def erase_line(
                    times: int = 0,
                    testing: bool = False
                   ) -> None:
        """Erase the given number of previously printed lines.

        Args:
            times: How many lines, counting upward from the cursor,
                to clear.
        """
        if testing:
            return
        for _ in range(0, times):
            print(end="\033[2K\033[A\r")
            Printer.print_by_letter(
                "                                                           ",
                0.015,
                0
            )
            print("\033[2K\r\033[A", end="")
