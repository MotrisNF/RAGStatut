from functools import wraps
from typing import Callable, Any
import time


def spell_timer(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def timer(*args: Any, **kwargs: Any) -> Any:
        print(f"Casting {func.__name__}")
        initial_time: float = time.time()
        result = func(*args, **kwargs)
        total_time = time.time() - initial_time
        print(f"Spell completed in {total_time:.3f} seconds")
        return result

    return timer


def power_validator(min_power: int) -> Callable[..., Any]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            power = kwargs.get('power', args[-1] if args else 0)
            if power >= min_power:
                return func(*args, **kwargs)
            return "Insufficient power for this spell"
        return wrapper
    return decorator


def retry_spell(max_attempts: int) -> Callable[..., Any]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    print(
                        f"Spell failed, retrying... (attempt {attempt}/"
                        f"{max_attempts})"
                        )
            return f"Spell casting failed after {max_attempts} attempts"
        return wrapper
    return decorator


class MageGuild:
    @staticmethod
    def validate_mage_name(name: str) -> bool:
        counter: int = 0
        for leter in name:
            if leter.isalpha() or leter == " ":
                counter += 1
            else:
                return False
        if counter >= 3:
            return True
        return False

    @power_validator(min_power=10)
    def cast_spell(self, spell_name: str, power: int) -> str:
        return f"Successfully cast {spell_name} with {power} power"


@spell_timer
def fireball() -> str:
    time.sleep(0.1)
    return "Result: Fireball cast!"


def cursed_spell() -> str:
    raise RuntimeError("The spell is cursed")


def make_flaky_spell() -> Callable[[], str]:
    attempts = 0

    def flaky_spell() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("Spell fizzled unexpectedly")
        return "Waaaaaaagh spelled !"

    return flaky_spell


if __name__ == "__main__":
    print("Testing spell timer...")
    fireball()
    print()

    print("Testing retrying spell...")
    protected = retry_spell(3)(cursed_spell)
    print(protected())

    flaky = make_flaky_spell()
    protected_flaky = retry_spell(3)(flaky)
    print(protected_flaky())
    print()

    print("Testing MageGuild...")
    mage = MageGuild()
    print(mage.validate_mage_name("valid"))
    print(mage.validate_mage_name("1234r5"))
    print(mage.cast_spell("Lightning", 15))
    print(mage.cast_spell("Poop", 5))
