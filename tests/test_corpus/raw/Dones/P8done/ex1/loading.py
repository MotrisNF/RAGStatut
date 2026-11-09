import sys
from importlib import metadata
from types import ModuleType

np: ModuleType | None
try:
    import numpy
except ImportError:
    np = None
else:
    np = numpy

pd: ModuleType | None
try:
    import pandas  
except ImportError:
    pd = None
else:
    pd = pandas

plt: ModuleType | None
try:
    import matplotlib  # type: ignore [import-untyped]
    matplotlib.use("Agg")
    import matplotlib.pyplot  # type: ignore [import-untyped]
except ImportError:
    plt = None
else:
    plt = matplotlib.pyplot

requests: ModuleType | None
try:
    import requests as requests_module
except ImportError:
    requests = None
else:
    requests = requests_module

DESCRIPTIONS = {
    "pandas": "Data manipulation",
    "numpy": "Numerical computation",
    "requests": "Network access",
    "matplotlib": "Visualization",
}


def check_dependency(name: str, module: object | None) -> bool:
    if module is None:
        print(f"[MISSING] {name} - install with: pip install {name}")
        return False
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        version = "unknown"
    print(f"[OK] {name} ({version}) - {DESCRIPTIONS[name]} ready")
    return True


def check_dependencies() -> bool:
    print("Checking dependencies:")
    pandas_ok = check_dependency("pandas", pd)
    numpy_ok = check_dependency("numpy", np)
    requests_ok = check_dependency("requests", requests)
    matplotlib_ok = check_dependency("matplotlib", plt)
    return pandas_ok and numpy_ok and requests_ok and matplotlib_ok


def print_installation_instructions() -> None:
    print()
    print("Some dependencies are missing. Install them with pip:")
    print("  pip install -r requirements.txt")
    print("or with Poetry:")
    print("  poetry install")


def run_analysis() -> None:
    if np is None or pd is None or plt is None:
        return

    print()
    print("Analyzing Matrix data...")

    size = 1000
    time = np.linspace(0.0, 10.0 * np.pi, size)
    signal = np.sin(time)
    frame = pd.DataFrame({"time": time, "signal": signal})

    print(f"Processing {len(frame)} data points...")
    print("Generating visualization...")

    figure, axes = plt.subplots()
    axes.plot(frame["time"], frame["signal"], color="green")
    axes.set_title("Matrix Data Stream Analysis")
    axes.set_xlabel("Time")
    axes.set_ylabel("Signal value")
    figure.savefig("matrix_analysis.png")
    plt.close(figure)

    print()
    print("Analysis complete!")
    print("Results saved to: matrix_analysis.png")


def is_running_under_poetry() -> bool:
    return "pypoetry" in sys.prefix


def print_dependency_comparison() -> None:
    is_poetry = is_running_under_poetry()
    print()
    print("Dependency management comparison:")
    print("pip    -> requirements.txt, flat install via "
          "'pip install -r requirements.txt'")
    print("Poetry -> pyproject.toml + poetry.lock, locked and "
          "reproducible via 'poetry install'")
    print(f"Active manager detected: {'Poetry' if is_poetry else 'pip'}")


def main() -> None:
    print("LOADING STATUS: Loading programs...")
    print()

    if not check_dependencies():
        print_installation_instructions()
        sys.exit(1)

    run_analysis()
    print_dependency_comparison()


if __name__ == "__main__":
    main()
