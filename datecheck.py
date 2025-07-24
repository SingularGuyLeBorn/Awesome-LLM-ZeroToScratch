import sys

try:
    import trl
    print(f"trl version: {trl.__version__}")
except ImportError:
    print("trl is not installed.")
    sys.exit(1)
except AttributeError:
    print("trl is installed, but version information (__version__) is not available.")
    print("It might be an older version or installed incorrectly.")
    sys.exit(1)

# Optional: You can add more checks here if needed, e.g., checking specific modules
try:
    from trl import PPOConfig
    print("Successfully imported PPOConfig from trl.")
except ImportError:
    print("Could not import PPOConfig from trl. This might indicate an older or incomplete installation.")
    sys.exit(1)