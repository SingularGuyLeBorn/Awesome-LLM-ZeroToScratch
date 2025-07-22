# FILE: src/evaluation/__init__.py
# Bedrock: This file makes the directory a Python package.

# Mandate of Zero Ambiguity: Expose the primary evaluation function.
# [API IMPORT FIX] Removed 'run_evaluation' as it's no longer a direct export.
# The evaluate_llm.py script is meant to be run directly via command line.
# from .evaluate_llm import run_evaluation # Removed

# END OF FILE: src/evaluation/__init__.py