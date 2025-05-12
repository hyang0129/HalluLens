"""
Activation Logging Package for LLM Hallucination Analysis

This package provides tools to run a vLLM-compatible server that logs
neural activations when an LLM is generating responses, particularly useful
for studying hallucination patterns.

Main components:
- activations_logger: Core LMDB-based logger for storing activations
- server: OpenAI API-compatible server with activation logging
"""

__version__ = "0.1.0" 