"""Aegis: real-time financial-crime detection.

Layers, from the bottom up:

    aegis._core      C++20 engine (features, MLP inference, rules, SHA-256, audit chain)
    aegis.datagen    synthetic labelled payments with five fraud typologies
    aegis.model      NumPy MLP trained from scratch, exported to the C++ format
    aegis.train      end-to-end training + evaluation + model signing
    aegis.engine     scoring facade used by the API (explanations, audit records)
    aegis.agent      Claude-powered investigator with prompt-injection defences
    aegis.service    FastAPI server + live dashboard
"""

__version__ = "1.0.0"
