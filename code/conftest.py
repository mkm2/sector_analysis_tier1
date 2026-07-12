"""Make the qca_fragmentation / qca_analytics packages importable under pytest."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
