# test_import.py
print("Testing imports...")

try:
    from cleaning import DataCleaner
    print("✓ DataCleaner imported successfully from cleaning")
except Exception as e:
    print(f"✗ Error importing from cleaning: {e}")

try:
    from derived_kpi import KPIGenerator
    print("✓ KPIGenerator imported successfully from derived_kpi")
except Exception as e:
    print(f"✗ Error importing from derived_kpi: {e}")

try:
    from eda import EDAnalyzer
    print("✓ EDAnalyzer imported successfully from eda")
except Exception as e:
    print(f"✗ Error importing from eda: {e}")

try:
    from hypothesis import HypothesisTester
    print("✓ HypothesisTester imported successfully from hypothesis")
except Exception as e:
    print(f"✗ Error importing from hypothesis: {e}")

try:
    from performance_analysis import PerformanceAnalyzer
    print("✓ PerformanceAnalyzer imported successfully from performance_analysis")
except Exception as e:
    print(f"✗ Error importing from performance_analysis: {e}")

try:
    from predictive_modeling import PredictiveModeler
    print("✓ PredictiveModeler imported successfully from predictive_modeling")
except Exception as e:
    print(f"✗ Error importing from predictive_modeling: {e}")

try:
    from recommendations import RecommendationEngine
    print("✓ RecommendationEngine imported successfully from recommendations")
except Exception as e:
    print(f"✗ Error importing from recommendations: {e}")

print("\nAll imports tested!")