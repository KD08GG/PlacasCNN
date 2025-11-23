# run_tests.py
"""
Script para ejecutar todos los tests del proyecto.
"""
import unittest
import sys
from pathlib import Path

def run_all_tests():
    """Ejecuta todos los tests."""
    print("="*70)
    print("EJECUTANDO TESTS - Sistema ALPR")
    print("="*70 + "\n")

    # Descubrir y ejecutar tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')

    if suite.countTestCases() == 0:
        print("⚠ No se encontraron tests en el directorio 'tests/'")
        return False

    print(f"Encontrados {suite.countTestCases()} tests\n")

    # Ejecutar con verbosidad
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Resumen
    print("\n" + "="*70)
    print("RESUMEN")
    print("="*70)
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"Exitosos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Fallidos: {len(result.failures)}")
    print(f"Errores: {len(result.errors)}")
    print("="*70 + "\n")

    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
