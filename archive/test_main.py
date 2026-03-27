import unittest

def main():
    # Discover all tests in the "tests" folder
    loader = unittest.defaultTestLoader
    suite = loader.discover(start_dir="tests/heat_discretised_model", pattern="test*.py")

    # Create runner (this is the "runner")
    runner = unittest.TextTestRunner(
        verbosity=2,   # detailed output
        failfast=False,  # stop on first failure if True
        buffer=False     # hide print output unless failure if True
    )

    # Run tests
    result = runner.run(suite)

    # Optional: return exit code (useful for CI / scripts)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
