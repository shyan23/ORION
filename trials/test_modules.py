#!/usr/bin/env python3
"""
Quick test script to verify modular imports work correctly
"""

def test_imports():
    print("🧪 Testing modular imports...")

    try:
        from modules.models import Point, RectangularROI, CircularROI
        print("✅ Models module imported successfully")

        # Test basic functionality
        p1 = Point(0, 0)
        p2 = Point(10, 10)
        rect = RectangularROI(p1, p2)
        print(f"   📐 Rectangle area: {rect.get_area()}")

    except ImportError as e:
        print(f"❌ Models import failed: {e}")
        return False

    try:
        from modules.cache_storage import DiskBasedGlobalContextCache
        print("✅ Cache storage module imported successfully")

    except ImportError as e:
        print(f"❌ Cache storage import failed: {e}")
        return False

    try:
        from modules.utils import calculate_distance
        print("✅ Utils module imported successfully")

        # Test basic functionality
        p1 = Point(0, 0)
        p2 = Point(3, 4)
        distance = calculate_distance(p1, p2)
        print(f"   📏 Distance calculation: {distance}")

    except ImportError as e:
        print(f"❌ Utils import failed: {e}")
        return False

    try:
        from modules.ai_core import LocalMedSAM, LocalROIAnalyzer
        print("✅ AI core module imported successfully")

    except ImportError as e:
        print(f"❌ AI core import failed: {e}")
        return False

    print("🎉 All modular imports successful!")
    return True

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n✅ Modularization is working correctly!")
        print("🚀 Ready for GitHub deployment!")
    else:
        print("\n❌ Some imports failed. Check the module files.")
        exit(1)