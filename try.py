import joblib

print("Loading preprocessor...")

try:
    preprocessor = joblib.load("preprocessor.pkl")
    print("✅ Preprocessor loaded successfully")
    print("Type:", type(preprocessor))
except Exception as e:
    print("❌ Failed to load preprocessor:", e)
    exit()

# Inspect attributes
print("\nAvailable attributes:", dir(preprocessor))

# Check transformers
if hasattr(preprocessor, "transformers"):
    print("\n=== Transformers ===")
    for name, transformer, cols in preprocessor.transformers:
        print(f"- {name}: {transformer} | Columns: {cols}")
else:
    print("\nNo 'transformers' attribute found")

# Check feature names out
if hasattr(preprocessor, "get_feature_names_out"):
    try:
        feature_names = preprocessor.get_feature_names_out()
        print("\n=== Feature Names Out ===")
        print(feature_names)
    except Exception as e:
        print("⚠️ Could not get feature names:", e)
else:
    print("\nNo 'get_feature_names_out' method available")

# Check remainder
if hasattr(preprocessor, "remainder"):
    print("\nRemainder handling:", preprocessor.remainder)
