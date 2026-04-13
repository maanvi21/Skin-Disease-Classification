import os
from PIL import Image
from predict import predict_image

# Find any image in the data folder to test
test_image_path = "image.png"

if not os.path.exists(test_image_path):
    print("image.png not found, trying another image...")
    # Walk to find first image
    for root, dirs, files in os.walk('data'):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                test_image_path = os.path.join(root, file)
                break
        if test_image_path != "image.png":
            break

print(f"Testing with image: {test_image_path}")

try:
    img = Image.open(test_image_path).convert('RGB')
    result = predict_image(img)
    print("Prediction:", result["prediction"])
    print("Confidence:", result["confidence"])
    print("Message:", result["message"])
    print("Heatmap shape:", result["heatmap"].shape)
    print("🎉 Test passed successfully!")
except Exception as e:
    print("❌ Test failed:", str(e))
    import traceback
    traceback.print_exc()
