from PIL import Image
from panel_extractor.panel_extractor import PanelExtractor
import os

IMAGE_DIR = r"panel_extractor\images"
OUT_DIR = "debug_panels"

def main():
    if not os.path.isdir(IMAGE_DIR):
        print("❌ Directory not found:", IMAGE_DIR)
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    # ❗ НИКАКИХ лишних аргументов
    extractor = PanelExtractor(debug=True)

    images = [
        f for f in sorted(os.listdir(IMAGE_DIR))
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]

    if not images:
        print("❌ No images found in", IMAGE_DIR)
        return

    for img_name in images:
        img_path = os.path.join(IMAGE_DIR, img_name)
        print(f"\n▶ Processing {img_name}")

        img = Image.open(img_path).convert("RGB")
        panels = extractor.extract_panels(img)

        print(f"  Found {len(panels)} panels")

        base = os.path.splitext(img_name)[0]
        page_dir = os.path.join(OUT_DIR, base)
        os.makedirs(page_dir, exist_ok=True)

        for i, p in enumerate(panels):
            p.save(os.path.join(page_dir, f"panel_{i:02d}.png"))

        print(f"  Saved to {page_dir}")

    print("\n✅ DONE")

if __name__ == "__main__":
    main()
