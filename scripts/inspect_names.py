import cv2
from pathlib import Path

# Let's inspect some of the images in uploads
uploads = sorted(Path("uploads").glob("*.jpeg"))
print("First 10 uploads timestamps / names:")
for u in uploads:
    print(u.name)
