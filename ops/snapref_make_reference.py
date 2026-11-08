"""Create a local test reference picture and a no-op variant, so the SnapRef paths have
something concrete to read. Draws a simple flat-shaded character-ish bust on white — enough
to be a *reference image* (the adapter only needs real pixels; quality is irrelevant here).
"""
import os

from PIL import Image, ImageDraw

OUT = os.environ.get("SNAPREF_REF_DIR", "E:/Workspace/colab/snapref_local/content/anima_snapref")


def draw(path, hair, coat, iris):
  img = Image.new("RGB", (768, 1024), (255, 255, 255))
  d = ImageDraw.Draw(img)
  d.ellipse((274, 980, 494, 1060), fill=(235, 235, 235))          # shoulders shadow
  d.polygon([(240, 1024), (528, 1024), (492, 800), (276, 800)], fill=coat)   # torso
  d.ellipse((300, 300, 468, 520), fill=(247, 219, 199))           # face
  d.polygon([(286, 316), (482, 316), (470, 250), (298, 250)], fill=hair)      # fringe
  d.ellipse((284, 250, 484, 470), outline=hair, width=26)         # hair silhouette
  d.polygon([(284, 400), (300, 560), (330, 430)], fill=hair)      # side lock L
  d.polygon([(484, 400), (468, 560), (438, 430)], fill=hair)      # side lock R
  d.ellipse((336, 392, 372, 428), fill=(255, 255, 255))           # eye L
  d.ellipse((396, 392, 432, 428), fill=(255, 255, 255))           # eye R
  d.ellipse((348, 402, 364, 418), fill=iris)
  d.ellipse((408, 402, 424, 418), fill=iris)
  d.arc((368, 440, 400, 468), 20, 160, fill=(190, 120, 120), width=4)       # mouth
  img.save(path, "PNG")
  return path


if __name__ == "__main__":
  os.makedirs(OUT, exist_ok=True)
  a = draw(os.path.join(OUT, "ref_character.png"), (40, 32, 30), (176, 46, 58), (58, 92, 156))
  b = draw(os.path.join(OUT, "ref_other.png"), (222, 210, 170), (46, 90, 120), (90, 140, 70))
  print("wrote", a)
  print("wrote", b)
