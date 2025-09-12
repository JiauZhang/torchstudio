from PIL import Image, ImageFilter

def blur(image: Image, radius: int = 48):
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

def blur_rect(image: Image, rect: tuple, radius: int = 48):
    x, y, w, h = rect
    left, top, right, bottom = x, y, x + w, y + h
    roi = image.crop((left, top, right, bottom))
    blurred_roi = blur(roi, radius=radius)
    image.paste(blurred_roi, (left, top, right, bottom))
