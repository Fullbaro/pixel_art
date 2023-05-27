from pixel_art import PixelArt
from pdf import PDF

PDF(
    PixelArt(
        path = "OG.png", # PNG format preffered
        scale = 11,
        contrast=1.1,
        color_variance = 30,
        canvas_w= 500,
        canvas_h= 500
    ).start()
).start()