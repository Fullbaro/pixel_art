from pixel_art import PixelArt
from pdf import PDF

PDF(
    PixelArt(
        path = "OG.png", # PNG format preffered
        scale = 11,
        contrast=1.1,
        color_variance = 30,
        canvas_w= 490,
        canvas_h= 490
    ).start()
).start()