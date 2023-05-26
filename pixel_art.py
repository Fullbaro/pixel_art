import os
import cv2
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.cluster import KMeans


class PixelArt:

    def __init__(self, path, scale, color_variance, contrast, canvas_w, canvas_h):
        self.path = path
        self.scale = scale
        self.reduce = color_variance
        self.cont = contrast
        self.grid_w = canvas_w
        self.grid_h = canvas_h
        self.image: np.ndarray
        self.image_width = int
        self.image_height = int
        self.colors: np.ndarray
        self.pixel_size: int
        self.data = {
            "color_variance":  color_variance,
            "contrast":  contrast,
            "canvas_w":  canvas_w,
            "canvas_h":  canvas_h,
            "scale": scale
        }

    def start(self):
        self.delete("*.png")
        self.delete("*.pdf")

        self.load()
        self.resize()
        self.contrast()
        self.reduce_colors()
        self.collect_colors()
        self.sort_colors()
        self.calculate_grid()
        self.generate_grid()
        self.colors_preview()

        print("Your art is ready!")
        return self.data

    def load(self):
        image = cv2.cvtColor(cv2.imread(self.path), cv2.COLOR_BGR2RGB)
        self.image_width = image.shape[1]
        self.image_height = image.shape[0]
        self.image = image
        self.save("original", self.image)
        print("Image loaded")

    def save(self, name, img, scale=False):
        if scale:
            # Needs to scaled up to look good in PDF
            new_width = 800
            new_height = int(img.shape[0] * (new_width / img.shape[1]))
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(f"/home/deni/Cloud/Laptop/Projects/Python/pixel_art/V2/assets/{name}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def delete(self, file):
        folder_path = "/home/deni/Cloud/Laptop/Projects/Python/pixel_art/V2/assets"
        files = glob.glob(os.path.join(folder_path, file))
        for file_path in files:
            os.remove(file_path)
        print("Previus data deleted")

    def show(self, title=""): # Display image
        plt.imshow(self.image)
        plt.title(title)
        plt.show()

    def resize(self): # Resize image with percentage. Keeps the aspect reatio
        self.data["original_shape"] = f"{self.image_width}x{self.image_height}"

        self.image_width = int(self.image_width * self.scale / 100)
        self.image_height = int(self.image_height * self.scale / 100)
        self.image = cv2.resize(self.image, (self.image_width, self.image_height))
        self.save("resized", self.image, True)

        self.data["resized_shape"] = f"{self.image_width}x{self.image_height}"
        print(f"Image resized")

    def contrast(self):
        self.image = cv2.convertScaleAbs(self.image, alpha=self.cont)
        self.save("contrasted", self.image, True)
        print("Contrast set")

    def reduce_colors(self): # Clasters the pixels to reduce colors variance
        self.data["original_color_count"] = len(np.unique(self.image.reshape(-1, 3), axis=0))

        img = np.array(self.image, dtype=np.float64) / 255
        image_array = np.reshape(img, (self.image_width * self.image_height, 3))
        image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
        kmeans = KMeans(n_clusters=self.reduce, n_init="auto", random_state=0).fit(image_array_sample)
        labels = kmeans.predict(image_array)

        reduced = kmeans.cluster_centers_[labels].reshape(self.image_height, self.image_width, -1) * 255
        self.image = reduced.astype(np.uint8)
        self.save("reduced", self.image, True)
        print("Color count reduced")

    def collect_colors(self): # Collect new unique colors and add a 4. value. % = what percentage of the total image is represented by a given pixel
        self.colors = np.unique(self.image.reshape(-1, self.image.shape[2]), axis=0) # Save new unique colors

        extended_colors = []
        for color in self.colors:
            color_pixels = np.count_nonzero(np.all(self.image == color, axis=2))
            color_percentage = round((color_pixels / (self.image_width * self.image_height)) * 100, 2)
            extended_color = [round(c, 8) for c in color] + [color_percentage]
            extended_colors.append(extended_color)
        self.colors = extended_colors

        print("Unique colors collected")

    def sort_colors(self):
        hsl_colors = [(*color[:-1], color[-1]) for color in self.colors]
        sorted_hsl_colors = sorted(hsl_colors, key=lambda x: x[:-1])
        self.colors = [(*color[:-1], color[-1]) for color in sorted_hsl_colors]

        self.data["colors"] = self.colors
        print("Colors sorted")

    def calculate_grid(self):
        pixel_w = self.grid_w / self.image_width
        pixel_h = self.grid_h / self.image_height
        self.pixel_size = math.floor(pixel_w if pixel_w < pixel_h else pixel_h)

        fit = int(min(self.grid_w / self.pixel_size, self.grid_h / self.pixel_size))
        side_border = self.grid_w - fit * self.pixel_size
        top_border = self.grid_h - fit * self.pixel_size

        self.data["pixel_size"] = f"{self.pixel_size}x{self.pixel_size}"
        self.data["top_border"] = top_border / 2
        self.data["side_border"] = side_border / 2
        print("Grid dimensions calculated")

    def generate_grid(self):
        cell_size = 30
        grid_width = self.image_width * cell_size
        grid_height = self.image_height * cell_size
        grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

        # Draw grid
        for x in range(0, grid_width, cell_size):
            cv2.line(grid, (x, 0), (x, grid_height), (0, 0, 0), 1)
        for y in range(0, grid_height, cell_size):
            cv2.line(grid, (0, y), (grid_width, y), (0, 0, 0), 1)

        # Place munbers on grid
        for count_x, x in enumerate(range(cell_size // 2, grid_width, cell_size)):
            for count_y, y in enumerate(range(cell_size // 2, grid_height, cell_size)):
                number = str(self.get_color_index(self.image[count_y][count_x])).zfill(len(str(len(self.colors))))
                font_scale = cell_size / (25 * len(str(number)))
                number = str(int(number))
                font_thickness = max(1, int(font_scale))
                (text_width, text_height), _ = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                text_org = ((x - text_width // 2), (y + text_height // 2))
                cv2.putText(grid, number, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        self.save("grid", grid)
        print("Grid generated")

    def get_color_index(self, color):
        return next((i for i, c in enumerate(self.colors) if np.array_equal(c[:-1], color)), -1)

    def colors_preview(self):
        for index, color in enumerate(self.colors):
            image = np.full((100, 100, 3), color[:-1], dtype=np.uint8)

            self.save(f"color_{index}", image)
            self.save(f"placement_{index}", self.color_placements(color[:-1]), True)
        print("Color placements saved")

    def color_placements(self, color):
        img = self.image.copy()
        target_color = np.array(color)
        replacement_color = np.array(color)
        mask = np.all(img == target_color, axis=-1)
        img[:, :] = 255
        img[mask] = replacement_color

        return img