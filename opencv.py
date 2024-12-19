import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk

class ImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Filters")

        self.image_path = ""
        self.original_image = None
        self.processed_image = None
        self.dynamic_kernel = None

        self.create_widgets()

    def create_widgets(self):
        load_button = tk.Button(self.root, text="Load Image", command=self.load_image)
        load_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.canvas_original = tk.Canvas(self.root)
        self.canvas_original.grid(row=1, column=0, padx=5, pady=5, columnspan=2)  # Set columnspan to 2

        self.canvas_processed = tk.Canvas(self.root)
        self.canvas_processed.grid(row=1, column=2, padx=5, pady=5, columnspan=2)  # Set columnspan to 2

        filter_buttons = [
            ("Point Detection", "point_detection"),
            ("Horizontal Line Detection", "horizontal_line"),
            ("Vertical Line Detection", "vertical_line"),
            ("Diagonal Line Detection (+45 degrees)", "diagonal_line_45"),
            ("Diagonal Line Detection (-45 degrees)", "diagonal_line_minus_45"),
            ("Prewitt Vertical", "prewitt_vertical"),
            ("Prewitt Horizontal", "prewitt_horizontal"),
            ("Prewitt Diagonal (+45 degrees)", "prewitt_diagonal_45"),
            ("Prewitt Diagonal (-45 degrees)", "prewitt_diagonal_minus_45"),
            ("Sobel Vertical", "sobel_vertical"),
            ("Sobel Horizontal", "sobel_horizontal"),
            ("Sobel Diagonal (+45 degrees)", "sobel_diagonal_45"),
            ("Sobel Diagonal (-45 degrees)", "sobel_diagonal_minus_45"),
            ("Roberts Filter", "roberts"),
            ("2nd Derivative Filter", "second_derivative"),
            ("Laplacian Filter", "laplacian"),
            ("Laplacian of Gaussian Filter", "laplacian_of_gaussian"),
        ]

        row_offset = 2
        column_offset = 0

        for button_text, command in filter_buttons:
            if callable(command):
                filter_button = tk.Button(self.root, text=button_text, command=command)
            else:
                filter_button = tk.Button(self.root, text=button_text, command=lambda ft=command: self.apply_filter(ft))

            filter_button.grid(row=row_offset, column=column_offset, padx=5, pady=5)
            column_offset += 1
            if column_offset == 4:
                column_offset = 0
                row_offset += 1

        user_filter_buttons = [
            ("Apply User-defined Filter", self.apply_user_defined_filter),
            ("Define Filter", self.define_filter),
            ("Reset User Filter", self.reset_user_filter),
        ]

        user_filter_frame = tk.Frame(self.root)
        user_filter_frame.grid(row=row_offset, column=0, columnspan=4, pady=5)
        for button_text, command in user_filter_buttons:
            user_filter_button = tk.Button(user_filter_frame, text=button_text, command=command)
            user_filter_button.pack(side=tk.LEFT, padx=5)

        save_button = tk.Button(self.root, text="Save Image", command=self.save_image)
        save_button.grid(row=row_offset + 1, column=0, columnspan=4, pady=5)

    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.original_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            self.original_image = cv2.resize(self.original_image, (400, 307))
            self.display_image(self.original_image, self.canvas_original)

    def display_image(self, image, canvas):
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image=image)
            canvas.config(width=photo.width(), height=photo.height())
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.image = photo

    def apply_user_defined_filter(self):
        if self.original_image is not None and self.dynamic_kernel is not None:
            try:
                print("User Defined Filter Applied:")
                print(self.dynamic_kernel)
                self.processed_image = cv2.filter2D(self.original_image, -1, self.dynamic_kernel)
                self.display_image(self.processed_image, self.canvas_processed)
            except Exception as e:
                messagebox.showerror("Error", f"Error applying user-defined filter: {str(e)}")

    def apply_filter(self, filter_type):
        if self.original_image is not None:
            if filter_type == "point_detection":
                kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
            elif filter_type == "horizontal_line":
                kernel = np.array([[-1, -1, -1],
                                   [ 2,  2,   2],
                                   [-1, -1, -1]])
            elif filter_type == "vertical_line":
                kernel = np.array([[-1, 2, -1],
                                   [-1, 2, -1],
                                   [-1, 2, -1]])
            elif filter_type == "diagonal_line_45":
                kernel = np.array([[-1, -1,  2],
                                   [-1,  2, -1],
                                   [ 2, -1, -1]])
            elif filter_type == "diagonal_line_minus_45":
                kernel = np.array([[ 2, -1, -1],
                                   [-1,  2, -1],
                                   [-1, -1,  2]])
            elif filter_type == "prewitt_vertical":
                kernel = np.array([[-1, 0, 1],
                                   [-1, 0, 1],
                                   [-1, 0, 1]])
            elif filter_type == "prewitt_horizontal":
                kernel = np.array([[-1, -1, -1],
                                   [ 0,  0,  0],
                                   [ 1,  1,  1]])
            elif filter_type == "prewitt_diagonal_45":
                kernel = np.array([[ 0, -1, -1],
                                   [ 1,  0, -1],
                                   [ 1,  1,  0]])
            elif filter_type == "prewitt_diagonal_minus_45":
                kernel = np.array([[-1, -1, 0],
                                   [-1,  0, 1],
                                   [ 0,  1,  1]])
            elif filter_type == "sobel_vertical":
                kernel = np.array([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]])
            elif filter_type == "sobel_horizontal":
                kernel = np.array([[-1, -2, -1],
                                   [ 0,  0,  0],
                                   [ 1,  2,  1]])
            elif filter_type == "sobel_diagonal_45":
                kernel = np.array([[ 0, 1, 2],
                                   [-1, 0, 1],
                                   [-2, -1, 0]])
            elif filter_type == "sobel_diagonal_minus_45":
                kernel = np.array([[-2, -1, 0],
                                   [-1, 0, 1],
                                   [0, 1, 2]])
            elif filter_type == "roberts":
                kernel = np.array([[1, 0],
                                   [0, -1]])
            elif filter_type == "second_derivative":
                kernel = np.array([[1, -2, 1],
                                   [-2, 4, -2],
                                   [1, -2, 1]])
            elif filter_type == "laplacian":
                kernel = np.array([[0, -1, 0],
                                   [-1, 4, -1],
                                   [0, -1, 0]])
            elif filter_type == "laplacian_of_gaussian":
                kernel = cv2.getGaussianKernel(5, 0) * cv2.getGaussianKernel(5, 0).T
                kernel *= -1
                kernel[2, 2] = 1 + 4

            self.processed_image = cv2.filter2D(self.original_image, -1, kernel)
            self.display_image(self.processed_image, self.canvas_processed)

    def define_filter(self):
        try:
            size = simpledialog.askinteger("Define Filter", "Enter the size of the matrix \n [ex: 3 for a 3x3 filter] :")
            if size is None or size <= 0:
                messagebox.showerror("Error", "Invalid matrix size.")
                return

            entry_value = simpledialog.askstring("Define Filter", f"Enter {size**2} values separated by spaces :")
            if entry_value is not None:
                values = [float(val) for val in entry_value.split()]
                self.dynamic_kernel = np.array(values).reshape((size, size))
        except Exception as e:
            messagebox.showerror("Error", f"Error defining dynamic filter: {str(e)}")

    def reset_user_filter(self):
        self.dynamic_kernel = None
        print("User filter reset.")

    def save_image(self):
        if self.processed_image is not None:
            output_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if output_path:
                cv2.imwrite(output_path, self.processed_image)
                print(f"Image saved to {output_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSegmentationApp(root)
    root.mainloop()
