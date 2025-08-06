from numpy import ndarray

def get_outer_bounds(points: dict) -> list:
    x_min = 1000 * 1000
    x_max = 0
    y_min = 1000 * 1000
    y_max = 0
    for point_data in points.values():
        x = point_data['x']
        y = point_data['y']
        if x_min >= x:
            x_min = x
        if x_max <= x:
            x_max = x
        if y_min >= y:
            y_min = y
        if y_max <= y:
            y_max = y
    return [x_min, y_min, x_max, y_max]

def zoom_bounds(bounds: list, factor: float) -> list:
    x_min, y_min, x_max, y_max = bounds
    x_len = x_max - x_min
    y_len = y_max - y_min
    x_increase = int(x_len * factor)
    y_increase = int(y_len * factor)
    x_min = x_min - x_increase
    x_max = x_max + x_increase
    y_min = y_min - y_increase
    y_max = y_max + y_increase
    return [x_min, y_min, x_max, y_max]

def trim_bounds(bounds: list, image: ndarray) -> list:
    x_min, y_min, x_max, y_max = bounds
    img_height, img_width = image.shape[:2]
    x_min = max(0, x_min)
    x_max = min(img_width, x_max)
    y_min = max(0, y_min)
    y_max = min(img_height, y_max)
    return [x_min, y_min, x_max, y_max]

def get_square_bounds(image: ndarray, bounds: list) -> list:
    x_min, y_min, x_max, y_max = bounds
    x_mid = int((x_min + x_max) / 2)
    y_mid = int((y_min + y_max) / 2)
    radius = int(max(x_max - x_min, y_max - y_min))
    square_bounds = [x_mid - radius, y_mid - radius, x_mid + radius, y_mid + radius]
    square_bounds = trim_bounds(square_bounds, image)
    return square_bounds
