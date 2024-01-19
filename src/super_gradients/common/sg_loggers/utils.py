from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PlottableMetricOutput:
    scalar: float

    title: str
    x: List[float]
    y: List[float]
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None

    def __str__(self):
        return str(self.scalar)

    def __repr__(self):
        return str(self.scalar)

    def __float__(self):
        return float(self.scalar)

    def __round__(self, n=None):
        return round(self.scalar, n)

    # Comparison operations
    def __lt__(self, other):
        return self.scalar < other

    def __le__(self, other):
        return self.scalar <= other

    def __eq__(self, other):
        return self.scalar == other

    def __ne__(self, other):
        return self.scalar != other

    def __gt__(self, other):
        return self.scalar > other

    def __ge__(self, other):
        return self.scalar >= other

    # Unary operations
    def __neg__(self):
        return -self.scalar

    def __pos__(self):
        return +self.scalar

    def __abs__(self):
        return abs(self.scalar)

    # Boolean value testing
    def __bool__(self):
        return bool(self.scalar)

    # Arithmetic operations with improved type handling and error messages
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return self.scalar + other
        elif isinstance(other, PlottableMetricOutput):
            return self.scalar + other.scalar
        raise TypeError(f"Unsupported operand type(s) for +. Got {type(other)}")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return self.scalar - other
        elif isinstance(other, PlottableMetricOutput):
            return self.scalar - other.scalar
        raise TypeError(f"Unsupported operand type(s) for -. Got {type(other)}")

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return other - self.scalar
        elif isinstance(other, PlottableMetricOutput):
            return other.scalar - self.scalar
        raise TypeError(f"Unsupported operand type(s) for -. Got {type(other)}")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return self.scalar * other
        elif isinstance(other, PlottableMetricOutput):
            return self.scalar * other.scalar
        raise TypeError(f"Unsupported operand type(s) for *. Got {type(other)}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self.scalar / other
        elif isinstance(other, PlottableMetricOutput):
            return self.scalar / other.scalar
        raise TypeError(f"Unsupported operand type(s) for /. Got {type(other)}")

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return other / self.scalar
        elif isinstance(other, PlottableMetricOutput):
            return other.scalar / self.scalar
        raise TypeError(f"Unsupported operand type(s) for /. Got {type(other)}")

    def __floordiv__(self, other):
        if isinstance(other, (int, float)):
            return self.scalar // other
        elif isinstance(other, PlottableMetricOutput):
            return self.scalar // other.scalar
        raise TypeError(f"Unsupported operand type(s) for //. Got {type(other)}")

    def __rfloordiv__(self, other):
        if isinstance(other, (int, float)):
            return other // self.scalar
        elif isinstance(other, PlottableMetricOutput):
            return other.scalar // self.scalar
        raise TypeError(f"Unsupported operand type(s) for //. Got {type(other)}")

    def __mod__(self, other):
        if isinstance(other, (int, float)):
            return self.scalar % other
        elif isinstance(other, PlottableMetricOutput):
            return self.scalar % other.scalar
        raise TypeError(f"Unsupported operand type(s) for %. Got {type(other)}")

    def __rmod__(self, other):
        if isinstance(other, (int, float)):
            return other % self.scalar
        elif isinstance(other, PlottableMetricOutput):
            return other.scalar % self.scalar
        raise TypeError(f"Unsupported operand type(s) for %. Got {type(other)}")

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            return self.scalar**other
        elif isinstance(other, PlottableMetricOutput):
            return self.scalar**other.scalar
        raise TypeError(f"Unsupported operand type(s) for **. Got {type(other)}")

    def __rpow__(self, other):
        if isinstance(other, (int, float)):
            return other**self.scalar
        elif isinstance(other, PlottableMetricOutput):
            return other.scalar**self.scalar
        raise TypeError(f"Unsupported operand type(s) for **. Got {type(other)}")

    def draw_plot(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.x, self.y)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_title(self.title)

        # Create a numpy array to store the image data
        fig.canvas.draw()
        image_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_data = image_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image_data
