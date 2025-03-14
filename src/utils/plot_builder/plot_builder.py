import numpy as np
import matplotlib.pyplot as plt
from .plot_types import PlotType


class PlotBuilder:
    def __init__(self):
        self.fig, self.axes = None, None
        self.plots = []

    def set_grid(self, rows, cols):
        self.fig, self.axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        self.axes = self.axes.flatten() if rows * cols > 1 else [self.axes]
        return self

    def add_plot(self, index, data, plot_type=PlotType.UNCHANGED, **kwargs):
        if self.fig is None or self.axes is None:
            raise ValueError("Grid is not set. Use set_grid(rows, cols) first.")

        if index >= len(self.axes):
            raise IndexError("Index exceeds the number of grid cells.")

        ax = self.axes[index]
        image_kwargs = {k: v for k, v in kwargs.items() if k not in ["xlabel", "ylabel", "title", "colorbar", "annotations"]}
        img = self._generate_image(ax, data, plot_type, **image_kwargs)

        self._apply_axes_settings(ax, **kwargs)
        self._apply_special_settings(ax, img, **kwargs)

        self.plots.append((index, data, plot_type, kwargs))
        return self

    def _generate_image(self, ax, data, plot_type, **kwargs):
        plot_actions = {
            PlotType.UNCHANGED:  lambda: ax.imshow(data, **kwargs),
            PlotType.REAL:       lambda: ax.imshow(data.real, **kwargs),
            PlotType.IMAG:       lambda: ax.imshow(data.imag, **kwargs),
            PlotType.ABS:        lambda: ax.imshow(np.abs(data), **kwargs),
            PlotType.ANGLE:      lambda: ax.imshow(np.angle(data), **kwargs),
            PlotType.REAL_IMAG:  lambda: ax.imshow(data.imag * data.real, **kwargs),
            PlotType.ABS_SQUARE: lambda: ax.imshow(np.abs(data) ** 2, **kwargs)
        }

        if plot_type not in plot_actions:
            valid_types = ", ".join([pt.value for pt in PlotType])
            raise ValueError(f"Unsupported plot_type. Use PlotType enum values: {valid_types}.")

        return plot_actions[plot_type]()

    def _apply_axes_settings(self, ax, **kwargs):
        for key, value in kwargs.items():
            if key in ["xlabel", "ylabel", "title"]:
                getattr(ax, f"set_{key}")(value)

    def _apply_special_settings(self, ax, img, **kwargs):
        if "colorbar" in kwargs and kwargs["colorbar"]:
            self.fig.colorbar(img, ax=ax)

        if "annotations" in kwargs and isinstance(kwargs["annotations"], list):
            for annotation in kwargs["annotations"]:
                ax.annotate(
                    annotation.get("text", ""),
                    xy=annotation.get("xy", (0, 0)),
                    xytext=annotation.get("xytext", None),
                    arrowprops=annotation.get("arrowprops", None),
                    **annotation.get("kwargs", {})
                )

    def build(self):
        empty_axes = [ax for ax in self.axes if not ax.has_data()]
        for ax in empty_axes:
            ax.remove()

        self.axes = [ax for ax in self.axes if ax in self.fig.axes]
        plt.tight_layout()
        return self.fig, self.axes
