import numpy as np

from utils import wrap_angle


class FlyCamera:
    def __init__(
        self,
        width=128,
        height=64,
        fov_degrees=120
    ):
        self.width = width
        self.height = height

        # Use radians internally
        self.fov = np.deg2rad(fov_degrees)

        # Perspective focal length:
        #
        #          W / 2
        # f = ----------------
        #      tan(FOV / 2)
        #
        self.focal_length = (
            self.width / 2
        ) / np.tan(self.fov / 2)

        # RGB colours
        self.colours = {
            "blue": np.array([50, 100, 230], dtype=np.uint8),
            "red": np.array([220, 60, 60], dtype=np.uint8),
            "green": np.array([60, 180, 90], dtype=np.uint8),
            "black": np.array([20, 20, 20], dtype=np.uint8),
            "white": np.array([245, 245, 245], dtype=np.uint8),
        }


    def render(self, heading, landmarks):
        """
        Create the RGB image seen by the fly.

        Output shape:
            (64, 128, 3)
        """

        # Light background
        image = np.full(
            (
                self.height,
                self.width,
                3
            ),
            235,
            dtype=np.uint8
        )

        for landmark in landmarks:

            world_bearing = landmark["bearing"]

            # ----------------------------------
            # Relative bearing
            #
            # alpha_i = wrap(phi_i - theta_t)
            # ----------------------------------

            relative_bearing = wrap_angle(
                world_bearing - heading
            )

            # ----------------------------------
            # Check field of view
            #
            # 120° FOV means +/- 60°
            # ----------------------------------

            if abs(relative_bearing) > self.fov / 2:
                continue

            # ----------------------------------
            # Perspective projection
            #
            # x_i = W/2 + f tan(alpha_i)
            # ----------------------------------

            x_center = (
                self.width / 2
                + self.focal_length
                * np.tan(relative_bearing)
            )

            # All landmarks share the same
            # vertical centre for now
            y_center = self.height / 2

            # Convert normalized landmark sizes
            # into pixel dimensions
            pixel_width = max(
                4,
                int(
                    landmark["width"]
                    * self.width
                )
            )

            pixel_height = max(
                4,
                int(
                    landmark["height"]
                    * self.height
                )
            )

            self._draw_landmark(
                image=image,
                x_center=int(round(x_center)),
                y_center=int(round(y_center)),
                width=pixel_width,
                height=pixel_height,
                shape=landmark["shape"],
                colour=landmark["colour"]
            )

        return image


    def _draw_landmark(
        self,
        image,
        x_center,
        y_center,
        width,
        height,
        shape,
        colour
    ):

        if shape == "striped_rectangle":

            self._draw_striped_rectangle(
                image,
                x_center,
                y_center,
                width,
                height,
                colour
            )

        elif shape == "circle":

            self._draw_circle(
                image,
                x_center,
                y_center,
                width,
                height,
                colour
            )

        elif shape == "triangle":

            self._draw_triangle(
                image,
                x_center,
                y_center,
                width,
                height,
                colour
            )

        elif shape == "checkerboard":

            self._draw_checkerboard(
                image,
                x_center,
                y_center,
                width,
                height
            )

        else:

            raise ValueError(
                f"Unknown landmark shape: {shape}"
            )


    def _bounds(
        self,
        x_center,
        y_center,
        width,
        height
    ):

        x0 = int(
            x_center - width / 2
        )

        x1 = int(
            x_center + width / 2
        )

        y0 = int(
            y_center - height / 2
        )

        y1 = int(
            y_center + height / 2
        )

        return x0, x1, y0, y1


    def _draw_striped_rectangle(
        self,
        image,
        x_center,
        y_center,
        width,
        height,
        colour
    ):

        x0, x1, y0, y1 = self._bounds(
            x_center,
            y_center,
            width,
            height
        )

        colour_rgb = self.colours[colour]

        # Clip to image boundaries
        cx0 = max(0, x0)
        cx1 = min(self.width, x1)

        cy0 = max(0, y0)
        cy1 = min(self.height, y1)

        if cx0 >= cx1 or cy0 >= cy1:
            return

        image[
            cy0:cy1,
            cx0:cx1
        ] = self.colours["white"]

        stripe_width = max(
            2,
            width // 6
        )

        for x in range(
            x0,
            x1,
            stripe_width * 2
        ):

            sx0 = max(0, x)

            sx1 = min(
                self.width,
                x + stripe_width
            )

            if sx0 < sx1:

                image[
                    cy0:cy1,
                    sx0:sx1
                ] = colour_rgb


    def _draw_circle(
        self,
        image,
        x_center,
        y_center,
        width,
        height,
        colour
    ):

        radius_x = width / 2
        radius_y = height / 2

        x0 = max(
            0,
            int(x_center - radius_x)
        )

        x1 = min(
            self.width,
            int(x_center + radius_x) + 1
        )

        y0 = max(
            0,
            int(y_center - radius_y)
        )

        y1 = min(
            self.height,
            int(y_center + radius_y) + 1
        )

        if x0 >= x1 or y0 >= y1:
            return

        yy, xx = np.ogrid[
            y0:y1,
            x0:x1
        ]

        mask = (
            ((xx - x_center) / radius_x) ** 2
            +
            ((yy - y_center) / radius_y) ** 2
            <= 1
        )

        region = image[
            y0:y1,
            x0:x1
        ]

        region[mask] = self.colours[colour]


    def _draw_triangle(
        self,
        image,
        x_center,
        y_center,
        width,
        height,
        colour
    ):

        top_y = y_center - height / 2
        bottom_y = y_center + height / 2

        y_start = max(
            0,
            int(top_y)
        )

        y_end = min(
            self.height,
            int(bottom_y) + 1
        )

        colour_rgb = self.colours[colour]

        for y in range(
            y_start,
            y_end
        ):

            relative_y = (
                y - top_y
            ) / height

            half_width = (
                width / 2
            ) * relative_y

            x0 = int(
                x_center - half_width
            )

            x1 = int(
                x_center + half_width
            )

            x0 = max(
                0,
                x0
            )

            x1 = min(
                self.width,
                x1
            )

            if x0 < x1:

                image[
                    y,
                    x0:x1
                ] = colour_rgb


    def _draw_checkerboard(
        self,
        image,
        x_center,
        y_center,
        width,
        height
    ):

        x0, x1, y0, y1 = self._bounds(
            x_center,
            y_center,
            width,
            height
        )

        rows = 4
        columns = 4

        cell_width = max(
            1,
            width // columns
        )

        cell_height = max(
            1,
            height // rows
        )

        for row in range(rows):

            for column in range(columns):

                cell_x0 = (
                    x0
                    + column * cell_width
                )

                cell_y0 = (
                    y0
                    + row * cell_height
                )

                if column == columns - 1:
                    cell_x1 = x1
                else:
                    cell_x1 = (
                        cell_x0
                        + cell_width
                    )

                if row == rows - 1:
                    cell_y1 = y1
                else:
                    cell_y1 = (
                        cell_y0
                        + cell_height
                    )

                # Clip
                cx0 = max(
                    0,
                    cell_x0
                )

                cx1 = min(
                    self.width,
                    cell_x1
                )

                cy0 = max(
                    0,
                    cell_y0
                )

                cy1 = min(
                    self.height,
                    cell_y1
                )

                if cx0 >= cx1 or cy0 >= cy1:
                    continue

                if (
                    row + column
                ) % 2 == 0:

                    colour = self.colours["black"]

                else:

                    colour = self.colours["white"]

                image[
                    cy0:cy1,
                    cx0:cx1
                ] = colour
