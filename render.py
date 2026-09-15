import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle, Rectangle, Polygon


class FlyRenderer:

    def __init__(self, landmarks, arena_radius=1.0):

        self.arena_radius = arena_radius
        self.landmark_radius = 0.88

        # Create figure
        self.fig, self.ax = plt.subplots(
            figsize=(7, 7)
        )

        self.ax.set_xlim(-1.3, 1.3)
        self.ax.set_ylim(-1.3, 1.3)

        self.ax.set_aspect("equal")

        self.ax.set_title(
            "THE FLY — Arena with Landmarks"
        )

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        self.ax.grid(alpha=0.2)

        # --------------------------------
        # Arena boundary
        # --------------------------------

        arena = Circle(
            (0, 0),
            radius=self.arena_radius,
            fill=False,
            edgecolor="black",
            linewidth=2
        )

        self.ax.add_patch(arena)

        # --------------------------------
        # Fly
        # --------------------------------

        self.ax.scatter(
            0,
            0,
            s=90,
            zorder=10
        )

        # --------------------------------
        # Heading line
        # --------------------------------

        self.heading_line, = self.ax.plot(
            [0, 0],
            [0, 0.7],
            linewidth=3,
            zorder=9
        )

        # --------------------------------
        # Heading text
        # --------------------------------

        self.heading_text = self.ax.text(
            -1.25,
            1.15,
            "Heading: 0.0°",
            fontsize=12
        )

        # Draw fixed landmarks
        self.draw_landmarks(landmarks)


    def bearing_to_xy(
        self,
        bearing,
        radius
    ):

        """
        0 degrees = up

        positive angle = clockwise/right
        """

        x = radius * np.sin(bearing)

        y = radius * np.cos(bearing)

        return x, y


    def draw_landmarks(
        self,
        landmarks
    ):

        for landmark in landmarks:

            bearing = landmark["bearing"]

            shape = landmark["shape"]

            colour = landmark["colour"]

            width = landmark["width"]

            height = landmark["height"]

            x, y = self.bearing_to_xy(
                bearing,
                self.landmark_radius
            )

            if shape == "striped_rectangle":

                self.draw_striped_rectangle(
                    x,
                    y,
                    width,
                    height,
                    colour
                )

            elif shape == "circle":

                self.draw_circle_landmark(
                    x,
                    y,
                    width,
                    height,
                    colour
                )

            elif shape == "triangle":

                self.draw_triangle(
                    x,
                    y,
                    width,
                    height,
                    colour
                )

            elif shape == "checkerboard":

                self.draw_checkerboard(
                    x,
                    y,
                    width,
                    height,
                    colour
                )


    def draw_striped_rectangle(
        self,
        center_x,
        center_y,
        width,
        height,
        colour
    ):

        x0 = center_x - width / 2

        y0 = center_y - height / 2

        rectangle = Rectangle(
            (x0, y0),
            width,
            height,
            facecolor="white",
            edgecolor=colour,
            linewidth=2,
            zorder=5
        )

        self.ax.add_patch(rectangle)

        number_of_stripes = 6

        for i in range(
            1,
            number_of_stripes
        ):

            stripe_x = (
                x0
                + i * width / number_of_stripes
            )

            self.ax.plot(
                [stripe_x, stripe_x],
                [y0, y0 + height],
                color=colour,
                linewidth=2,
                zorder=6
            )


    def draw_circle_landmark(
        self,
        center_x,
        center_y,
        width,
        height,
        colour
    ):

        radius = min(
            width,
            height
        ) / 2

        circle = Circle(
            (center_x, center_y),
            radius=radius,
            facecolor=colour,
            edgecolor="black",
            linewidth=1.5,
            zorder=5
        )

        self.ax.add_patch(circle)


    def draw_triangle(
        self,
        center_x,
        center_y,
        width,
        height,
        colour
    ):

        points = np.array([

            [
                center_x,
                center_y + height / 2
            ],

            [
                center_x - width / 2,
                center_y - height / 2
            ],

            [
                center_x + width / 2,
                center_y - height / 2
            ]

        ])

        triangle = Polygon(
            points,
            closed=True,
            facecolor=colour,
            edgecolor="black",
            linewidth=1.5,
            zorder=5
        )

        self.ax.add_patch(triangle)


    def draw_checkerboard(
        self,
        center_x,
        center_y,
        width,
        height,
        colour
    ):

        x0 = center_x - width / 2

        y0 = center_y - height / 2

        rows = 4

        columns = 4

        cell_width = width / columns

        cell_height = height / rows

        for row in range(rows):

            for column in range(columns):

                if (
                    row + column
                ) % 2 == 0:

                    face_colour = "black"

                else:

                    face_colour = "white"

                square = Rectangle(

                    (
                        x0
                        + column * cell_width,

                        y0
                        + row * cell_height
                    ),

                    cell_width,

                    cell_height,

                    facecolor=face_colour,

                    edgecolor="black",

                    linewidth=0.3,

                    zorder=5
                )

                self.ax.add_patch(square)

        border = Rectangle(
            (x0, y0),
            width,
            height,
            fill=False,
            edgecolor=colour,
            linewidth=2,
            zorder=6
        )

        self.ax.add_patch(border)


    def update_heading(
        self,
        heading
    ):

        """
        Move the heading arrow to the
        current fly heading.
        """

        heading_length = 0.7

        x = (
            heading_length
            * np.sin(heading)
        )

        y = (
            heading_length
            * np.cos(heading)
        )

        self.heading_line.set_data(
            [0, x],
            [0, y]
        )

        heading_degrees = (
            np.rad2deg(heading)
        )

        self.heading_text.set_text(
            f"Heading: {heading_degrees:.1f}°"
        )

        return (
            self.heading_line,
            self.heading_text
        )