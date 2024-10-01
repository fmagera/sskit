"""
Taken from the SoccerNet-calibration code https://github.com/SoccerNet/sn-calibration/blob/main/src/soccerpitch.py
"""
import numpy as np

class SoccerPitch:
    """Static class variables that are specified by the rules of the game """
    GOAL_LINE_TO_PENALTY_MARK = 11.0
    PENALTY_AREA_WIDTH = 40.32
    PENALTY_AREA_LENGTH = 16.5
    GOAL_AREA_WIDTH = 18.32
    GOAL_AREA_LENGTH = 5.5
    CENTER_CIRCLE_RADIUS = 9.15
    GOAL_HEIGHT = 2.44
    GOAL_LENGTH = 7.32

    lines_classes = [
        'Big rect. left bottom',
        'Big rect. left main',
        'Big rect. left top',
        'Big rect. right bottom',
        'Big rect. right main',
        'Big rect. right top',
        'Circle central',
        'Circle left',
        'Circle right',
        'Goal left crossbar',
        'Goal left post left',
        'Goal left post right',
        'Goal right crossbar',
        'Goal right post left',
        'Goal right post right',
        'Goal unknown',
        'Line unknown',
        'Middle line',
        'Side line bottom',
        'Side line left',
        'Side line right',
        'Side line top',
        'Small rect. left bottom',
        'Small rect. left main',
        'Small rect. left top',
        'Small rect. right bottom',
        'Small rect. right main',
        'Small rect. right top'
    ]


    def __init__(self, pitch_length=105., pitch_width=68.):
        """
        Initialize 3D coordinates of all elements of the soccer pitch.
        :param pitch_length: According to FIFA rules, length belong to [90,120] meters
        :param pitch_width: According to FIFA rules, length belong to [45,90] meters
        """


        self.PITCH_LENGTH = pitch_length
        self.PITCH_WIDTH = pitch_width

        self.circles = [
            'Circle central',
            'Circle left',
            'Circle right'
        ]
        self.center_mark = np.array([0, 0, 0], dtype='float')
        self.halfway_and_bottom_touch_line_mark = np.array([0, pitch_width / 2.0, 0], dtype='float')
        self.halfway_and_top_touch_line_mark = np.array([0, -pitch_width / 2.0, 0], dtype='float')
        self.halfway_line_and_center_circle_top_mark = np.array([0, -SoccerPitch.CENTER_CIRCLE_RADIUS, 0],
                                                                dtype='float')
        self.halfway_line_and_center_circle_bottom_mark = np.array([0, SoccerPitch.CENTER_CIRCLE_RADIUS, 0],
                                                                   dtype='float')
        self.bottom_right_corner = np.array([pitch_length / 2.0, pitch_width / 2.0, 0], dtype='float')
        self.bottom_left_corner = np.array([-pitch_length / 2.0, pitch_width / 2.0, 0], dtype='float')
        self.top_right_corner = np.array([pitch_length / 2.0, -pitch_width / 2.0, 0], dtype='float')
        self.top_left_corner = np.array([-pitch_length / 2.0, -pitch_width / 2.0, 0], dtype='float')

        self.pitch_corners = [self.bottom_right_corner, self.bottom_left_corner, self.top_left_corner, self.top_right_corner]

        self.left_goal_bottom_left_post = np.array([-pitch_length / 2.0, SoccerPitch.GOAL_LENGTH / 2., 0.],
                                                   dtype='float')
        self.left_goal_top_left_post = np.array(
            [-pitch_length / 2.0, SoccerPitch.GOAL_LENGTH / 2., -SoccerPitch.GOAL_HEIGHT], dtype='float')
        self.left_goal_bottom_right_post = np.array([-pitch_length / 2.0, -SoccerPitch.GOAL_LENGTH / 2., 0.],
                                                    dtype='float')
        self.left_goal_top_right_post = np.array(
            [-pitch_length / 2.0, -SoccerPitch.GOAL_LENGTH / 2., -SoccerPitch.GOAL_HEIGHT], dtype='float')

        self.right_goal_bottom_left_post = np.array([pitch_length / 2.0, -SoccerPitch.GOAL_LENGTH / 2., 0.],
                                                    dtype='float')
        self.right_goal_top_left_post = np.array(
            [pitch_length / 2.0, -SoccerPitch.GOAL_LENGTH / 2., -SoccerPitch.GOAL_HEIGHT], dtype='float')
        self.right_goal_bottom_right_post = np.array([pitch_length / 2.0, SoccerPitch.GOAL_LENGTH / 2., 0.],
                                                     dtype='float')
        self.right_goal_top_right_post = np.array(
            [pitch_length / 2.0, SoccerPitch.GOAL_LENGTH / 2., -SoccerPitch.GOAL_HEIGHT], dtype='float')

        self.left_penalty_mark = np.array([-pitch_length / 2.0 + SoccerPitch.GOAL_LINE_TO_PENALTY_MARK, 0, 0],
                                          dtype='float')
        self.right_penalty_mark = np.array([pitch_length / 2.0 - SoccerPitch.GOAL_LINE_TO_PENALTY_MARK, 0, 0],
                                           dtype='float')

        self.left_penalty_area_top_right_corner = np.array(
            [-pitch_length / 2.0 + SoccerPitch.PENALTY_AREA_LENGTH, -SoccerPitch.PENALTY_AREA_WIDTH / 2.0, 0],
            dtype='float')
        self.left_penalty_area_top_left_corner = np.array(
            [-pitch_length / 2.0, -SoccerPitch.PENALTY_AREA_WIDTH / 2.0, 0],
            dtype='float')
        self.left_penalty_area_bottom_right_corner = np.array(
            [-pitch_length / 2.0 + SoccerPitch.PENALTY_AREA_LENGTH, SoccerPitch.PENALTY_AREA_WIDTH / 2.0, 0],
            dtype='float')
        self.left_penalty_area_bottom_left_corner = np.array(
            [-pitch_length / 2.0, SoccerPitch.PENALTY_AREA_WIDTH / 2.0, 0],
            dtype='float')
        self.right_penalty_area_top_right_corner = np.array(
            [pitch_length / 2.0, -SoccerPitch.PENALTY_AREA_WIDTH / 2.0, 0],
            dtype='float')
        self.right_penalty_area_top_left_corner = np.array(
            [pitch_length / 2.0 - SoccerPitch.PENALTY_AREA_LENGTH, -SoccerPitch.PENALTY_AREA_WIDTH / 2.0, 0],
            dtype='float')
        self.right_penalty_area_bottom_right_corner = np.array(
            [pitch_length / 2.0, SoccerPitch.PENALTY_AREA_WIDTH / 2.0, 0],
            dtype='float')
        self.right_penalty_area_bottom_left_corner = np.array(
            [pitch_length / 2.0 - SoccerPitch.PENALTY_AREA_LENGTH, SoccerPitch.PENALTY_AREA_WIDTH / 2.0, 0],
            dtype='float')

        self.left_goal_area_top_right_corner = np.array(
            [-pitch_length / 2.0 + SoccerPitch.GOAL_AREA_LENGTH, -SoccerPitch.GOAL_AREA_WIDTH / 2.0, 0], dtype='float')
        self.left_goal_area_top_left_corner = np.array([-pitch_length / 2.0, - SoccerPitch.GOAL_AREA_WIDTH / 2.0, 0],
                                                       dtype='float')
        self.left_goal_area_bottom_right_corner = np.array(
            [-pitch_length / 2.0 + SoccerPitch.GOAL_AREA_LENGTH, SoccerPitch.GOAL_AREA_WIDTH / 2.0, 0], dtype='float')
        self.left_goal_area_bottom_left_corner = np.array([-pitch_length / 2.0, SoccerPitch.GOAL_AREA_WIDTH / 2.0, 0],
                                                          dtype='float')
        self.right_goal_area_top_right_corner = np.array([pitch_length / 2.0, -SoccerPitch.GOAL_AREA_WIDTH / 2.0, 0],
                                                         dtype='float')
        self.right_goal_area_top_left_corner = np.array(
            [pitch_length / 2.0 - SoccerPitch.GOAL_AREA_LENGTH, -SoccerPitch.GOAL_AREA_WIDTH / 2.0, 0], dtype='float')
        self.right_goal_area_bottom_right_corner = np.array([pitch_length / 2.0, SoccerPitch.GOAL_AREA_WIDTH / 2.0, 0],
                                                            dtype='float')
        self.right_goal_area_bottom_left_corner = np.array(
            [pitch_length / 2.0 - SoccerPitch.GOAL_AREA_LENGTH, SoccerPitch.GOAL_AREA_WIDTH / 2.0, 0], dtype='float')

        x = -pitch_length / 2.0 + SoccerPitch.PENALTY_AREA_LENGTH
        dx = SoccerPitch.PENALTY_AREA_LENGTH - SoccerPitch.GOAL_LINE_TO_PENALTY_MARK
        y = -np.sqrt(SoccerPitch.CENTER_CIRCLE_RADIUS * SoccerPitch.CENTER_CIRCLE_RADIUS - dx * dx)
        self.top_left_16M_penalty_arc_mark = np.array([x, y, 0], dtype='float')

        x = pitch_length / 2.0 - SoccerPitch.PENALTY_AREA_LENGTH
        dx = SoccerPitch.PENALTY_AREA_LENGTH - SoccerPitch.GOAL_LINE_TO_PENALTY_MARK
        y = -np.sqrt(SoccerPitch.CENTER_CIRCLE_RADIUS * SoccerPitch.CENTER_CIRCLE_RADIUS - dx * dx)
        self.top_right_16M_penalty_arc_mark = np.array([x, y, 0], dtype='float')

        x = -pitch_length / 2.0 + SoccerPitch.PENALTY_AREA_LENGTH
        dx = SoccerPitch.PENALTY_AREA_LENGTH - SoccerPitch.GOAL_LINE_TO_PENALTY_MARK
        y = np.sqrt(SoccerPitch.CENTER_CIRCLE_RADIUS * SoccerPitch.CENTER_CIRCLE_RADIUS - dx * dx)
        self.bottom_left_16M_penalty_arc_mark = np.array([x, y, 0], dtype='float')

        x = pitch_length / 2.0 - SoccerPitch.PENALTY_AREA_LENGTH
        dx = SoccerPitch.PENALTY_AREA_LENGTH - SoccerPitch.GOAL_LINE_TO_PENALTY_MARK
        y = np.sqrt(SoccerPitch.CENTER_CIRCLE_RADIUS * SoccerPitch.CENTER_CIRCLE_RADIUS - dx * dx)
        self.bottom_right_16M_penalty_arc_mark = np.array([x, y, 0], dtype='float')


        self.point_dict = {
            "CENTER_MARK": self.center_mark,
            "L_PENALTY_MARK": self.left_penalty_mark,
            "R_PENALTY_MARK": self.right_penalty_mark,
            "TL_PITCH_CORNER": self.top_left_corner,
            "BL_PITCH_CORNER": self.bottom_left_corner,
            "TR_PITCH_CORNER": self.top_right_corner,
            "BR_PITCH_CORNER": self.bottom_right_corner,
            "L_PENALTY_AREA_TL_CORNER": self.left_penalty_area_top_left_corner,
            "L_PENALTY_AREA_TR_CORNER": self.left_penalty_area_top_right_corner,
            "L_PENALTY_AREA_BL_CORNER": self.left_penalty_area_bottom_left_corner,
            "L_PENALTY_AREA_BR_CORNER": self.left_penalty_area_bottom_right_corner,
            "R_PENALTY_AREA_TL_CORNER": self.right_penalty_area_top_left_corner,
            "R_PENALTY_AREA_TR_CORNER": self.right_penalty_area_top_right_corner,
            "R_PENALTY_AREA_BL_CORNER": self.right_penalty_area_bottom_left_corner,
            "R_PENALTY_AREA_BR_CORNER": self.right_penalty_area_bottom_right_corner,
            "L_GOAL_AREA_TL_CORNER": self.left_goal_area_top_left_corner,
            "L_GOAL_AREA_TR_CORNER": self.left_goal_area_top_right_corner,
            "L_GOAL_AREA_BL_CORNER": self.left_goal_area_bottom_left_corner,
            "L_GOAL_AREA_BR_CORNER": self.left_goal_area_bottom_right_corner,
            "R_GOAL_AREA_TL_CORNER": self.right_goal_area_top_left_corner,
            "R_GOAL_AREA_TR_CORNER": self.right_goal_area_top_right_corner,
            "R_GOAL_AREA_BL_CORNER": self.right_goal_area_bottom_left_corner,
            "R_GOAL_AREA_BR_CORNER": self.right_goal_area_bottom_right_corner,
            "L_GOAL_TL_POST": self.left_goal_top_left_post,
            "L_GOAL_TR_POST": self.left_goal_top_right_post,
            "L_GOAL_BL_POST": self.left_goal_bottom_left_post,
            "L_GOAL_BR_POST": self.left_goal_bottom_right_post,
            "R_GOAL_TL_POST": self.right_goal_top_left_post,
            "R_GOAL_TR_POST": self.right_goal_top_right_post,
            "R_GOAL_BL_POST": self.right_goal_bottom_left_post,
            "R_GOAL_BR_POST": self.right_goal_bottom_right_post,
            "T_TOUCH_AND_HALFWAY_LINES_INTERSECTION": self.halfway_and_top_touch_line_mark,
            "B_TOUCH_AND_HALFWAY_LINES_INTERSECTION": self.halfway_and_bottom_touch_line_mark,
            "T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION": self.halfway_line_and_center_circle_top_mark,
            "B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION": self.halfway_line_and_center_circle_bottom_mark,
            "TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION": self.top_left_16M_penalty_arc_mark,
            "BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION": self.bottom_left_16M_penalty_arc_mark,
            "TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION": self.top_right_16M_penalty_arc_mark,
            "BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION": self.bottom_right_16M_penalty_arc_mark
        }
        self.points = self.point_dict

        self.line_extremities = dict()
        self.line_extremities["Big rect. left bottom"] = (self.point_dict["L_PENALTY_AREA_BL_CORNER"],
                                                          self.point_dict["L_PENALTY_AREA_BR_CORNER"])
        self.line_extremities["Big rect. left top"] = (self.point_dict["L_PENALTY_AREA_TL_CORNER"],
                                                       self.point_dict["L_PENALTY_AREA_TR_CORNER"])
        self.line_extremities["Big rect. left main"] = (self.point_dict["L_PENALTY_AREA_TR_CORNER"],
                                                        self.point_dict["L_PENALTY_AREA_BR_CORNER"])
        self.line_extremities["Big rect. right bottom"] = (self.point_dict["R_PENALTY_AREA_BL_CORNER"],
                                                           self.point_dict["R_PENALTY_AREA_BR_CORNER"])
        self.line_extremities["Big rect. right top"] = (self.point_dict["R_PENALTY_AREA_TL_CORNER"],
                                                        self.point_dict["R_PENALTY_AREA_TR_CORNER"])
        self.line_extremities["Big rect. right main"] = (self.point_dict["R_PENALTY_AREA_TL_CORNER"],
                                                         self.point_dict["R_PENALTY_AREA_BL_CORNER"])

        self.line_extremities["Small rect. left bottom"] = (self.point_dict["L_GOAL_AREA_BL_CORNER"],
                                                            self.point_dict["L_GOAL_AREA_BR_CORNER"])
        self.line_extremities["Small rect. left top"] = (self.point_dict["L_GOAL_AREA_TL_CORNER"],
                                                         self.point_dict["L_GOAL_AREA_TR_CORNER"])
        self.line_extremities["Small rect. left main"] = (self.point_dict["L_GOAL_AREA_TR_CORNER"],
                                                          self.point_dict["L_GOAL_AREA_BR_CORNER"])
        self.line_extremities["Small rect. right bottom"] = (self.point_dict["R_GOAL_AREA_BL_CORNER"],
                                                             self.point_dict["R_GOAL_AREA_BR_CORNER"])
        self.line_extremities["Small rect. right top"] = (self.point_dict["R_GOAL_AREA_TL_CORNER"],
                                                          self.point_dict["R_GOAL_AREA_TR_CORNER"])
        self.line_extremities["Small rect. right main"] = (self.point_dict["R_GOAL_AREA_TL_CORNER"],
                                                           self.point_dict["R_GOAL_AREA_BL_CORNER"])

        self.line_extremities["Side line top"] = (self.point_dict["TL_PITCH_CORNER"],
                                                  self.point_dict["TR_PITCH_CORNER"])
        self.line_extremities["Side line bottom"] = (self.point_dict["BL_PITCH_CORNER"],
                                                     self.point_dict["BR_PITCH_CORNER"])
        self.line_extremities["Side line left"] = (self.point_dict["TL_PITCH_CORNER"],
                                                   self.point_dict["BL_PITCH_CORNER"])
        self.line_extremities["Side line right"] = (self.point_dict["TR_PITCH_CORNER"],
                                                    self.point_dict["BR_PITCH_CORNER"])
        self.line_extremities["Middle line"] = (self.point_dict["T_TOUCH_AND_HALFWAY_LINES_INTERSECTION"],
                                                self.point_dict["B_TOUCH_AND_HALFWAY_LINES_INTERSECTION"])

        self.line_extremities["Goal left crossbar"] = (self.point_dict["L_GOAL_TR_POST"],
                                                       self.point_dict["L_GOAL_TL_POST"])
        self.line_extremities["Goal left post left"] = (self.point_dict["L_GOAL_TL_POST"],
                                                         self.point_dict["L_GOAL_BL_POST"])
        self.line_extremities["Goal left post right"] = (self.point_dict["L_GOAL_TR_POST"],
                                                         self.point_dict["L_GOAL_BR_POST"])

        self.line_extremities["Goal right crossbar"] = (self.point_dict["R_GOAL_TL_POST"],
                                                        self.point_dict["R_GOAL_TR_POST"])
        self.line_extremities["Goal right post left"] = (self.point_dict["R_GOAL_TL_POST"],
                                                         self.point_dict["R_GOAL_BL_POST"])
        self.line_extremities["Goal right post right"] = (self.point_dict["R_GOAL_TR_POST"],
                                                          self.point_dict["R_GOAL_BR_POST"])
        self.line_extremities["Circle right"] = (self.point_dict["TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION"],
                                                 self.point_dict["BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION"])
        self.line_extremities["Circle left"] = (self.point_dict["TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION"],
                                                self.point_dict["BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION"])



    def sample_field_circle(self, circle_id, dist):
        if circle_id == "Circle right":
            top = self.point_dict["TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION"]
            bottom = self.point_dict["BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION"]
            center = self.point_dict["R_PENALTY_MARK"]
            toAngle = np.arctan2(top[1] - center[1],
                                 top[0] - center[0]) + 2 * np.pi
            fromAngle = np.arctan2(bottom[1] - center[1],
                                   bottom[0] - center[0]) + 2 * np.pi
        elif circle_id == "Circle left":
            top = self.point_dict["TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION"]
            bottom = self.point_dict["BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION"]
            center = self.point_dict["L_PENALTY_MARK"]
            fromAngle = np.arctan2(top[1] - center[1],
                                   top[0] - center[0]) + 2 * np.pi
            toAngle = np.arctan2(bottom[1] - center[1],
                                 bottom[0] - center[0]) + 2 * np.pi
        elif circle_id == "Circle central":
            center = self.point_dict["CENTER_MARK"]
            fromAngle = 0.
            toAngle = 2 * np.pi
        else:
            raise ValueError(f"Unknown circle: {circle_id}")

        if toAngle < fromAngle:
            toAngle += 2 * np.pi
        x1 = center[0] + np.cos(fromAngle) * SoccerPitch.CENTER_CIRCLE_RADIUS
        y1 = center[1] + np.sin(fromAngle) * SoccerPitch.CENTER_CIRCLE_RADIUS
        z1 = 0.
        xn = center[0] + np.cos(toAngle) * SoccerPitch.CENTER_CIRCLE_RADIUS
        yn = center[1] + np.sin(toAngle) * SoccerPitch.CENTER_CIRCLE_RADIUS
        zn = 0.

        start = np.array((x1, y1, z1))
        end = np.array((xn, yn, zn))
        polyline = [start]
        length = SoccerPitch.CENTER_CIRCLE_RADIUS * (toAngle - fromAngle)
        nb_pts = int(length / dist)
        dangle = dist / SoccerPitch.CENTER_CIRCLE_RADIUS
        for i in range(1, nb_pts + 1):
            angle = fromAngle + i * dangle
            x = center[0] + np.cos(angle) * SoccerPitch.CENTER_CIRCLE_RADIUS
            y = center[1] + np.sin(angle) * SoccerPitch.CENTER_CIRCLE_RADIUS
            z = 0.
            point = np.array((x, y, z))
            polyline.append(point)
        polyline.append(end)
        return polyline

    def sample_field_points(self, dist=0.1, dist_circles=0.2):
        """
        Samples each pitch element every dist meters, returns a dictionary associating the class of the element with a list of points sampled along this element.
        :param dist: the distance in meters between each point sampled
        :param dist_circles: the distance in meters between each point sampled on circles
        :return:  a dictionary associating the class of the element with a list of points sampled along this element.
        """
        polylines = dict()
        for circle_id in self.circles:
            polylines[circle_id] = self.sample_field_circle(circle_id, dist_circles)

        for key, line in self.line_extremities.items():

            if "Circle" in key:
                pass
            else:
                polylines[key] = self.sample_field_line(key, dist)
        return polylines

    def sample_field_line(self, line_id, dist):
        line_extremities = self.line_extremities[line_id]
        start = line_extremities[0]
        end = line_extremities[1]

        polyline = [start]

        total_dist = np.sqrt(np.sum(np.square(start - end)))
        nb_pts = int(total_dist / dist - 1)

        v = end - start
        v /= np.linalg.norm(v)
        prev_pt = start
        for i in range(nb_pts):
            pt = prev_pt + dist * v
            prev_pt = pt
            if v[2] == 0 and "cross" not in line_id:
                pt[2] = 0.
            polyline.append(pt)
        polyline.append(end)
        return polyline

