"""Review sustained trajectories towards a calibrated goal in a stationary view."""
from collections import deque
from dataclasses import dataclass
from math import hypot

import cv2
import numpy as np

from src.events.goal_mouth import GoalMonitor


@dataclass
class Path:
    points: list
    length: int
    monitor: GoalMonitor | None = None


class GoalMotion:
    def __init__(self, goal):
        self.goal = goal
        self.side = np.sign(goal.project(goal.field)[1])
        ends = np.array([goal.first, goal.second])
        padding = min(380, max(90, round(4 * np.linalg.norm(ends[0]-ends[1]))))
        self.left, self.top = np.maximum(0, ends.min(axis=0)-padding).astype(int)
        self.right, self.bottom = np.minimum([1280, 720], ends.max(axis=0)+padding).astype(int)
        self.visible_centre = None
        self.reference = None
        self.history = deque(maxlen=5)
        self.paths = []
        self.last_event = {}
        self.display_posts = (goal.first, goal.second)

    def reset(self):
        self.paths.clear()
        self.history.clear()
        self.last_event.clear()
        self.display_posts = ()

    def observe(self, frame, seconds):
        self.visible_centre = None
        view = cv2.resize(frame, (1280, 720))
        gray = cv2.cvtColor(view[self.top:self.bottom, self.left:self.right], cv2.COLOR_BGR2GRAY)
        factor = min(1., 320 / max(gray.shape))
        small = cv2.resize(gray, None, fx=factor, fy=factor)
        if self.reference is None:
            self.reference = small
        transform = np.eye(2, 3, dtype=np.float32)
        try:
            score, transform = cv2.findTransformECC(self.reference, small, transform, cv2.MOTION_TRANSLATION,
                                                    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, .01))
        except cv2.error:
            self.reset()
            return []
        if not np.isfinite(score) or not np.isfinite(transform).all() or score < .85 or np.any(np.abs(transform[:, 2]) > .12 * np.array(small.shape[::-1])):
            self.reset()
            return []
        transform[:, 2] /= factor
        self.display_posts = tuple(np.array(p) + transform[:, 2] for p in (self.goal.first, self.goal.second))
        aligned = cv2.warpAffine(gray, transform, gray.shape[::-1], flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REPLICATE)
        points = []
        if len(self.history) >= 3:
            background = np.median(self.history, axis=0).astype(np.uint8)
            mask = (cv2.subtract(aligned, background) > 23).astype(np.uint8)
            _, _, stats, centres = cv2.connectedComponentsWithStats(mask)
            for (x, y, w, h, area), centre in zip(stats[1:], centres[1:]):
                if 4 <= area <= 100 and 2 <= min(w, h) and max(w, h) <= 22 and area > .35*w*h:
                    points.append((centre[0]+self.left, centre[1]+self.top))
        self.history.append(aligned)
        # Widespread motion indicates an unsuitable view for this detector.
        if len(points) > 100:
            self.paths.clear()
            return []
        events = self.advance(seconds, points)
        confirmed = [path for path in self.paths if path.monitor is not None]
        if confirmed:
            _, centre = max(confirmed, key=lambda path: path.length).points[-1]
            self.visible_centre = np.asarray(centre) + transform[:, 2]
        return events

    def advance(self, seconds, points):
        """Associate observations by direction and constant-velocity prediction."""
        proposals = []
        for index, path in enumerate(self.paths):
            last_time, last = path.points[-1]
            dt = seconds - last_time
            if not 0 < dt <= .151:
                continue
            along, distance = self.goal.project(last)
            if distance * self.side <= 0:
                continue
            for j, point in enumerate(points):
                current_along, current_distance = self.goal.project(point)
                forward = (distance-current_distance) * self.side
                step = hypot(point[0]-last[0], point[1]-last[1])
                if forward < 80*dt or step > 1200*dt:
                    continue
                intersection = along + (current_along-along)*abs(distance)/forward
                if not -.1 <= intersection <= 1.1:
                    continue
                error = step
                if len(path.points) >= 2:
                    prev_time, prev = path.points[-2]
                    ratio = dt / (last_time-prev_time)
                    error = hypot(point[0]-last[0]-(last[0]-prev[0])*ratio,
                                  point[1]-last[1]-(last[1]-prev[1])*ratio)
                    if error > max(3., .45*step):
                        continue
                proposals.append((-path.length, error, index, j))
        claimed, used, paths, events = set(), set(), [], []
        for _, _, index, j in sorted(proposals):
            if index in used or j in claimed:
                continue
            used.add(index)
            claimed.add(j)
            path = self.paths[index]
            path.points.append((seconds, points[j]))
            path.points = path.points[-4:]
            path.length += 1
            observations = []
            if path.length == 4:
                path.monitor = GoalMonitor(self.goal)
                observations = path.points
            elif path.monitor:
                observations = path.points[-1:]
            for time, position in observations:
                for event in path.monitor.observe(time, position, "detected"):
                    kind = event["event"]
                    if event["seconds"] - self.last_event.get(kind, -10) >= .5:
                        self.last_event[kind] = event["seconds"]
                        events.append(event)
            paths.append(path)
        paths.extend(Path([(seconds, point)], 1) for j, point in enumerate(points) if j not in claimed)
        self.paths = paths
        return events
