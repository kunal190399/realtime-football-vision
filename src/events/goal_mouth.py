"""Review candidates from directly observed ball centres in a fixed camera view."""
from dataclasses import dataclass
from math import hypot, isfinite


@dataclass(frozen=True)
class GoalMouth:
    first: tuple
    second: tuple
    field: tuple

    def __post_init__(self):
        for x, y in (self.first, self.second, self.field):
            if not (0 <= x <= 1280 and 0 <= y <= 720):
                raise ValueError("Goal coordinates must fit the 1280 × 720 reference view.")
        if self.first == self.second or abs(self.project(self.field)[1]) < 1:
            raise ValueError("Choose distinct posts and a field point clear of the goal line.")

    def project(self, point):
        dx, dy = (b-a for a, b in zip(self.first, self.second))
        x, y = (b-a for a, b in zip(self.first, point))
        length = hypot(dx, dy)
        return (x*dx + y*dy) / length**2, (dx*y - dy*x) / length


class GoalMonitor:
    def __init__(self, goal):
        self.goal = goal
        self.side = 1 if goal.project(goal.field)[1] > 0 else -1
        self.reset()

    def reset(self):
        self.previous = None
        self.advances = 0
        self.announced = False

    def observe(self, seconds, point, source, identity=None):
        if source not in {"tracked", "detected", "appearance"} or point is None or not all(map(isfinite, point)):
            self.reset()
            return []
        along, distance = self.goal.project(point)
        distance *= self.side
        previous = self.previous
        self.previous = (seconds, point, along, distance, identity)
        if previous is None:
            return []
        old_time, old_point, old_along, old_distance, old_id = previous
        dt = seconds - old_time
        if (not 0 < dt <= 0.151 or hypot(point[0]-old_point[0], point[1]-old_point[1]) > 1200*dt
                or (identity is not None and old_id is not None and identity != old_id)):
            self.advances = 0
            self.announced = False
            return []
        advance = old_distance - distance
        if advance < 80*dt:
            self.advances = 0
            self.announced = False
            return []
        self.advances += 1
        fraction = old_distance / advance
        intersection = old_along + fraction*(along-old_along)
        if not 0 <= intersection <= 1 or old_distance <= 0:
            return []
        events = []
        if distance > 0 and self.advances >= 2 and not self.announced and distance*dt/advance <= 2:
            events.append(dict(event="shot_candidate", seconds=seconds))
            self.announced = True
        if distance <= 0:
            events.append(dict(event="goal_line_crossing_candidate", seconds=old_time + fraction*dt))
        return events
