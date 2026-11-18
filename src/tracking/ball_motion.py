"""Short, appearance-checked continuations of a directly observed ball."""
import cv2


class BallMotion:
    def __init__(self, fps: float) -> None:
        self.limit = min(5, max(1, round(0.1 * fps)))
        self.age = 0
        self.template = None
        self.centre = None
        self.previous_view = None

    def update(self, frame, x: float | None, y: float | None) -> tuple[int, int] | None:
        # Keep the original template throughout a gap to limit accumulated drift.
        view = frame[::32, ::32]
        cut = (self.previous_view is not None and
               (self.previous_view.shape != view.shape or
                cv2.absdiff(view, self.previous_view).mean() > 28))
        self.previous_view = view.copy()
        if cut:
            self.template = None
        if x is not None:
            self.age = 0
            self.centre = (round(x), round(y))
            patch = cv2.getRectSubPix(frame, (15, 15), self.centre)
            self.template = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            return None
        self.age += 1
        if self.template is None or self.age > self.limit:
            return None
        cx, cy = self.centre
        left, top = max(0, cx - 32), max(0, cy - 32)
        search = frame[top:cy + 33, left:cx + 33]
        if min(search.shape[:2]) < 15 or self.template.std() < 12:
            self.template = None
            return None
        gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
        scores = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, (px, py) = cv2.minMaxLoc(scores)
        # Reject ambiguous alternatives, including repeated turf and pitch markings.
        scores[max(0, py - 5):py + 6, max(0, px - 5):px + 6] = -1
        if score < 0.84 or score - float(scores.max()) < 0.12:
            self.template = None
            return None
        self.centre = (left + px + 7, top + py + 7)
        return self.centre
