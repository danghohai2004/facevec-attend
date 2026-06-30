from abc import ABC, abstractmethod


class LivenessChecker(ABC):
    @abstractmethod
    def check(self, frame: bytes) -> bool:
        """Return True if frame is a real face, False if spoofing detected."""


class PassThroughChecker(LivenessChecker):
    # ponytail: placeholder, swap for real liveness model when available
    def check(self, frame: bytes) -> bool:
        return True


_checker: LivenessChecker = PassThroughChecker()


def get_liveness_checker() -> LivenessChecker:
    return _checker
