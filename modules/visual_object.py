"""

by Hao Xue @ 30/08/18

"""

from gv_tools.tracking.tracking_region import TrackingRegion
from gv_tools.tracking.tracklet import Tracklet


class VisualObject:

    def __init__(self, tracklet: Tracklet, scale: float = 1.0, color=(0, 255, 0)):
        self.tracklet: Tracklet = tracklet
        self.scale: float = scale
        self.color: tuple = color

        # Find the frame index to use as the avatar image.
        local_frame_index = len(tracklet.track_frames) // 2
        avatar_tracklet = tracklet.track_frames[local_frame_index]
        self.avatar_frame_index: int = avatar_tracklet.frame_index
        self.avatar_region: TrackingRegion = avatar_tracklet.display_region

        # Operational attributes.
        self.decay_process_index = None

    @property
    def rgb_color(self):
        return self.color[2], self.color[1], self.color[0]