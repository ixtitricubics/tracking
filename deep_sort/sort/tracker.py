# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np

from deep_sort.sort import detection
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track, TrackState
from utils.data.dataset import Dataset

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric,db:Dataset, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.height, self.width = 1,1
        self.db = db
        self._initiate_track_from_db()
    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
            # print("init track")

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        # active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        # features, targets = [], []
        # for track in self.tracks:
        #     if not track.is_confirmed():
        #         continue
        #     features += track.features
        #     targets += [track.track_id for _ in track.features]
        #     track.features = []
        # self.metric.partial_fit(
        #     np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        """
        this match function only matches the detections using their x,y,a,h
        and ious.
        """
        def gated_metric(tracks, dets, track_indices, detection_indices):
            # features = np.array([dets[i].feature for i in detection_indices])
            # bboxes = np.array([dets[i].tlwh for i in detection_indices])
            # bboxes = np.array([dets[i].tlwh for i in detection_indices])
            # # preds_det = np.array([dets[i].pred for i in detection_indices])
            # means = np.array([tracks[i].mean for i in track_indices])
            # targets = np.array([tracks[i].track_id for i in track_indices])
            # target_preds = np.array([tracks[i].preds for i in track_indices])
            tracks_ = [tracks[i] for i in track_indices]
            dets_ = [dets[i] for i in detection_indices]
            cost_matrix = self.metric.distance(tracks_, dets_, self.height, self.width)
            # # Invalidate infeasible entries in cost matrix based on the state distributions obtained by Kalman filtering.
            # # this loss calculates the distance beetween each detection and the distribution of each track(mean, covariance)
            # # i dont understand how though.

            # cost_matrix = linear_assignment.gate_cost_matrix(
            #     self.kf, cost_matrix, tracks, dets, track_indices,
            #     detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if len(t.features) >0 ] # if t.is_confirmed()
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if len(t.features)  == 0]
            #i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.min_cost_matching(
                gated_metric, self.metric.matching_threshold, self.tracks,
                detections, confirmed_tracks)
            # linear_assignment.matching_cascade(
            #     gated_metric, self.metric.matching_threshold, self.max_age,
            #     self.tracks, detections, confirmed_tracks)
        
        
        # print("----matches a ", len(matches_a))

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update <=5]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update >5]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, from_db = True):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
    def _initiate_track_from_db(self):
        print("initializing from db started ")
        # detections = []
        # for i in range(len(self.db.db)):
        #     det = detection.Detection([-1,-1,-1, -1], 1.0, self.db.db[i][Dataset.Items.EMBEDDINGS], [-1,-1])
        #     det.id = 2
        #     detections.append(det)
        # self.predict()
        # self.update(detections)
        # self._initiate_track(det, True)
        for i in range(len(self.db.db)):
        #     det = detection.Detection([-1,-1,-1, -1], 1.0, , [-1,-1])
        #     det.id = 2
        #     detections.append(det)
            mean, covariance = self.kf.initiate([0,0,1, 1])
            track = Track(mean, covariance, self._next_id, self.n_init, self.max_age,
                None)
            track.features = list(self.db.db[i][Dataset.Items.EMBEDDINGS])
            track.state = TrackState.Confirmed
            track.hits = 5

            self.tracks.append(track)
            self._next_id += 1

        print("initializing from db finished ")