class PresentationType:
    ONLINE_SMALL = "online_small"   # 화상 발표 (줌 등)
    SMALL = "small"                 # 소규모 발표
    LARGE = "large"                 # 대규모 발표


FEEDBACK_CRITERIA = {
    PresentationType.ONLINE_SMALL: {
        "wpm_min": 120,
        "wpm_max": 140,
        "fillers_per_min": 2,
        "pose_min": 0.010,
        "pose_max": 0.018,
        "gaze_front_ratio": 0.65,
    },
    PresentationType.SMALL: {
        "wpm_min": 130,
        "wpm_max": 150,
        "fillers_per_min": 3,
        "pose_min": 0.012,
        "pose_max": 0.020,
        "gaze_front_ratio": 0.60,
    },
    PresentationType.LARGE: {
        "wpm_min": 140,
        "wpm_max": 160,
        "fillers_per_min": 4,
        "pose_min": 0.014,
        "pose_max": 0.022,
        "gaze_front_ratio": 0.55,
    },
}
