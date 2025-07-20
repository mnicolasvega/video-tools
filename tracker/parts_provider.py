import mediapipe as mp

def get_pose_points(
    draw_face: bool = True,
    draw_trunk: bool = True,
    draw_arms: bool = True,
    draw_legs: bool = True,
    draw_hands: bool = False,
    draw_feet: bool = False,
) -> dict:
    pose = mp.solutions.pose
    POINTS_FACE = {
        'NOSE':           pose.PoseLandmark.NOSE,
        'EYE_INNER (L)':  pose.PoseLandmark.LEFT_EYE_INNER,
        'EYE (L)':        pose.PoseLandmark.LEFT_EYE,
        'EYE_OUTER (L)':  pose.PoseLandmark.LEFT_EYE_OUTER,
        'EYE_INNER (R)':  pose.PoseLandmark.RIGHT_EYE_INNER,
        'EYE (R)':        pose.PoseLandmark.RIGHT_EYE,
        'EYE_OUTER (R)':  pose.PoseLandmark.RIGHT_EYE_OUTER,
        'EAR (L)':        pose.PoseLandmark.LEFT_EAR,
        'EAR (R)':        pose.PoseLandmark.RIGHT_EAR,
        'MOUTH (L)':      pose.PoseLandmark.MOUTH_LEFT,
        'MOUTH (R)':      pose.PoseLandmark.MOUTH_RIGHT,
    }
    POINTS_TRUNK = {
        'SHOULDER (L)':   pose.PoseLandmark.LEFT_SHOULDER,
        'SHOULDER (R)':   pose.PoseLandmark.RIGHT_SHOULDER,
        'HIP (L)':        pose.PoseLandmark.LEFT_HIP,
        'HIP (R)':        pose.PoseLandmark.RIGHT_HIP,
    }
    POINTS_ARMS = {
        'SHOULDER (L)':   pose.PoseLandmark.LEFT_SHOULDER,
        'SHOULDER (R)':   pose.PoseLandmark.RIGHT_SHOULDER,
        'ELBOW (L)':      pose.PoseLandmark.LEFT_ELBOW,
        'ELBOW (R)':      pose.PoseLandmark.RIGHT_ELBOW,
        'WRIST (L)':      pose.PoseLandmark.LEFT_WRIST,
        'WRIST (R)':      pose.PoseLandmark.RIGHT_WRIST,
    }
    POINTS_LEGS = {
        'HIP (L)':        pose.PoseLandmark.LEFT_HIP,
        'HIP (R)':        pose.PoseLandmark.RIGHT_HIP,
        'KNEE (L)':       pose.PoseLandmark.LEFT_KNEE,
        'KNEE (R)':       pose.PoseLandmark.RIGHT_KNEE,
        'ANKLE (L)':      pose.PoseLandmark.LEFT_ANKLE,
        'ANKLE (R)':      pose.PoseLandmark.RIGHT_ANKLE,
    }
    POINTS_HANDS = {
        'WRIST (L)':      pose.PoseLandmark.LEFT_WRIST,
        'WRIST (R)':      pose.PoseLandmark.RIGHT_WRIST,
        'PINKY (L)':      pose.PoseLandmark.LEFT_PINKY,
        'PINKY (R)':      pose.PoseLandmark.RIGHT_PINKY,
        'THUMB (L)':      pose.PoseLandmark.LEFT_THUMB,
        'THUMB (R)':      pose.PoseLandmark.RIGHT_THUMB,
        'INDEX (L)':      pose.PoseLandmark.LEFT_INDEX,
        'INDEX (R)':      pose.PoseLandmark.RIGHT_INDEX,
    }
    POINTS_FEET = {
        'ANKLE (L)':      pose.PoseLandmark.LEFT_ANKLE,
        'ANKLE (R)':      pose.PoseLandmark.RIGHT_ANKLE,
        'HEEL (L)':       pose.PoseLandmark.LEFT_HEEL,
        'HEEL (R)':       pose.PoseLandmark.RIGHT_HEEL,
        'FOOT_INDEX (L)': pose.PoseLandmark.LEFT_FOOT_INDEX,
        'FOOT_INDEX (R)': pose.PoseLandmark.RIGHT_FOOT_INDEX
    }
    parts = [
        POINTS_FACE if draw_face else {},
        POINTS_TRUNK if draw_trunk else {},
        POINTS_ARMS if draw_arms else {},
        POINTS_LEGS if draw_legs else {},
        POINTS_HANDS if draw_hands else {},
        POINTS_FEET if draw_feet else {},
    ]
    all_parts = {}
    for part in parts:
        all_parts.update(part)
    return all_parts
