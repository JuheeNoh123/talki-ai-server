import numpy as np
import math

# 시선 분석을 위한 랜드마크 인덱스 상수
LEFT_IRIS  = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]
LEFT_EYE_OUTER, LEFT_EYE_INNER = 33, 133
RIGHT_EYE_INNER, RIGHT_EYE_OUTER = 362, 263
UPPER_LID = [159, 386] 
LOWER_LID = [145, 374] 

# 클라이언트/서버 공용: 분석에 필수적인 얼굴 랜드마크 인덱스 모음
REQUIRED_FACE_IDX = (
    LEFT_IRIS + RIGHT_IRIS + 
    [LEFT_EYE_OUTER, LEFT_EYE_INNER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER] +
    UPPER_LID + LOWER_LID
)

# 포즈 분석을 위한 랜드마크 인덱스 (서버 분석용)
# MediaPipe Constants (copied to avoid dependency)
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14

HAND_KEYS = [
    POSE_LEFT_WRIST, POSE_RIGHT_WRIST, 
    POSE_LEFT_ELBOW, POSE_RIGHT_ELBOW
]

# 포즈 필수 인덱스 (Client 전송용)
REQUIRED_POSE_IDX = HAND_KEYS

def _get_pt(lms, idx):
    """
    랜드마크 컨테이너(lms)에서 idx에 해당하는 (x, y) 좌표를 반환하는 헬퍼 함수.
    lms는 list(MediaPipe 객체)이거나 dict(JSON)일 수 있음.
    """
    # 1. 딕셔너리 키 접근 (Str -> Int 변환 시도)
    if isinstance(lms, dict):
        # 468 vs "468"
        val = lms.get(idx) or lms.get(str(idx))
        if val is None:
            return 0.0, 0.0 # Default fallback
    else:
        # 리스트 인덱스 접근
        try:
            val = lms[idx]
        except IndexError:
            return 0.0, 0.0

    # 2. 값 추출 (객체.x vs 딕셔너리['x'])
    if hasattr(val, 'x'):
        return val.x, val.y
    elif isinstance(val, dict):
        return val.get('x', 0.0), val.get('y', 0.0)
    
    return 0.0, 0.0

def _mean_xy(lms, idxs):
    """랜드마크 리스트에서 특정 인덱스들의 평균 (x, y) 좌표를 구합니다."""
    xs, ys = [], []
    for i in idxs:
        x, y = _get_pt(lms, i)
        # 0.0, 0.0 might bias mean? Assuming valid landmarks usually.
        xs.append(x)
        ys.append(y)
    
    if not xs: return 0.0, 0.0
    return float(np.mean(xs)), float(np.mean(ys))

def gaze_from_landmarks(lms):
    """
    얼굴 랜드마크에서 시선 방향(dx, dy)을 계산합니다.
    lms: MediaPipe Landmark List 또는 Dict
    """
    lix, liy = _mean_xy(lms, LEFT_IRIS)
    rix, riy = _mean_xy(lms, RIGHT_IRIS)

    # 눈의 가로 길이 계산
    lx_out_x, _ = _get_pt(lms, LEFT_EYE_OUTER)
    lx_in_x, _  = _get_pt(lms, LEFT_EYE_INNER)
    rx_in_x, _  = _get_pt(lms, RIGHT_EYE_INNER)
    rx_out_x, _ = _get_pt(lms, RIGHT_EYE_OUTER)

    left_w  = abs(lx_out_x - lx_in_x)
    right_w = abs(rx_out_x - rx_in_x)
    eye_w = (left_w + right_w) / 2.0

    # 눈의 세로 길이 계산 (눈꺼풀 기준)
    _, ly_top_y = _get_pt(lms, UPPER_LID[0]) # approx using first point y
    _, ry_top_y = _get_pt(lms, UPPER_LID[1]) # actually need y of both?
    # Original logic: (lms[UPPER_LID[0]].y + lms[UPPER_LID[1]].y) / 2.0
    # Let's match original logic accurately
    _, ul0_y = _get_pt(lms, UPPER_LID[0])
    _, ul1_y = _get_pt(lms, UPPER_LID[1])
    lid_top = (ul0_y + ul1_y) / 2.0

    _, ll0_y = _get_pt(lms, LOWER_LID[0])
    _, ll1_y = _get_pt(lms, LOWER_LID[1])
    lid_bot = (ll0_y + ll1_y) / 2.0
    
    eye_h = abs(lid_bot - lid_top)

    # 눈 중심 좌표 계산
    eye_center_x = ( (lx_out_x + lx_in_x)/2 + (rx_out_x + rx_in_x)/2 ) / 2
    eye_center_y = (lid_top + lid_bot) / 2

    # 홍채(눈동자) 중심 좌표
    iris_x = (lix + rix) / 2
    iris_y = (liy + riy) / 2

    # 정규화된 편차 계산 (눈 크기에 비례하여 보정)
    dx = (iris_x - eye_center_x) / (eye_w + 1e-6)
    dy = (iris_y - eye_center_y) / (eye_h + 1e-6)

    # 방향 판별 (임계값 0.35 -> 0.15로 완화)
    horiz = "center"
    if   dx >  0.15: horiz = "right"
    elif dx < -0.15: horiz = "left"
    
    vert = "center"
    if   dy >  0.15: vert = "down"
    elif dy < -0.15: vert = "up"

    return {"dx": dx, "dy": dy, "horiz": horiz, "vert": vert}

def movement_speed(prev_points, curr_points):
    """이전 프레임과 현재 프레임 사이의 손/팔 랜드마크 이동 거리를 계산합니다."""
    # prev_points, curr_points: dict {index: (x, y)} or similar
    if prev_points is None or curr_points is None:
        return None
    dists = []
    for k in HAND_KEYS:
        # Check presence in both
        # k is int. Input keys might be string if from JSON.
        
        # Helper to get (x,y) from points container
        # Since calling logic usually passes a dict of {i: (x,y)} OR {i: LandmarkObj},
        # let's be flexible.
        
        p_xy = None
        c_xy = None
        
        # Try finding k (int) or str(k)
        if k in prev_points: p_xy = prev_points[k]
        elif str(k) in prev_points: p_xy = prev_points[str(k)]
        
        if k in curr_points: c_xy = curr_points[k]
        elif str(k) in curr_points: c_xy = curr_points[str(k)]
        
        if p_xy is not None and c_xy is not None:
            # Check format of p_xy
            px, py = 0.0, 0.0
            cx, cy = 0.0, 0.0
            
            if hasattr(p_xy, 'x'): px, py = p_xy.x, p_xy.y
            elif isinstance(p_xy, (tuple, list)): px, py = p_xy[0], p_xy[1]
            elif isinstance(p_xy, dict): px, py = p_xy.get('x', 0), p_xy.get('y', 0)
            
            if hasattr(c_xy, 'x'): cx, cy = c_xy.x, c_xy.y
            elif isinstance(c_xy, (tuple, list)): cx, cy = c_xy[0], c_xy[1]
            elif isinstance(c_xy, dict): cx, cy = c_xy.get('x', 0), c_xy.get('y', 0)

            dists.append(math.dist((px, py), (cx, cy)))

    return float(np.mean(dists)) if dists else None
