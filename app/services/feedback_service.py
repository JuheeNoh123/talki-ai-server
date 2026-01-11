# app/services/feedback_service.py
import numpy as np

def generate_feedback(analysis_result: dict):
    """
    녹화 영상 분석 결과(JSON)를 기반으로
    사용자에게 종합적인 피드백 메시지와 통계를 생성합니다.
    """
    feedback = []

    # --- 시선 피드백 ---
    # analyze_record_video 결과라면 "horiz_counts"가 포함되어 있음
    gaze = analysis_result.get("eyes", {})
    horiz_mode = gaze.get("horiz_mode")
    horiz_counts = gaze.get("horiz_counts")
    vert_mode = gaze.get("vert_mode")
    samples = gaze.get("samples", 0)

    # 부정적 피드백이 발생했는지 체크용 플래그
    negative_gaze_feedback = False 

    if horiz_counts and samples > 0:
        # 비율 계산 (test_record_multiprocess에서 counts를 줌)
        # N/A는 counts에 포함 안되어 있을 수 있음 (gaze_from_landmarks가 N/A 리턴 안함? 아, None 리턴시 gazes에 append 안함)
        # test_record_multiprocess:237 if gaze: gazes.append(gaze)
        # 따라서 N/A 비율은 (total_frames - samples)로 추정해야 함.
        # 하지만 여기서는 단순화하여 counts 내 비율만 봅니다.
        
        most_common_h = max(horiz_counts, key=horiz_counts.get)
        ratio_h = horiz_counts[most_common_h] / samples
        
        # 임계값 0.4 적용
        if ratio_h > 0.4:
            if most_common_h == "left":
                feedback.append("시선이 계속 왼쪽을 향해 있습니다. 중앙을 봐주세요.")
                negative_gaze_feedback = True
            elif most_common_h == "right":
                feedback.append("시선이 계속 오른쪽을 향해 있습니다. 중앙을 봐주세요.")
                negative_gaze_feedback = True
            # Center인 경우는 아래에서 Vert 체크 후 칭찬 여부 결정
        else:
            # 특정 방향 쏠림이 40% 이하 -> 골고루 봤거나 산만함
            pass
    elif horiz_mode:
        # Fallback (Legacy)
        if horiz_mode == "left":
            feedback.append("시선이 계속 왼쪽을 향해 있습니다. 중앙을 봐주세요.")
            negative_gaze_feedback = True
        elif horiz_mode == "right":
            feedback.append("시선이 계속 오른쪽을 향해 있습니다. 중앙을 봐주세요.")
            negative_gaze_feedback = True
        elif horiz_mode == "n/a":
             feedback.append("얼굴이 잘 보이지 않습니다. 정면을 바라봐주세요.")
             negative_gaze_feedback = True
    
    if vert_mode == "down":
        feedback.append("고개가 조금 숙여져 있습니다. 시선을 약간 위로 올려보세요.")
        negative_gaze_feedback = True
    elif vert_mode == "up":
        feedback.append("시선이 산만합니다. 청중을 집중해서 바라봐주세요.")
        negative_gaze_feedback = True

    # 아무런 부정적 피드백이 없고, 수평 시선이 중앙 위주일 때만 칭찬
    if not negative_gaze_feedback:
        # 비율 데이터가 있다면 확인
        if horiz_counts and samples > 0:
             most_common_h = max(horiz_counts, key=horiz_counts.get)
             if most_common_h == "center":
                  feedback.append("시선이 안정적입니다. 좋아요!")
        elif horiz_mode == "center":
             feedback.append("시선이 안정적입니다. 좋아요!")


    # --- 자세/동작 피드백 ---
    # 녹화 분석 결과에는 handArmMovementAvg, handArmMovementMaxRolling, pose_warning_count가 있음
    avg_speed = analysis_result.get("handArmMovementAvg", 0.0)
    #max_rolling_speed = analysis_result.get("handArmMovementMaxRolling", avg_speed)
    #warning_count = analysis_result.get("pose_warning_count", 0)
    warning_ratio = analysis_result.get("pose_warning_ratio", 0.0)
    pose_samples = analysis_result.get("pose_samples", 0)
    

    RECORD_WARN_RATIO_TH = 0.3   # 녹화 기준
    RECORD_WARN_COUNT_TH = 30    # (선택)


    # 임계값 적용 (실시간 로직 시뮬레이션 결과 반영):
    # 실시간 분석 로직이 "경고"를 조금이라도(혹은 일정 비율 이상) 냈다면 여기서도 경고
    if pose_samples == 0:
        feedback.append("자세 인식이 어렵습니다. 상체가 화면에 잘 보이도록 촬영해 주세요.")
    elif warning_ratio > RECORD_WARN_RATIO_TH:
        feedback.append("몸을 비교적 많이 움직이는 편입니다. 조금 더 차분한 자세를 취해보세요.")
    else:
        feedback.append("자세가 안정적으로 유지되었습니다.")

    # --- 음성 피드백 ---
    wpm = analysis_result.get("WPM", 0)

    if wpm > 0:
        if wpm < 100:
            feedback.append("말이 조금 느립니다. 자신감 있게 말씀해 보세요.")
        elif wpm > 180:
            feedback.append("말이 너무 빠릅니다. 조금 천천히 말씀해 보세요.")
        else:
            feedback.append("적절한 말 속도입니다. 좋습니다!")
    else:
        feedback.append("음성 입력이 없습니다. 마이크 상태를 확인해주세요.")

    # --- 종합 요약 ---
    summary = " / ".join(feedback)
    score = calc_score(analysis_result)
    
    # 상세 수치 반환
    details = {
        "score": score,
        "feedback_text": summary,
        "metrics": {
            "gaze_horiz_ratio": round(ratio_h, 2) if 'ratio_h' in locals() else 0.0,
            "gaze_horiz_mode": most_common_h if 'most_common_h' in locals() else (horiz_mode or "N/A"),
            "pose_avg_speed": round(avg_speed, 4),
            #"pose_max_rolling_speed": round(max_rolling_speed, 4), 
            "pose_warning_ratio": warning_ratio, # 실시간 경고 발생 횟수
            "pose_samples": analysis_result.get("pose_samples", 0),
            "speech_wpm": round(wpm, 1) if 'wpm' in locals() else 0,
            "speech_fillers": fillers if 'fillers' in locals() else 0
        }
    }
    return details


def calc_score(result: dict):
    """
    간단한 피드백 점수 계산 (0~100)
    """
    base = 100

    # 시선 안정성 평가
    gaze = result.get("gaze", {})
    # horiz_mode가 center가 아니면 감점? 
    # ratio > 0.4 and mode != center 일때 감점하는게 논리적.
    # 여기선 단순화:
    if gaze.get("horiz_mode") != "center":
        # 하지만 앞단에서 0.4 이하면 피드백 안줬음. 점수도?
        # 복잡하니 일단 mode 기준 유지하되 조금 완화?
        # 그냥 mode 기준 유지. (엄격하게)
        base -= 10
    if gaze.get("vert_mode") != "center":
        base -= 5

    # 음성 평가
    speech = result.get("speech")
    if speech:
        wpm = speech.get("wpm", 0)
        fillers = speech.get("fillers_freq", 0)
        if wpm < 100 or wpm > 180:
            base -= 10
        base -= int(min(fillers, 5))  # 필러 많으면 감점

    # 자세 평가
    #if "handArmMovementAvg" in result:
    if result.get("handArmMovementAvg", 0) > 0.05:
        base -= 10
    


    return max(base, 0)
