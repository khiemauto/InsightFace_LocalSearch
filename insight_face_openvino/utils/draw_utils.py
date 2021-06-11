import cv2
import numpy as np

points_colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 0, 0)]


def draw_boxes(
    img, boxes, name_tags=None, similarities=None, conf_threshold=0.0, sim_threshold=0.4, color=(0, 0, 255), thickness=2
):

    for i, box in enumerate(boxes):

        conf = box[4]
        box = box[:4]

        if conf < conf_threshold:
            continue

        font_scale = float(min(box[2] - box[0], box[3] - box[1]) * 0.01)

        text = f"{conf:.2f}"
        b = list(map(int, box))
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), color=color, thickness=thickness)
        tx = b[0]
        ty = b[1] + 12
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255))

        if similarities:
            similarity_score = similarities[i]
            if similarity_score > sim_threshold:
                cv2.putText(
                    img,
                    str(similarity_score)[:4],
                    (b[2] - 12, b[3] + 24),
                    cv2.FONT_HERSHEY_DUPLEX,
                    font_scale,
                    (255, 255, 255),
                )

                if name_tags:
                    tag = name_tags[i]
                    cv2.putText(img, tag, (tx - 32, b[3] + 50), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255))


def draw_landmarks(img, landmarks):
    for landm in landmarks:
        landm = np.reshape(landm, (2, 5))
        thickness = int(np.min(landm.max(axis=1) - landm.min(axis=1)) * 0.1)
        for i, (x, y) in enumerate(zip(landm[0], landm[1])):
            cv2.circle(img, (x, y), 1, points_colors[i], thickness)
