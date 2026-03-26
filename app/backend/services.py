import cv2
import os
import torch


def run_pipeline(image, model, mp_face, mp_hands, save_dir):

    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    os.makedirs(save_dir, exist_ok=True)

    persons = []
    faces = []
    hands_list = []

    # ===== YOLO =====
    results = model(image, conf=0.3)[0]

    for i, box in enumerate(results.boxes):

        cls = int(box.cls[0])

        if cls == 0:

            x1,y1,x2,y2 = map(int, box.xyxy[0])

            crop = image[y1:y2, x1:x2]

            path = f"{save_dir}/person_{i}.jpg"
            cv2.imwrite(path, crop)

            persons.append(path)

    # ===== FACE =====
    with mp_face.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.8
    ) as face:

        res = face.process(rgb)

        if res.detections:

            for j, det in enumerate(res.detections):

                bbox = det.location_data.relative_bounding_box

                x1 = int(bbox.xmin*w)
                y1 = int(bbox.ymin*h)
                bw = int(bbox.width*w)
                bh = int(bbox.height*h)

                crop = image[y1:y1+bh, x1:x1+bw]

                path = f"{save_dir}/face_{j}.jpg"
                cv2.imwrite(path, crop)

                faces.append(path)

    # ===== HAND =====
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.4
    ) as hands:

        res = hands.process(rgb)

        if res.multi_hand_landmarks:

            for k, hand in enumerate(res.multi_hand_landmarks):

                xs = [int(lm.x*w) for lm in hand.landmark]
                ys = [int(lm.y*h) for lm in hand.landmark]

                crop = image[min(ys):max(ys), min(xs):max(xs)]

                path = f"{save_dir}/hand_{k}.jpg"
                cv2.imwrite(path, crop)

                hands_list.append(path)

    return {
        "persons": persons,
        "faces": faces,
        "hands": hands_list
    }


def get_image_score(image, model, transform, device):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)

        confidence = torch.max(probs).item()
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        entropy_score = 1 / (1 + entropy)

        score = (0.7 * confidence) + (0.3 * entropy_score)

    return score