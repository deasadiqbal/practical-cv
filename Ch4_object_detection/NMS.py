def NMS(boxes, class_confidence):
    result_box = []
    for b1 in boxes:
        dicard = False
        for b2 in boxes:
            if IOU(b1, b2)> A:
                if class_confidence[b2]> class_confidence[b1]:
                    dicard = True
        if not dicard:
            result_box.append(b1)
        return result_box