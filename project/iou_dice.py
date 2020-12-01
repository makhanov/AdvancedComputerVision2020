def iou_numpy(outputs, labels):
    mask = 255
    intersections = 0
    h, w = outputs.shape
    for i in range(h):
        for j in range(w):
            if labels[i][j]==mask:
                if labels[i][j]==outputs[i][j]:
                    intersections += 1

    test = outputs[:]+labels[:]
    union = test[test>0]
    iou = (intersections)/ float(len(union))
    
    return iou


def dice_coeff(outputs, labels): #F1 score
    mask = 255
    intersections = 0
    h, w = outputs.shape
    for i in range(h):
        for j in range(w):
            if labels[i][j]==mask:
                if labels[i][j]==outputs[i][j]:
                    intersections += 1
                    
    union = len(outputs[outputs>0])+len(labels[labels>0])
    dice = 2*intersections/union
    return dice