import cv2
import numpy as np


def get_lines(frame):
    # edge detection
    original_frame = frame
    gray_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    processed_frame = cv2.Canny(blurred_frame, 50, 150)

    # frame segmentation
    [rows, cols] = processed_frame.shape
    vertices = np.array(
        [[(0, rows), (cols // 2, rows // 2 + 50), (cols, rows)]], np.int32)
    mask = np.zeros_like(processed_frame)
    cv2.fillPoly(mask, vertices, 255)
    segment_frame = cv2.bitwise_and(processed_frame, mask)

    # Hough transform
    lines = cv2.HoughLinesP(
        segment_frame, 1, np.pi / 180, 100, np.array([]),
        minLineLength=100, maxLineGap=50)

    # calculate the parameters of lines
    min_y, max_y = float('inf'), processed_frame.shape[0]
    for line in lines:
        for coords in line:
            min_y = min([min_y, coords[1], coords[3]])
    line_dict = {}

    for i, line in enumerate(lines):
        for coords in line:
            X, Y = (coords[0], coords[2]), (coords[1], coords[3])
            A = np.vstack([X, np.ones(len(X))]).T
            m, b = np.linalg.lstsq(A, Y)[0]

            x1, x2 = (min_y - b) / m, (max_y - b) / m
            line_dict[i] = [m, b, [int(x1), min_y, int(x2), max_y]]

    correct_lines = {}
    for i in line_dict:
        correct_lines_copy = correct_lines.copy()
        m, b, line = line_dict[i]
        if not len(correct_lines):
            correct_lines[m] = [[m, b, line]]
        else:
            # It is considered as a straight line within the error range
            found_copy = False
            for other_m in correct_lines_copy:
                if not found_copy:
                    if abs(other_m * 0.8) < abs(m) < abs(other_m * 1.2):
                        other_b = correct_lines[other_m][0][1]
                        if abs(other_b * 0.8) < abs(b) < abs(other_b * 1.2):
                            correct_lines[other_m].append([m, b, line])
                            found_copy = True
                            break
                    else:
                        correct_lines[m] = [[m, b, line]]

    count_line = {}
    for line in correct_lines:
        count_line[line] = len(correct_lines[line])
    final_linesId = sorted(
        count_line.items(), key=lambda item: item[1])[::-1][:2]
    line1 = np.mean(np.array(correct_lines[final_linesId[0][0]])[
                    :, 2].tolist(), axis=0)
    line2 = np.mean(np.array(correct_lines[final_linesId[1][0]])[
                    :, 2].tolist(), axis=0)

    return line1.astype(np.int32), line2.astype(np.int32)


def main():
    video = cv2.VideoCapture('input.mp4')
    while video.isOpened():
        ret, frame = video.read()
        line1, line2 = get_lines(frame)
        cv2.line(frame, (line1[0], line1[1]),
                 (line1[2], line1[3]), [0, 255, 0], 10)
        cv2.line(frame, (line2[0], line2[1]),
                 (line2[2], line2[3]), [0, 255, 0], 10)
        cv2.imshow('frame', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


main()
