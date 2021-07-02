import cv2
import numpy as np


# ----------------------------------- Functions

def show_pic(img, t=0):
    cv2.imshow('pic', img)
    cv2.waitKey(t)


def convolution(img, mask):
    rows, cols = img.shape
    output = np.ones((rows - 2, cols - 2), dtype=np.uint8)

    for i in range(1, rows - 1):
        print(i)
        for j in range(1, cols - 1):
            roi = img[i - 1:i + 2, j - 1:j + 2]
            output[i - 1, j - 1] = np.abs(np.sum(np.multiply(roi, mask)))

    return output


def convolution_n(img, scale):
    mask = np.ones((scale, scale), dtype=float) * (1 / scale ** 2)
    rows, cols = img.shape
    w, h = mask.shape
    output = np.ones((rows - (w - 1), cols - (h - 1)), dtype=np.uint8)

    w_lim = int((w - 1) / 2)
    h_lim = int((h - 1) / 2)

    for i in range(w_lim, rows - h_lim):
        for j in range(h_lim, cols - h_lim):
            roi = img[i - w_lim:i + w_lim + 1, j - h_lim:j + h_lim + 1]
            output[i - w_lim, j - h_lim] = np.abs(np.sum(np.multiply(roi, mask)))

    return output


# ---------------------------------------


if __name__ == "__main__":

    # question 2
    # image_lion = cv2.imread('q2/lion.png', cv2.IMREAD_GRAYSCALE)
    # mask_ = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    # out_lion = convolution(image_lion, mask=mask_)
    # cv2.imwrite('q2/lion_masked.jpg', out_lion)
    # show_pic(out_lion)
    # --------------------------------------------------------------

    # question 3
    # image = cv2.imread('q3/building.tif', cv2.IMREAD_GRAYSCALE)
    # mask_vertical = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    # mask_horizontal = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
    # out_building = convolution(image, mask=mask_horizontal)
    # cv2.imwrite('q3/building_masked_h.jpg', out_building)
    # show_pic(out_building)
    # --------------------------------------------------------------

    # question 4
    # pic = cv2.imread('q4/Mona_Lisa.jpg', cv2.IMREAD_GRAYSCALE)
    # for num in [3, 5, 7, 15]:
    #     result = convolution_n(pic, num)
    #     cv2.imwrite(f'q4/Mona_lisa_{num}.jpg', result)
    # --------------------------------------------------------------

    # question 5
    video = cv2.VideoCapture(0)

    v_w = int(video.get(3))
    v_h = int(video.get(4))
    # v_w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    # v_h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # fps = video.get(cv2.cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('captured.mp4', fourcc, 20.0, (v_w, v_h))  #   0x7634706d

    while True:
        flag, frame = video.read()

        # Wait for 'q' key to stop the program
        if cv2.waitKey(1) == ord('q'):
            break

        if flag:
            gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # print(gray_image.shape)

            rows, cols = gray_image.shape
            center = (round(rows / 2), round(cols / 2))
            sub_frame = gray_image[center[1] - 30:center[1] + 30, center[0] - 30: center[0] + 30]

            # frame = convolution_n(gray_image, 15)
            frame = cv2.GaussianBlur(gray_image, (31, 31), 30)

            # -----------
            sub_frame_masked = convolution_n(sub_frame, 7)
            print(np.mean(sub_frame_masked))
            if np.mean(sub_frame_masked) < 50:
                text = 'Black'
            elif 50 < np.mean(sub_frame_masked) < 100:
                text = 'Gray'
            else:
                text = 'White'

            frame = cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2, cv2.LINE_AA)

            frame[center[1] - 30:center[1] + 30, center[0] - 30: center[0] + 30] = sub_frame
            cv2.rectangle(frame, (center[0] + 30, center[1] + 30), (center[0] - 30, center[1] - 30), color=(0, 255, 0),
                          thickness=3)

            show_pic(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            writer.write(frame)


        else:
            break

    video.release()
    writer.release()
    cv2.destroyAllWindows()
