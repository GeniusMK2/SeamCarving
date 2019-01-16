import numpy
from tqdm import tqdm


def load_matrix(path):
    from matplotlib.image import imread
    return imread(path)


def to_grayscale_image(raw_image_matrix, parameters=(0.299, 0.587, 0.114)):
    return numpy.dot(raw_image_matrix[..., :3], list(parameters))


def shift_1_pixel(image_matrix, direct_x, direct_y):
    shape = image_matrix.shape
    shifted_image = numpy.zeros((shape[0] + 2, shape[1] + 2))
    shifted_image[
    1 + direct_x: shape[0] + 1 + direct_x,
    1 + direct_y: shape[1] + 1 + direct_y] \
        = image_matrix[:, :]
    # from matplotlib.image import imsave
    # imsave("x"+str(x)+"y"+str(y)+".jpg", shifted_image)
    return shifted_image


def energy(raw_image_matrix):
    gray_scale = to_grayscale_image(raw_image_matrix)
    gray_scale = gray_scale / numpy.max(gray_scale)
    step = 1
    orthogonal_shift_direction = [(-step, 0), (step, 0), (0, step), (0, -step)]
    diagonal_shift_direction = [(-step, -step), (step, -step), (-step, step), (step, step)]
    energy_image = numpy.zeros(shape=(raw_image_matrix.shape[0] + 2 * step, raw_image_matrix.shape[1] + 2 * step))
    for _dir_ in orthogonal_shift_direction:
        energy_image += shift_1_pixel(gray_scale[:, :], _dir_[0], _dir_[1])

    energy_image = \
        energy_image[step:energy_image.shape[0] - step,
        step:energy_image.shape[1] - step]
    gray_scale = gray_scale * 4
    ret = gray_scale - energy_image
    return ret


def find_entry(line):
    argmin = numpy.where(line == numpy.min(line))
    from random import randint
    ret = []
    for i in range(0, len(argmin)):
        ret.append(argmin[i][randint(0, len(argmin) - 1)])
    return ret


def find_entry_point(energy_image):
    argmin_point = numpy.where(energy_image == numpy.min(energy_image))
    from random import randint
    rand = randint(0, len(argmin_point) - 1)
    return [argmin_point[0][rand], argmin_point[1][rand]]


# Need Modify
def find_path_line(energy_image, is_horizontal):
    path_list = []
    # Ver and hor is reversed.
    if is_horizontal:
        vertical = find_entry(energy_image[:, 0])[0]
        path_list.append(vertical)
        for horizontal in range(1, energy_image.shape[1]):
            _min_ = energy_image[vertical, horizontal]
            if vertical > 0 and energy_image[vertical - 1, horizontal] < _min_:
                vertical -= 1
            if vertical < energy_image.shape[0] and energy_image[vertical + 1, horizontal] < _min_:
                vertical += 1
            path_list.append(vertical)
    else:
        horizontal = find_entry(energy_image[0, :])[0]
        path_list.append(horizontal)
        for vertical in range(1, energy_image.shape[0]):
            _min_ = energy_image[vertical, horizontal]
            if horizontal > 0 and energy_image[vertical, horizontal - 1] < _min_:
                horizontal -= 1
            if horizontal < energy_image.shape[0] and energy_image[vertical, horizontal + 1] < _min_:
                horizontal += 1
            path_list.append(horizontal)
    return path_list


# TODO:ADD
def carving_one_time(raw_image, is_horizontal):
    raw_image.flags.writeable = True
    # ret = raw_image[:, :, :]
    path_line = find_path_line(energy(raw_image), is_horizontal)
    if is_horizontal:
        # for color_index in range(0, raw_image.shape[2]):
        for horizontal in range(0, raw_image.shape[1]):
            vertical = path_line[horizontal]
            raw_image[vertical - 1, horizontal, :] = \
                (raw_image[vertical - 1, horizontal, :] +
                 raw_image[vertical, horizontal, :]) / 2
            raw_image[vertical + 1, horizontal, :] = \
                (raw_image[vertical + 1, horizontal, :] +
                 raw_image[vertical, horizontal, :]) / 2
            raw_image[vertical:raw_image.shape[0] - 1, horizontal, :] \
                = raw_image[vertical + 1:raw_image.shape[0], horizontal, :]
    else:
        # for color_index in range(0, raw_image.shape[2]):
        for vertical in range(0, raw_image.shape[0]):
            horizontal = path_line[vertical]
            raw_image[vertical, horizontal - 1, :] = \
                (raw_image[vertical, horizontal - 1, :] +
                 raw_image[vertical, horizontal, :]) / 2
            raw_image[vertical, horizontal + 1, :] = \
                (raw_image[vertical, horizontal + 1, :] +
                 raw_image[vertical, horizontal, :]) / 2
            raw_image[vertical, horizontal:raw_image.shape[1] - 1, :] \
                = raw_image[vertical, horizontal + 1:raw_image.shape[1], :]

    return raw_image[0:raw_image.shape[0] - 1, 0:raw_image.shape[1] - 1, :]


def cross_carving(raw_image, times):
    ret = raw_image
    for i in tqdm(range(0, times)):
        ret = carving_one_time(ret, True)
        ret = carving_one_time(ret, False)
    return ret


if __name__ == '__main__':
    im = load_matrix("01.jpg")
    ima = cross_carving(im, 20)
    from matplotlib.pyplot import imsave

    imsave("s.jpg", ima)
