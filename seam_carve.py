import random

import numpy
from tqdm import tqdm
from matplotlib.pyplot import imsave
from random import randint
from matplotlib.image import imread


def load_matrix(path):
    return imread(path)


def to_grayscale_image(raw_image_matrix, parameters=(0.299, 0.587, 0.114)):
    return numpy.dot(raw_image_matrix[..., :3], list(parameters))


def shift_pixel(image_matrix, direct_x, direct_y, step=1):
    shape = image_matrix.shape
    shifted_image = numpy.zeros((shape[0] + 2 * step, shape[1] + 2 * step))
    shifted_image[step + direct_x: shape[0] + step + direct_x, step + direct_y: shape[1] + step + direct_y] \
        = image_matrix[:, :]
    return shifted_image


def energy(raw_image_matrix, step=1):
    gray_scale = to_grayscale_image(raw_image_matrix)  
    gray_scale = gray_scale / numpy.max(gray_scale)  
    orthogonal_shift_direction = [(-step, 0), (step, 0), (0, step), (0, -step)]
    diagonal_shift_direction = [(-step, -step), (step, -step), (-step, step), (step, step)]
    energy_image = numpy.zeros(shape=(raw_image_matrix.shape[0] + 2 * step, raw_image_matrix.shape[1] + 2 * step))
    for _dir_ in orthogonal_shift_direction:
        energy_image += shift_pixel(gray_scale[:, :], _dir_[0], _dir_[1], step)
    energy_image = energy_image[
                   step:energy_image.shape[0] - step,
                   step:energy_image.shape[1] - step]  
    return gray_scale * 4 - energy_image  


def find_entries(line):
    pass


def find_entry(line):
    # probablity algorithm
    try:
        p_matrix = (numpy.max(line) - line)/sum((numpy.max(line) - line))
        return numpy.random.choice(range(len(line)), p=p_matrix)
    except ValueError:
        # greedy algorithm
        argmin = numpy.where(line == numpy.min(line))
        ret = []
        for i in range(0, len(argmin)):
            ret.append(argmin[i][randint(0, len(argmin) - 1)])
        return ret[0]


def find_next(energy_line, entry_point_index):
    if random.uniform(0, 1) < 0.000:
        return find_entry(energy_line)
    if entry_point_index - 1 <= 0 or entry_point_index + 1 >= len(energy_line) - 1: 
        return find_entry(energy_line)
    return entry_point_index + find_entry(energy_line[
                      max(entry_point_index - 1, 0):
                      min(entry_point_index + 1, len(energy_line) - 1)])


def find_left_or_right(max_index, entry_point_index, left=True):
    if entry_point_index >= max_index - 1:
        return entry_point_index - 1
    if entry_point_index <= 0:
        return entry_point_index + 1
    return (entry_point_index - 1) if left else (entry_point_index + 1)


def carve_1_step(raw_image, is_horizontal_carve, add_mode, energy_step=1):
    carve_index_record = []

    if is_horizontal_carve:
        raw_image = raw_image.swapaxes(0, 1)

    if add_mode:
        ret = numpy.zeros((raw_image.shape[0], raw_image.shape[1] + 1, raw_image.shape[2]))
        ret2 = numpy.zeros((raw_image.shape[0], raw_image.shape[1] + 1, raw_image.shape[2]))
    else:
        ret = numpy.zeros((raw_image.shape[0], raw_image.shape[1] - 1, raw_image.shape[2]))

    energy_image = energy(raw_image, energy_step)

    to_carve_index = 0

    for horizontal_index in range(0, energy_image.shape[0]):
        if horizontal_index == 0:
            to_carve_index = find_entry(energy_image[horizontal_index, :])
        else:
            to_carve_index = find_next(energy_image[horizontal_index, :], to_carve_index)
        raw_image_current_line = raw_image[horizontal_index, :, :]

        if add_mode:
            left = True if random.uniform(0, 1) > 0.5 else False
            interpolation_point = numpy.zeros((raw_image.shape[2]))
            tmp_index = find_left_or_right(raw_image.shape[1], to_carve_index, left)
            for j in range(0, raw_image.shape[2]):
                interpolation_point[j] = \
                    (float(raw_image[horizontal_index, tmp_index, j]) + float(raw_image[horizontal_index, to_carve_index, j]))/2
            if left:
                raw_image_current_line = numpy.insert(raw_image_current_line, to_carve_index, interpolation_point, 0)
            else:
                raw_image_current_line = numpy.insert(raw_image_current_line, to_carve_index+1, interpolation_point, 0)
            pass
        else:
            raw_image_current_line = numpy.delete(raw_image_current_line, to_carve_index, 0)
        ret[horizontal_index] = raw_image_current_line

    if is_horizontal_carve:
        ret = ret.swapaxes(0, 1)
    return ret.astype(numpy.uint8), carve_index_record


def cross_carving(raw_image: numpy.ndarray, times: int):
    add_mode = True
    ret = raw_image

    for _ in tqdm(range(0, times)):
        ret, _ = carve_1_step(ret, True, add_mode, 1)
        ret, _ = carve_1_step(ret, False, add_mode, 1)
    return ret


if __name__ == '__main__':
    x = cross_carving(load_matrix("01.jpg"), 100)
    imsave("s.jpg", x)
