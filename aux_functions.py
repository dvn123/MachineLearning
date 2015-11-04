import imghdr
import math
import struct


def get_image_size(f_name):
    with open(f_name, 'rb') as f_handle:
        head = f_handle.read(24)
        if len(head) != 24:
            return
        if imghdr.what(f_name) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        else:
            return
        return width, height


def get_percentage_of_list(list, percentage):
    size = math.floor(len(list) * percentage)
    return list[:size]
