"""
Group of Utility functions required by Sympyle
"""


def get_broadcast_axes(broadcasted_arr_shape, broadcasting_arr_shape):
    """
    Get the axes over which an array was broadcast into

    :param broadcasted_arr_shape: The "output" array that was the result of
                                  two input arrays
    :param broadcasting_arr_shape: One of the input arrays
    :return: A tuple containing all the dimensions that the input
            array(param 1) was broadcast into to give the output array(param 0)
    """
    dim1 = dim2 = 0
    broadcast_dims_list = []
    for i, (dim1, dim2) in enumerate(
            zip(broadcasting_arr_shape.shape[::-1],
                broadcasted_arr_shape.shape[::-1])):

        if dim2 != dim1 and dim1 != 1:
            break

        if dim1 == 1:
            broadcast_dims_list.append(
                    len(broadcasted_arr_shape.shape) - i - 1)

    return tuple(broadcast_dims_list)
