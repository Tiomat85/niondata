import numpy
import scipy.fft
import scipy.ndimage
import typing

from nion.data import Image

import cupy as cp
import cupyx as cpx
from cupyx.scipy.ndimage import fourier_uniform

_ShapeType = Image.ShapeType
_ImageDataType = Image._ImageDataType

def normalized_corr_gpu(imagesequence: typing.Sequence[_ImageDataType], template: _ImageDataType) -> typing.Sequence[_ImageDataType]:

    imagestack = numpy.stack(imagesequence, axis=0)
    # size_per_elem = 4
    (device_memory_info_free, device_memory_info_total) = cp.cuda.runtime.memGetInfo()
    usable_device_memory = device_memory_info_free * 0.8  # Lets not saturate the card, and give some space to deal with fragmentation
    page_size_elem = 20 * 1024 * 1024
    page_size_slices = int(numpy.floor(page_size_elem / template.size))

    start_used_memory = device_memory_info_total - device_memory_info_free
    max_used_memory = start_used_memory
    max_used_memory = max(max_used_memory, cp.cuda.runtime.memGetInfo()[1] - cp.cuda.runtime.memGetInfo()[0])

    # 24
    normalized_template = template - numpy.mean(template)

    np_signal_scale = template.size * numpy.sum(normalized_template ** 2)
    np_signal_slices = imagestack.shape[0]

    # template stuff first
    cp_normalized_template = cp.asarray(normalized_template)
    # 25
    cp_normalized_template_conj = cp.conj(cp.fft.fft2(cp_normalized_template))
    del cp_normalized_template
    # return cp.asnumpy(cp_normalized_template_conj)

    result = numpy.empty_like(imagestack)  # np.copy(np_signal)

    for slice_index in range(0, np_signal_slices, page_size_slices):
        print("Slice")
        end_index = min(slice_index + page_size_slices, np_signal_slices)
        num_working_slices = end_index - slice_index
        # Transfer the signal to the GPU
        cp_image = None
        cp_image = cp.asarray(imagestack[slice_index:end_index, :, :])

        # Perform FFT on the GPU
        # 26
        cp_fft_image = cp.fft.fft2(cp_image, axes=(1, 2))

        # 32, at the correlation point
        cp_fft_corr = cp_fft_image * cp_normalized_template_conj[cp.newaxis, :, :]

        cp_fourier_uniform = cp.ndarray(cp_fft_image.shape, cp.complex128)
        for slice in range(num_working_slices):
            cp_fourier_uniform[slice, :, :] = cpx.scipy.ndimage.fourier_uniform(cp_fft_image[slice, :, :],
                                                                                template.shape)

        max_used_memory = max(max_used_memory, cp.cuda.runtime.memGetInfo()[1] - cp.cuda.runtime.memGetInfo()[0])
        del cp_fft_image

        cp_image_squared_means = cp.fft.ifft2(cp_fourier_uniform, axes=(1, 2))
        max_used_memory = max(max_used_memory, cp.cuda.runtime.memGetInfo()[1] - cp.cuda.runtime.memGetInfo()[0])
        del cp_fourier_uniform
        cp_image_squared_means = cp.real(cp_image_squared_means)
        cp_image_squared_means = cp.square(cp_image_squared_means)

        # 27
        cp_image = cp.square(cp_image)
        cp_fft_image_squared = cp.fft.fft2(cp_image)
        max_used_memory = max(max_used_memory, cp.cuda.runtime.memGetInfo()[1] - cp.cuda.runtime.memGetInfo()[0])
        del cp_image

        # fourier uniform
        # 28
        cp_fft_image_squared_means = cp.ndarray(cp_fft_image_squared.shape, cp.complex128)
        for slice in range(num_working_slices):
            cp_fft_image_squared_means[slice, :, :] = cpx.scipy.ndimage.fourier_uniform(
                cp_fft_image_squared[slice, :, :], template.shape)

        max_used_memory = max(max_used_memory, cp.cuda.runtime.memGetInfo()[1] - cp.cuda.runtime.memGetInfo()[0])
        del cp_fft_image_squared

        # 34
        shift = (int(-1 * (imagestack.shape[0] - 1) / 2), int(-1 * (imagestack.shape[1] - 1) / 2))

        # 37
        cp_image_variance = cp.real(cp.fft.ifft2(cp_fft_image_squared_means, axes=(1, 2)))
        max_used_memory = max(max_used_memory, cp.cuda.runtime.memGetInfo()[1] - cp.cuda.runtime.memGetInfo()[0])
        del cp_fft_image_squared_means
        cp_image_variance = cp.subtract(cp_image_variance, cp_image_squared_means)
        max_used_memory = max(max_used_memory, cp.cuda.runtime.memGetInfo()[1] - cp.cuda.runtime.memGetInfo()[0])
        del cp_image_squared_means

        # 35
        cp_fft_corr = cp.fft.ifft2(cp_fft_corr, axes=(1, 2))
        cp_fft_corr = cp.real(cp_fft_corr)
        cp_corr = cp.roll(cp_fft_corr, shift=shift, axis=(1, 2))
        max_used_memory = max(max_used_memory, cp.cuda.runtime.memGetInfo()[1] - cp.cuda.runtime.memGetInfo()[0])
        del cp_fft_corr

        # 38
        cp_denom = cp_image_variance * np_signal_scale
        max_used_memory = max(max_used_memory, cp.cuda.runtime.memGetInfo()[1] - cp.cuda.runtime.memGetInfo()[0])
        del cp_image_variance
        cp_denom_max = cp.max(cp_denom)
        # cp_denom[cp_denom < 0] = cp_denom_max
        cp_denom = cp.where(cp_denom < 0, cp_denom_max, cp_denom)
        max_used_memory = max(max_used_memory, cp.cuda.runtime.memGetInfo()[1] - cp.cuda.runtime.memGetInfo()[0])
        del cp_denom_max

        # 40
        cp_denom = cp.sqrt(cp_denom)
        cp_corr = cp_corr / cp_denom[cp.newaxis, :, :]
        max_used_memory = max(max_used_memory, cp.cuda.runtime.memGetInfo()[1] - cp.cuda.runtime.memGetInfo()[0])
        del cp_denom

        # return cp.asnumpy(cp_corr)
        cp_corr = cp.where(cp_corr > 1.1, 0, cp_corr)
        result[slice_index:end_index, :, :] = cp.asnumpy(cp_corr)
        max_used_memory = max(max_used_memory, cp.cuda.runtime.memGetInfo()[1] - cp.cuda.runtime.memGetInfo()[0])
        del cp_corr
        fft_cache = cp.fft.config.get_plan_cache()
        fft_cache.clear()
        cp.cuda.runtime.deviceSynchronize()
        # result.append(cp.asnumpy(cp_corr))

    del cp_normalized_template_conj

    fft_cache = cp.fft.config.get_plan_cache()
    fft_cache.clear()

    (device_memory_info_free_end, device_memory_info_total_2nd) = cp.cuda.runtime.memGetInfo()
    usable_device_memory_end = device_memory_info_free_end  # Lets not saturate the card, and give some space to deal with fragmentation
    # print(usable_device_memory_end)
    #print(max_used_memory)

    return result



def normalized_corr(image: _ImageDataType, template: _ImageDataType) -> _ImageDataType:
    """
    Correctly normalized template matching by cross-correlation. The result should be the same as what you get from
    openCV's "match_template" function with method set to "ccoeff_normed", except for the output shape, which will
    be image.shape here (as opposed to openCV, where only the valid portion of the image is returned).
    Used ideas from here:
    http://scribblethink.org/Work/nvisionInterface/nip.pdf (which is an extended version of this paper:
    J. P. Lewis, "FastTemplateMatching", Vision Interface, p. 120-123, 1995)
    """
    template = template.astype(numpy.float64)
    image = image.astype(numpy.float64)
    normalized_template = template - numpy.mean(template)
    # inverting the axis of a real image is the same as taking the conjugate of the fourier transform
    fft_normalized_template_conj = scipy.fft.fft2(normalized_template[::-1, ::-1], s=image.shape)
    fft_image = scipy.fft.fft2(image)
    fft_image_squared = scipy.fft.fft2(image ** 2)
    fft_image_squared_means = scipy.ndimage.fourier_uniform(fft_image_squared, template.shape)
    image_means_squared = (scipy.fft.ifft2(
        scipy.ndimage.fourier_uniform(fft_image, template.shape)).real) ** 2
    # only normalizing the template is equivalent to normalizing both (see paper in docstring for details)
    fft_corr = fft_image * fft_normalized_template_conj
    # we need to shift the result back by half the template size
    shift = (int(-1 * (template.shape[0] - 1) / 2), int(-1 * (template.shape[1] - 1) / 2))
    corr = numpy.roll(scipy.fft.ifft2(fft_corr).real, shift=shift, axis=(0, 1))
    # use Var(X) = E(X^2) - E(X)^2 to calculate variance
    image_variance = scipy.fft.ifft2(fft_image_squared_means).real - image_means_squared
    denom = image_variance * template.size * numpy.sum(normalized_template ** 2)
    denom[denom < 0] = numpy.amax(denom)
    return typing.cast(_ImageDataType, corr / numpy.sqrt(denom))


def parabola_through_three_points(p1: typing.Tuple[int, int], p2: typing.Tuple[int, int], p3: typing.Tuple[int, int]) -> typing.Tuple[float, float, float]:
    """
    Calculates the parabola a*(x-b)**2+c through three points. The points should be given as (y, x) tuples.
    Returns a tuple (a, b, c)
    """
    # formula taken from http://stackoverflow.com/questions/4039039/fastest-way-to-fit-a-parabola-to-set-of-points
    # Avoid division by zero in calculation of s
    if p2[0] == p3[0]:
        temp = p2
        p2 = p1
        p1 = temp

    s = (p1[0] - p2[0]) / (p2[0] - p3[0])
    b = (-p1[1] ** 2 + p2[1] ** 2 + s * (p2[1] ** 2 - p3[1] ** 2)) / (2 * (-p1[1] + p2[1] + s * p2[1] - s * p3[1]))
    a = (p1[0] - p2[0]) / ((p1[1] - b) ** 2 - (p2[1] - b) ** 2)
    c = p1[0] - a * (p1[1] - b) ** 2
    return (a, b, c)


def find_ccorr_max(ccorr: _ImageDataType) -> typing.Tuple[int, typing.Optional[float], typing.Optional[typing.Tuple[float, ...]]]:
    max_pos: typing.Tuple[int, ...] = typing.cast(typing.Tuple[int, ...], numpy.unravel_index(numpy.argmax(ccorr), ccorr.shape))
    if ccorr.ndim == 2:
        if typing.cast(_ImageDataType, numpy.array(max_pos) < numpy.array((1,1))).any() or typing.cast(_ImageDataType, numpy.array(max_pos) > numpy.array(ccorr.shape) - 2).any():
            return 1, ccorr[max_pos], tuple(float(p) for p in max_pos)
    elif ccorr.ndim == 1:
        if max_pos[0] < 1 or max_pos[0] > ccorr.shape[0] - 2:
            return 1, ccorr[max_pos], tuple(float(p) for p in max_pos)
    else:
        return 1, None, None

    if ccorr.ndim == 2:
        max_y = ccorr[max_pos[0]-1:max_pos[0]+2, max_pos[1]]
        parabola_y = parabola_through_three_points((max_y[0], max_pos[0]-1),
                                                   (max_y[1], max_pos[0]  ),
                                                   (max_y[2], max_pos[0]+1))
        max_x = ccorr[max_pos[0], max_pos[1]-1:max_pos[1]+2]
        parabola_x = parabola_through_three_points((max_x[0], max_pos[1]-1),
                                                   (max_x[1], max_pos[1]  ),
                                                   (max_x[2], max_pos[1]+1))
        return 0, ccorr[max_pos], (parabola_y[1], parabola_x[1])

    max_ = ccorr[max_pos[0]-1:max_pos[0]+2]
    parabola = parabola_through_three_points((max_[0], max_pos[0]-1),
                                             (max_[1], max_pos[0]  ),
                                             (max_[2], max_pos[0]+1))

    return 0, ccorr[max_pos], (parabola[1],)


def match_template(image: _ImageDataType, template: _ImageDataType) -> _ImageDataType:
    ccorr = normalized_corr(image, template)
    ccorr[ccorr > 1.1] = 0
    return ccorr

def match_template_gpu(image: typing.Sequence[_ImageDataType], template: _ImageDataType) -> typing.Sequence[_ImageDataType]:
    xcorr = normalized_corr_gpu(image, template)
    result = []
    for item in xcorr:
        item[item > 1.1] = 0
        result.append(item)
    return result
