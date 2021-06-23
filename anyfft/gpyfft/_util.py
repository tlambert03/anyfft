import pyopencl as cl


def score_device(dev: cl.Device) -> float:
    score = 4e12 if dev.type == cl.device_type.GPU else 2e12
    score += dev.get_info(cl.device_info.GLOBAL_MEM_SIZE)
    return score


def get_context():
    devices = [d for platform in cl.get_platforms() for d in platform.get_devices()]
    if not devices:
        raise RuntimeError("No cl devices found.")
    best = sorted(devices, key=score_device, reverse=True)[0]
    return cl.Context(devices=[best])
