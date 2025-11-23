import math
import numpy as np


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def linear_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array([base_value - (base_value - final_value) * i / len(iters) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def piecewise_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                        start_warmup_value=0, milestones=None, gamma=0.8):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    if milestones is None:
        total_iters = epochs * niter_per_ep
        milestones = [int(total_iters * i // 10) for i in range(1, 10)]

    schedule_list = []
    if len(iters) > 0:
        schedule_list.append(base_value)
        for i in range(1, len(iters)):
            if i in milestones:
                schedule_list.append(schedule_list[-1] * gamma)
            else:
                schedule_list.append(schedule_list[-1])

    schedule = np.concatenate((warmup_schedule, np.array(schedule_list)))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
