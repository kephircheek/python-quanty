import numpy as np
from scipy import optimize


def brute_random(
    func,
    ranges,
    no_local_search=False,
    maxiter=10,
    callback=None,
    verbose=False,
    local_method_kwargs={},
    workers=1,
):
    min_value = None
    x_optimal = None
    for n_iter in range(maxiter):
        x = np.array([(up - down) * np.random.random() + down for down, up in ranges])
        if no_local_search:
            instant_value = func(x)
        else:
            res = optimize.minimize(func, x, **local_method_kwargs)
            x = res.x
            instant_value = res.fun

        if verbose:
            print(f"iter={n_iter} actual={instant_value:.3e}", end=" ")
            if min_value is not None:
                print(f"min={min_value:.3e}", end=" ")

        if min_value is None or instant_value < min_value:
            min_value = instant_value
            x_optimal = x
            if verbose and n_iter > 0:
                print("(new global minimum)", end="")

        if verbose:
            print()

        if callback is not None:
            if callback(x, instant_value) is False:  # maybe True ?!
                break

    return optimize.OptimizeResult(x=x_optimal, fun=min_value)
