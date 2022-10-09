"""
Bochkin et al. QIP 2022
"""
import sys

sys.path.append(sys.path[0] + "/..")

import numpy as np

from quanty import matrix
from quanty.basis import ComputationBasis
from quanty.geometry import UniformChain
from quanty.hamiltonian import XX
from quanty.model.homo import Homogeneous
from quanty.state.coherence import coherence_matrix
from quanty.task.transfer import ZeroCoherenceTransfer

geometry = UniformChain()
model = Homogeneous(geometry)
hamiltonian = XX(model)

task = ZeroCoherenceTransfer.init_classic(
    hamiltonian, length=15, n_sender=3, n_ancillas=3, excitations=1
)
task.info()
print()


def loss_frobenius_qip2022(m):
    """Return loss function based on Frobenius norm (not true)"""
    diff = m.copy()
    diff[-1, -1] -= 1
    return -np.real(np.sqrt((diff.conj() @ diff).trace()))


test_state_params = [
    4.440892098500626e-16,
    0.331888943862158,
    0.0024748290211880726,
    -0.0009708346540924062,
    0.3325838276825169,
    0.3355272284553247,
]
test_state = np.array(
    task.sender_state.evalf(subs=dict(zip(task.sender_params, test_state_params))),
    dtype=complex,
)
test_loss = -0.8138161353375049
assert loss_frobenius_qip2022(test_state) == test_loss


task.fit_transmission_time()
assert task.transmission_time == 17.27988
print(f"Fitted transmission time: {task.transmission_time}")

print("Find optimal perfect transferred state...")
method, method_kwargs = "brute_random", {"no_local_search": False, "maxiter": 1}
# method, method_kwargs = "dual_annealing", {"no_local_search": True, "maxiter": 1000}
task.fit_transfer(loss=loss_frobenius_qip2022, method=method, method_kwargs=method_kwargs)
residual_max = np.max(np.abs(task.perfect_transferred_state_residuals()))
print("Receiver extra elements residual: ", residual_max)

loss = loss_frobenius_qip2022(np.array(task.perfect_transferred_state(), dtype=complex))
print(f"Loss based on Frobenius norm: {loss} is a {loss / test_loss:.0%} of best")
