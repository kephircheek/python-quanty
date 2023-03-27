import numpy as np
import timeti

from quanty.geometry import UniformChain, ZigZagChain
from quanty.hamiltonian import XX, XXZ
from quanty.model.homo import Homogeneous
from quanty.task.transfer import ZeroCoherenceTransfer
from quanty.task.transfer_ import (
    FitTransmissionTimeTask,
    TransferZQCAlongChain,
    TransferZQCPerfectlyTask,
)

h_angle = np.pi / 2
width = 1
length = 13
n_sender = 3
excitations = 3

geometry = ZigZagChain.from_two_chain(2, width)
model = Homogeneous(geometry, h_angle=h_angle, norm_on=None)
hamiltonian = XXZ(model)
problem = TransferZQCAlongChain.init_classic(
    hamiltonian=hamiltonian, length=length, n_sender=n_sender, excitations=excitations
)

with timeti.profiler():
    for tt in [10, 50, 100]:
        task = TransferZQCPerfectlyTask(problem, transmission_time=tt)
        r = task.run()
