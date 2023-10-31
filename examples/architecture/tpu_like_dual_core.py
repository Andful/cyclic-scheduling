from stream.inputs.examples.hardware.cores.TPU_like import get_core as get_tpu_core
from stream.inputs.examples.hardware.cores.pooling import get_core as get_pooling_core
from stream.inputs.examples.hardware.cores.simd import get_core as get_simd_core
from stream.inputs.examples.hardware.cores.offchip import get_offchip_core
from stream.inputs.examples.hardware.nocs.mesh_2d import get_2d_mesh
from stream.classes.hardware.architecture.accelerator import Accelerator

cores = [get_tpu_core(id) for id in range(2)]  # 3 identical cores
offchip_core_id = 3
offchip_core = get_offchip_core(id=offchip_core_id)

cores_graph = get_2d_mesh(cores, 1, 2, 64, 0, None, None, offchip_core)

accelerator = Accelerator(
    "TPU-like-dual-core", cores_graph, offchip_core_id=offchip_core_id
)
