import itertools
import torch

# MUTlass 3.x tile configurations: (TileM, TileN, TileK)
candidates = {
    'TileM': [128, 256],
    'TileN': [128, 256],
    'TileK': [32, 64],
}

all_combinations = list(itertools.product(*candidates.values()))
valid_combinations = []
for combo in all_combinations:
    tm, tn, tk = combo
    if tm == 256 and tn == 256:
        continue
    valid_combinations.append(combo)

# 生成字典：key为参数元组，value为index
index_dict = {combo: idx for idx, combo in enumerate(valid_combinations)}

torch.save(
    index_dict,
    "../configs/AlgoDict.pt"
)

with open('../src/inc/gemm_instances.inc', 'w') as f_inc, \
     open('../src/tiling/gemm_tiling.cuh', 'w') as f_table:

    f_table.write("#include \"gemm_dispatcher.h\"\n\n")
    f_table.write("GemmFuncPtr gemm_func_table[] = {\n")

    for combo in valid_combinations:
        args = ', '.join(map(str, combo))
        f_inc.write(f'MUTLASS_GEMM_INIT({args});\n')
        f_table.write(f"    &mutlass_gemm<{args}>,\n")

    f_table.write("};\n")

with open('../src/inc/signal_instances.inc', 'w') as f_inc, \
     open('../src/tiling/signal_tiling.cuh', 'w') as f_table:

    f_table.write("#include \"gemm_dispatcher.h\"\n\n")
    f_table.write("SignalFuncPtr signal_func_table[] = {\n")

    for combo in valid_combinations:
        args = ', '.join(map(str, combo))
        f_inc.write(f'MUTLASS_SIGNAL_INIT({args});\n')
        f_table.write(f"    &mutlass_gemm_signal<{args}>,\n")

    f_table.write("};\n")

with open('../src/inc/scatter_instances.inc', 'w') as f_inc, \
     open('../src/tiling/scatter_tiling.cuh', 'w') as f_table:

    f_table.write("#include \"gemm_dispatcher.h\"\n\n")
    f_table.write("ScatterFuncPtr scatter_func_table[] = {\n")

    for combo in valid_combinations:
        args = ', '.join(map(str, combo))
        f_inc.write(f'MUTLASS_SCATTER_INIT({args});\n')
        f_table.write(f"    &mutlass_gemm_scatter<{args}>,\n")

    f_table.write("};\n")