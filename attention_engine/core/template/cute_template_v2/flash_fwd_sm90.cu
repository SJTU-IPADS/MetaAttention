// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

#include "flash_fwd_launch_template.h"

template void run_mha_fwd_<90, {{cutlass_dtype}}, {{dimqk}}, {{dimv}}, false, false, false, false>(Flash_fwd_params &params, cudaStream_t stream);
