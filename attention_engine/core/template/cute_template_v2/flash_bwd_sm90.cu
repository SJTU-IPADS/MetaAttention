// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
#include "flash_bwd_launch_template.h"

template<>
void run_mha_bwd_<90, {{cutlass_dtype}}, {{dim_round}}, false>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_hdim{{dim_round}}<90, {{cutlass_dtype}}, false>(params, stream);
}
