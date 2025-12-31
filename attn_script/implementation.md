# AttentionEngine iMPLEMENTATION

- decider: decide whether to fuse according to qkv shape
- lower: define the computing primitive: `reduce_max`, `reduce_sum`, `elementwise`, generate forward&backward computation graph
- template: define the code template for attention & linear attention; define the place where custom code is needed.
- codegen: generate the custom code needed for the template.
- autotuner: parallel compile and profile the template configs, return the best config.

# examples
attention: softmax attention `attn_script/mha.py`; retnet parallel `attn_script/retention.py`; sigmoid attention `attn_script/sigmoidattn.py`;  
linear attention: `attn_script/mamba2_ngroup1.py` `attn_script/simple_gla.py`

# Code Details

## attention operator Interface 
`python/attn_engine/attn_engine.py` `python/attn_engine/linear_attn_engine.py`


## FrontEnd Computation Graph&Op
`python/core/core.py` `python/core/graph.py`
define the operator, and computing graph.

## decider

`python/autotuner/decider.py` `python/autotuner/arch`

return isneedtofuse: True/False; configs: tile, memory

## codegen & template
- tl attn
`python/core/lower.py` `python/tl_gen.py` `python/tl_template/attn`
template defines the place where custom code is needed.
lower defines the code to insert into the template.
tl_gen generates the tl expression.

- tl linear attn
`python/core/lower_linear.py` `python/tl_gen.py` `python/tl_template/linear`

- cute attn
`python/core/lower_cute.py` `python/cute_gen.py` `python/cute_template`

## autotuner
`attnfwd_tunner_engine2.py`
parallel compile and profile the template configs, return the best config.
