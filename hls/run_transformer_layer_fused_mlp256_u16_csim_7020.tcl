set script_dir [file dirname [file normalize [info script]]]
cd $script_dir
open_project -reset vit_transformer_layer_fused_mlp256_prj
set_top vit_transformer_layer_fused
set cflags "-DVIT_FUSED_MLP_DIM=256 -DVIT_FUSED_MAC_UNROLL=16"
add_files -cflags $cflags vit_transformer_layer_fused.cpp
add_files -cflags $cflags vit_transformer_layer_fused.h
add_files -tb -cflags $cflags tb_vit_transformer_layer_fused.cpp
open_solution "solution_unroll16" -flow_target vivado
set_part {xc7z020-clg400-1}
create_clock -period 16.0 -name default
csim_design
exit
