set script_dir [file dirname [file normalize [info script]]]
cd $script_dir
open_project vit_transformer_layer_fused_mlp256_prj
open_solution "solution_unroll16" -flow_target vivado
config_export -format=ip_catalog
config_export -rtl=verilog
export_design -format ip_catalog
exit
