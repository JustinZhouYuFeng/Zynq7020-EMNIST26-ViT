set build_root "D:/FPGA_ViT_MLP256_Build/zynq_vit_qkv_system"
set proj_file "$build_root/zynq_vit_qkv_system.xpr"
set report_dir "$build_root/reports_mlp256_u16_rev25"
set xsa_out "$build_root/design_1_wrapper_mlp256_u16_rev25.xsa"

file mkdir $report_dir
open_project $proj_file
open_run impl_1
report_utilization -file "$report_dir/utilization_placed.rpt"
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose \
  -max_paths 10 -input_pins -file "$report_dir/timing_summary.rpt"
write_hw_platform -fixed -include_bit -force -file $xsa_out
puts "MLP256_U16_REV25_ARTIFACTS_EXPORTED"
puts "MLP256_U16_REV25_XSA=$xsa_out"
close_project
