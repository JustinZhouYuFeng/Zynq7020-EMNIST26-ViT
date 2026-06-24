set repo_root "C:/Users/19571/Desktop/FPGA_ViT_Project"
set build_root "D:/FPGA_ViT_MLP256_Build/zynq_vit_qkv_system"
set proj_file "$build_root/zynq_vit_qkv_system.xpr"
set ip_repo "$repo_root/hls/vit_transformer_layer_fused_mlp256_prj/solution_unroll16/impl/ip"
set report_dir "$build_root/reports_mlp256_u16_rev25"
set xsa_out "$build_root/design_1_wrapper_mlp256_u16_rev25.xsa"

file mkdir $report_dir
open_project $proj_file
set_property ip_repo_paths [list $ip_repo] [current_project]
update_ip_catalog

set bd_file [lindex [get_files -quiet */design_1.bd] 0]
if {$bd_file eq ""} {
  error "design_1.bd not found"
}
open_bd_design $bd_file

set fused_ips [get_ips -quiet -all *vit_tlayer_fused*]
if {[llength $fused_ips] == 0} {
  error "Fused transformer IP instance not found"
}
report_ip_status -file "$report_dir/ip_status_before.txt"
upgrade_ip $fused_ips
report_ip_status -file "$report_dir/ip_status_after.txt"

validate_bd_design
save_bd_design
generate_target all $bd_file
make_wrapper -files $bd_file -top -import
update_compile_order -fileset sources_1

reset_run synth_1
reset_run impl_1
launch_runs synth_1 -jobs 6
wait_on_run synth_1
if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} {
  error "synth_1 failed: [get_property STATUS [get_runs synth_1]]"
}

launch_runs impl_1 -to_step write_bitstream -jobs 6
wait_on_run impl_1
if {[get_property STATUS [get_runs impl_1]] ne "write_bitstream Complete!"} {
  error "impl_1 failed: [get_property STATUS [get_runs impl_1]]"
}

open_run impl_1
report_utilization -file "$report_dir/utilization_placed.rpt"
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose \
  -max_paths 10 -input_pins -file "$report_dir/timing_summary.rpt"
write_hw_platform -fixed -include_bit -force -file $xsa_out
puts "MLP256_U16_REV25_BUILD_COMPLETE"
puts "MLP256_U16_REV25_XSA=$xsa_out"
close_project
