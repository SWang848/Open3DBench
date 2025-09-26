# Post-placement timing analysis script
utl::set_metrics_stage "post_place_timing__{}"
source $::env(SCRIPTS_DIR)/load.tcl
load_design 3_place.odb 3_place.sdc "Starting post-placement timing analysis"

puts "\n=========================================================================="
puts "POST-PLACEMENT TIMING ANALYSIS"
puts "=========================================================================="

# Estimate parasitics for placement-based timing analysis
puts "Estimating parasitics for placement..."
estimate_parasitics -placement

# Set ideal clocks for placement timing analysis
puts "Using ideal clocks for post-placement timing analysis..."

# Generate comprehensive timing reports
puts "\n=========================================================================="
puts "TIMING ANALYSIS SUMMARY"
puts "--------------------------------------------------------------------------"

puts "\nWorst Negative Slack (Setup):"
report_worst_slack -max -digits 3

puts "\nWorst Negative Slack (Hold):"
report_worst_slack -min -digits 3

puts "\nTotal Negative Slack:"
report_tns -digits 3

puts "\n=========================================================================="
puts "DETAILED TIMING REPORTS"
puts "--------------------------------------------------------------------------"

# Generate detailed timing reports
puts "\nSetup timing check (worst 5 paths):"
report_checks -path_delay max -fields {slew cap input nets fanout} -format full_clock_expanded -group_count 5

puts "\nHold timing check (worst 5 paths):"
report_checks -path_delay min -fields {slew cap input nets fanout} -format full_clock_expanded -group_count 5

# Report violations
puts "\n=========================================================================="
puts "VIOLATION SUMMARY"
puts "--------------------------------------------------------------------------"

puts "Setup violation count: [sta::endpoint_violation_count max]"
puts "Hold violation count: [sta::endpoint_violation_count min]"
puts "Max slew violation count: [sta::max_slew_violation_count]"
puts "Max capacitance violation count: [sta::max_capacitance_violation_count]"
puts "Max fanout violation count: [sta::max_fanout_violation_count]"

# Write detailed reports to files
puts "\n=========================================================================="
puts "WRITING DETAILED REPORTS TO FILES"
puts "--------------------------------------------------------------------------"

# Write comprehensive timing reports to files
puts "Writing setup timing report..."
report_checks -path_delay max -fields {slew cap input nets fanout} -format full_clock_expanded -group_count 500000 > $::env(REPORTS_DIR)/post_place_timing_setup.rpt

# puts "Writing hold timing report..."
# report_checks -path_delay min -fields {slew cap input nets fanout} -format full_clock_expanded -group_count 100 > $::env(REPORTS_DIR)/post_place_timing_hold.rpt

# puts "Writing unconstrained paths report..."
# report_checks -unconstrained -fields {slew cap input nets fanout} -format full_clock_expanded > $::env(REPORTS_DIR)/post_place_timing_unconstrained.rpt

# puts "Writing violated paths summary..."
# report_checks -group_count 1000 > $::env(REPORTS_DIR)/post_place_timing_violations.rpt

# Use the metrics reporting for consistency with rest of flow
source $::env(SCRIPTS_DIR)/report_metrics.tcl
report_metrics "post-placement timing analysis" false false

puts "\n=========================================================================="
puts "POST-PLACEMENT TIMING ANALYSIS COMPLETE"
puts "Reports written to: $::env(REPORTS_DIR)/post_place_timing_*.rpt"
puts "=========================================================================="

# Save timing analysis checkpoint
if {![info exists save_checkpoint] || $save_checkpoint} {
  write_db $::env(RESULTS_DIR)/3_6_post_place_timing.odb
  write_sdc $::env(RESULTS_DIR)/3_6_post_place_timing.sdc
} 

# puts "\n=========================================================================="
# puts "HIGH FANOUT NET ANALYSIS"
# puts "--------------------------------------------------------------------------"

# Report high fanout nets
# puts "Reporting high fanout nets (fanout > 50)..."
# report_net -connections -verbose -min_fanout 50

# Report net length distribution
# puts "\nNet length distribution..."
# report_net -verbose -sort_by_length

# Report placement density around high fanout drivers
# puts "\nPlacement density analysis around critical drivers..."
# report_placement_density 