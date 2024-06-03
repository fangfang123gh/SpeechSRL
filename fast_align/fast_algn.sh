build_new/fast_align -i align_data -d -v -o -p fwd_params -q 0.1 -I 20 >fwd_align 2>fwd_err
build_new/fast_align -i align_data -r -d -v -o -p rev_params -q 0.1 -I 20 >rev_align 2>rev_err
