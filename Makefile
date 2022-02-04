CXX = dpcpp
CXXFLAGS = -Wall -std=c++20
OPTFLAGS = -O3
IFLAGS = -I./include

FPGA_EMU_FLAGS = -DFPGA_EMU -fintelfpga

# Another option is using `intel_a10gx_pac:pac_a10` as FPGA board and if you do so ensure that
# on Intel Devcloud you use `fpga_runtime:arria10` as offload target
#
# Otherwise if you stick to Stratix 10 board, consider offloading to `fpga_runtime:stratix10` VMs
# on Intel Devcloud
FPGA_OPT_FLAGS = -DFPGA_HW -fintelfpga -fsycl-link=early -Xshardware -Xsboard=intel_s10sx_pac:pac_s10

# Consider enabing -Xsprofile, when generating h/w image, so that execution can be profile
# using Intel Vtune
FPGA_HW_FLAGS = -DFPGA_HW -fintelfpga -Xshardware -Xsboard=intel_s10sx_pac:pac_s10

all: fpga_emu_test

fpga_emu_test: ./test/fpga_emu.out
	./$<

./test/fpga_emu.out: test/main.cpp include/*.hpp
	$(CXX) $(CXXFLAGS) $(IFLAGS) $(OPTFLAGS) $(FPGA_EMU_FLAGS) $< -o $@

fpga_opt_test:
	# output not supposed to be executed, instead consume report generated
	# inside `test/fpga_opt.prj/reports/` diretory
	$(CXX) $(CXXFLAGS) $(IFLAGS) $(OPTFLAGS) $(FPGA_OPT_FLAGS) test/main.cpp -o test/fpga_opt.a

fpga_hw_test:
	$(CXX) $(CXXFLAGS) $(IFLAGS) $(OPTFLAGS) $(FPGA_HW_FLAGS) -reuse-exe=test/fpga_hw.out test/main.cpp -o test/fpga_hw.out

fpga_emu_bench:
	# you should not rely on these numbers !
	$(CXX) $(CXXFLAGS) $(IFLAGS) $(OPTFLAGS) $(FPGA_EMU_FLAGS) benchmark/main.cpp -o benchmark/fpga_emu.out

fpga_hw_bench:
	$(CXX) $(CXXFLAGS) $(IFLAGS) $(OPTFLAGS) $(FPGA_HW_FLAGS) -reuse-exe=benchmark/fpga_hw.out benchmark/main.cpp -o benchmark/fpga_hw.out

clean:
	find . -name '*.out' -o -name '*.a' -o -name '*.o' | xargs rm -f

format:
	find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i --style=Mozilla
