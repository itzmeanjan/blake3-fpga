# blake3-fpga
BLAKE3 on FPGA

## Prerequisite

I'm on

```bash
lsb_release -d

Description:    Ubuntu 20.04.3 LTS
```

while using `dpcpp` as SYCL compiler

```bash
dpcpp --version

Intel(R) oneAPI DPC++/C++ Compiler 2022.0.0 (2022.0.0.20211123)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /opt/intel/oneapi/compiler/2022.0.2/linux/bin-llvm
```

You'd probably like to get Intel oneAPI basekit, which has everything required for FPGA development. See [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html). For running FPGA h/w synthesis and execution, you would probably use Intel Devcloud.

## Usage

You can check functional correctness of BLAKE3 implementation on CPU by emulation.

```bash
make
```

You probably would like to see optimization report, which can be generated on non-FPGA attached host.

```bash
make fpga_opt_test
```

*You also have the option of running benchmark on CPU emulation, but you probably don't want to use those numbers as actual benchmark.*

```bash
make fpga_emu_bench # don't use as actual benchmark !
```

For running FPGA h/w test/ benchmark you'll need to go through **long** h/w synthesis phase, which can be executed on Intel Devcloud platform. See [here](https://devcloud.intel.com/oneapi/get_started/opencl).

### Job Submission

For easing FPGA h/w compilation/ execution job submissions on Intel Devcloud platform, I've use following scripts.

Assuming you're in root of this project

```bash
git clone https://github.com/itzmeanjan/blake3-fpga.git
cd blake3-fpga
```

#### Compilation Flow

Create job submission bash script

```bash
touch build_fpga_bench_hw.sh
```

And populate it with following content

```bash
#!/bin/bash

# file name: build_fpga_hw.sh

# env setup
export PATH=/glob/intel-python/python2/bin/:${PATH}
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1

# hardware compilation
time make fpga_hw_bench
```

Now submit compilation job targeting Intel Arria 10 board, while noting down job id

```bash
qsub -l nodes=1:fpga_compile:ppn=2 -l walltime=24:00:00 -d . build_fpga_bench_hw.sh

# note down job id e.g. 1850154
```

**Note :** If you happen to be interested in targeting Intel Stratix 10 board, consider using following compilation command instead of above Make build recipe.

```bash
# hardware compilation
time dpcpp -Wall -std=c++20 -I./include -O3 -DFPGA_HW -fintelfpga -Xshardware -Xsboard=intel_s10sx_pac:pac_s10 -reuse-exe=benchmark/fpga_hw.out benchmark/main.cpp -o benchmark/fpga_hw.out

# or consider reading Makefile
```

And finally submit job on `fpga_compile` enabled VM with same command shown as above.

#### Execution Flow

Create job submission shell script

```bash
touch run_fpga_bench_hw.sh
```

And populate it with environment setup and binary execution commands

```bash
#!/bin/bash

# file name: run_fpga_hw.sh

# env setup
export PATH=/glob/intel-python/python2/bin/:${PATH}
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1

# hardware image execution
pushd benchmark; ./fpga_hw.out; popd
```

Now submit execution job on VM, enabled with `fpga_runtime` capability & Intel Arria 10 board, while creating job dependency chain, which will ensure as soon as **long** FPGA h/w synthesis is completed, h/w image execution will start running

```bash
qsub -l nodes=1:fpga_runtime:arria10:ppn=2 -d . run_fpga_bench_hw.sh -W depend=afterok:1850154

# use compilation flow job id ( e.g. 1850154 ) to create dependency chain
```

**Note :** If you compiled h/w image targeting Intel Stratix 10 board, consider using following job submission command

```bash
qsub -l nodes=1:fpga_runtime:stratix10:ppn=2 -d . run_fpga_bench_hw.sh -W depend=afterok:1850157

# place proper compilation job id ( e.g. 1850157 ), to form dependency chain
```

After completion of compilation/ execution job submission, consider checking status using

```bash
watch -n 1 qstat -n -1

# or just `qstat -n -1`
```

When completed, following command(s) should reveal newly created files, having stdout/ stderr output of compilation/ execution flow in `{build|run}_fpga_bench_hw.sh.{o|e}1850157` files

```bash
ls -lhrt   # created files shown towards end of list
git status # untracked, newly created files
```

> Note, I found [this](https://devcloud.intel.com/oneapi/documentation/job-submission) guide on job submission helpful.
