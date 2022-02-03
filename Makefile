CXX = dpcpp
CXXFLAGS = -Wall -std=c++20
SYCLFLAGS = -fsycl
OPTFLAGS = -O3
IFLAGS = -I ./include

all: test_blake3

test_blake3: test/a.out
	./$<

test/a.out: test/main.cpp include/*.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(OPTFLAGS) $(IFLAGS) $< -o $@

clean:
	find . -name 'a.out' -o -name '*.o' | xargs rm -f

format:
	find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i --style=Mozilla
