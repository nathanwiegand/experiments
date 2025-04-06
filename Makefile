CXX = g++
CXXFLAGS = -std=c++11 -fopenmp -mavx2 -O3 --std c++17 -mfma -march=native
LDFLAGS = -lopenblas

all: results.txt

bin:
	mkdir -p bin

results:
	mkdir -p results

bin/avx_similarity: avx_similarity.cc | bin
	$(CXX) $(CXXFLAGS) -o $@ $<

bin/aligned_similarity: aligned_similarity.cc | bin
	$(CXX) $(CXXFLAGS) -o $@ $<

bin/blas_similarity: blas.cc | bin
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

bin/simd_similarity: simd.cc | bin
	$(CXX) $(CXXFLAGS) -o $@ $<

# Results targets that only run if the code has changed
results/avx_results.txt: bin/avx_similarity | results
	@echo "Running AVX similarity" > $@; \
	./bin/avx_similarity >> $@; \
	echo "----------------------------------------" >> $@; 

results/aligned_results.txt: bin/aligned_similarity | results
	@echo "Running aligned similarity" > $@; \
	./bin/aligned_similarity >> $@; \
	echo "----------------------------------------" >> $@; 

results/blas_results.txt: bin/blas_similarity | results
	@echo "Running BLAS similarity" > $@; \
	./bin/blas_similarity >> $@; \
	echo "----------------------------------------" >> $@; 

results/simd_results.txt: bin/simd_similarity | results
	@echo "Running SIMD similarity" > $@; \
	./bin/simd_similarity >> $@; \
	echo "----------------------------------------" >> $@; 

results.txt: results/avx_results.txt results/aligned_results.txt results/blas_results.txt results/simd_results.txt
	cat $^ > $@

clean:
	rm -f bin/avx_similarity bin/aligned_similarity bin/blas_similarity bin/simd_similarity
	rm -f results/* results.txt

.PHONY: all clean