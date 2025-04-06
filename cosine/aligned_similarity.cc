#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <random>
#include <omp.h>
#include <immintrin.h>
#include <malloc.h>  // For aligned allocation
#include <iomanip>   // For std::setprecision and std::fixed

// Aligned memory allocation (16-byte boundary for SSE, 32-byte for AVX)
#define ALIGNMENT 32

struct SimilarityResult {
    int i;
    int j;
    float similarity;
};

// Use a flat array for embeddings with aligned memory
struct Dataset {
    std::vector<std::string> ids;
    float* embeddings;  // Aligned memory
    int n_vectors;
    int dim;
    
    Dataset(int n_vectors, int dim) : n_vectors(n_vectors), dim(dim) {
        ids.resize(n_vectors);
        // Aligned allocation
        embeddings = (float*)_mm_malloc(n_vectors * dim * sizeof(float), ALIGNMENT);
    }
    
    ~Dataset() {
        _mm_free(embeddings);
    }
    
    float* get_vector(int idx) const {
        return embeddings + idx * dim;
    }
};

// Create test data with random embeddings
Dataset create_test_data(int n_vectors, int dim) {
    Dataset result(n_vectors, dim);
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    // Generate random embeddings
    for (int i = 0; i < n_vectors; ++i) {
        result.ids[i] = "id_" + std::to_string(i);
        
        for (int j = 0; j < dim; ++j) {
            result.embeddings[i * dim + j] = dist(gen);
        }
    }
    
    return result;
}

// Calculate vector norm using AVX - optimized version
float calculate_norm_avx(const float* vec, int dim) {
    __m256 sum_sq = _mm256_setzero_ps();
    
    // Process 8 elements at a time
    int i = 0;
    for (; i <= dim - 32; i += 32) {
        // Process 32 elements in one loop iteration for better instruction pipelining
        __m256 v1 = _mm256_loadu_ps(vec + i);
        __m256 v2 = _mm256_loadu_ps(vec + i + 8);
        __m256 v3 = _mm256_loadu_ps(vec + i + 16);
        __m256 v4 = _mm256_loadu_ps(vec + i + 24);
        
        sum_sq = _mm256_fmadd_ps(v1, v1, sum_sq);
        sum_sq = _mm256_fmadd_ps(v2, v2, sum_sq);
        sum_sq = _mm256_fmadd_ps(v3, v3, sum_sq);
        sum_sq = _mm256_fmadd_ps(v4, v4, sum_sq);
    }
    
    // Handle remaining elements in groups of 8
    for (; i <= dim - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(vec + i);
        sum_sq = _mm256_fmadd_ps(v, v, sum_sq);
    }
    
    // Horizontal sum
    float result = 0.0f;
    float temp[8] __attribute__((aligned(32)));
    _mm256_store_ps(temp, sum_sq);
    
    for (int j = 0; j < 8; j++) {
        result += temp[j];
    }
    
    // Handle remaining elements
    for (; i < dim; ++i) {
        result += vec[i] * vec[i];
    }
    
    return std::sqrt(result);
}

// Normalize a vector in-place using AVX - optimized version
void normalize_vector_avx(float* vec, int dim) {
    float norm = calculate_norm_avx(vec, dim);
    __m256 norm_vec = _mm256_set1_ps(norm);
    
    // Process 8 elements at a time
    int i = 0;
    for (; i <= dim - 32; i += 32) {
        // Process 32 elements for better pipelining
        __m256 v1 = _mm256_loadu_ps(vec + i);
        __m256 v2 = _mm256_loadu_ps(vec + i + 8);
        __m256 v3 = _mm256_loadu_ps(vec + i + 16);
        __m256 v4 = _mm256_loadu_ps(vec + i + 24);
        
        _mm256_storeu_ps(vec + i, _mm256_div_ps(v1, norm_vec));
        _mm256_storeu_ps(vec + i + 8, _mm256_div_ps(v2, norm_vec));
        _mm256_storeu_ps(vec + i + 16, _mm256_div_ps(v3, norm_vec));
        _mm256_storeu_ps(vec + i + 24, _mm256_div_ps(v4, norm_vec));
    }
    
    // Handle remaining elements in groups of 8
    for (; i <= dim - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(vec + i);
        _mm256_storeu_ps(vec + i, _mm256_div_ps(v, norm_vec));
    }
    
    // Handle remaining elements
    for (; i < dim; ++i) {
        vec[i] /= norm;
    }
}

// Dot product using AVX - optimized version with loop unrolling and FMA
float dot_product_avx(const float* vec1, const float* vec2, int dim) {
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    __m256 sum4 = _mm256_setzero_ps();
    
    // Process in larger chunks with multiple accumulators
    int i = 0;
    for (; i <= dim - 32; i += 32) {
        __m256 a1 = _mm256_loadu_ps(vec1 + i);
        __m256 b1 = _mm256_loadu_ps(vec2 + i);
        sum1 = _mm256_fmadd_ps(a1, b1, sum1);
        
        __m256 a2 = _mm256_loadu_ps(vec1 + i + 8);
        __m256 b2 = _mm256_loadu_ps(vec2 + i + 8);
        sum2 = _mm256_fmadd_ps(a2, b2, sum2);
        
        __m256 a3 = _mm256_loadu_ps(vec1 + i + 16);
        __m256 b3 = _mm256_loadu_ps(vec2 + i + 16);
        sum3 = _mm256_fmadd_ps(a3, b3, sum3);
        
        __m256 a4 = _mm256_loadu_ps(vec1 + i + 24);
        __m256 b4 = _mm256_loadu_ps(vec2 + i + 24);
        sum4 = _mm256_fmadd_ps(a4, b4, sum4);
    }
    
    // Combine the accumulators
    sum1 = _mm256_add_ps(sum1, sum2);
    sum3 = _mm256_add_ps(sum3, sum4);
    sum1 = _mm256_add_ps(sum1, sum3);
    
    // Handle remaining elements in groups of 8
    for (; i <= dim - 8; i += 8) {
        __m256 a = _mm256_loadu_ps(vec1 + i);
        __m256 b = _mm256_loadu_ps(vec2 + i);
        sum1 = _mm256_fmadd_ps(a, b, sum1);
    }
    
    // Horizontal sum
    float result = 0.0f;
    float temp[8] __attribute__((aligned(32)));
    _mm256_store_ps(temp, sum1);
    
    for (int j = 0; j < 8; j++) {
        result += temp[j];
    }
    
    // Handle remaining elements
    for (; i < dim; ++i) {
        result += vec1[i] * vec2[i];
    }
    
    return result;
}

// Block-based similarity calculation - processes matrices in cache-friendly blocks
void calculate_similarities_blocked(
    const Dataset& A, const Dataset& B, 
    std::vector<std::vector<SimilarityResult>>& thread_results,
    int max_results, int block_size = 64) {
    
    const int A_size = A.n_vectors;
    const int B_size = B.n_vectors;
    const int dim = A.dim;
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_max_threads();
        
        // Determine the blocks this thread will process
        int blocks_A = (A_size + block_size - 1) / block_size;
        int blocks_per_thread = (blocks_A + num_threads - 1) / num_threads;
        int start_block = thread_id * blocks_per_thread;
        int end_block = std::min(start_block + blocks_per_thread, blocks_A);
        
        // Process assigned blocks
        for (int block_i = start_block; block_i < end_block; block_i++) {
            int start_i = block_i * block_size;
            int end_i = std::min(start_i + block_size, A_size);
            
            for (int block_j = 0; block_j < (B_size + block_size - 1) / block_size; block_j++) {
                int start_j = block_j * block_size;
                int end_j = std::min(start_j + block_size, B_size);
                
                // Process the block
                for (int i = start_i; i < end_i; i++) {
                    const float* vec_a = A.get_vector(i);
                    
                    for (int j = start_j; j < end_j; j++) {
                        const float* vec_b = B.get_vector(j);
                        float similarity = dot_product_avx(vec_a, vec_b, dim);
                        
                        thread_results[thread_id].push_back({i, j, similarity});
                    }
                }
            }
        }
    }
}


int main() {
    // Test parameters
    int A_size = 10'000;
    int B_size = 10'000;
    int embedding_dim = 1000;
    
    // Create test data
    std::cout << "Creating test data..." << std::endl;
    Dataset A = create_test_data(A_size, embedding_dim);
    Dataset B = create_test_data(B_size, embedding_dim);
    
    // Start timing
    std::cout << "Calculating similarities..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Normalize all vectors (in place, parallel)
    #pragma omp parallel for
    for (int i = 0; i < A_size; ++i) {
        normalize_vector_avx(A.get_vector(i), embedding_dim);
    }
    
    #pragma omp parallel for
    for (int i = 0; i < B_size; ++i) {
        normalize_vector_avx(B.get_vector(i), embedding_dim);
    }
    
    // Pre-allocate results array - use fixed size for testing
    const int max_results = A_size * B_size;
    std::vector<SimilarityResult> results;
    results.reserve(max_results);
    
    // Thread-local results
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<SimilarityResult>> thread_results(num_threads);
    
    for (int t = 0; t < num_threads; ++t) {
        thread_results[t].reserve(max_results / num_threads);
    }
    
    // Use block-based calculation
    calculate_similarities_blocked(A, B, thread_results, max_results, 32);
    
    // Merge results from all threads
    for (int t = 0; t < num_threads; ++t) {
        results.insert(results.end(), thread_results[t].begin(), thread_results[t].end());
    }
    
    // Stop timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
    
    std::cout << "Computation time: " << std::fixed << std::setprecision(2) << duration << " seconds" << std::endl;
    
    std::cout << "Number of results: " << results.size() << std::endl;
    return 0;
}