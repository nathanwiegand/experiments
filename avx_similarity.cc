#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <random>
#include <omp.h>
#include <immintrin.h> // For SSE/AVX intrinsics

// Use a flat array for embeddings (better cache locality)
struct Dataset {
    std::vector<std::string> ids;
    std::vector<float> embeddings; // Flat array [n_vectors * dim]
    int n_vectors;
    int dim;
    
    Dataset(int n_vectors, int dim) : n_vectors(n_vectors), dim(dim) {
        ids.resize(n_vectors);
        embeddings.resize(n_vectors * dim);
    }
    
    float* get_vector(int idx) {
        return &embeddings[idx * dim];
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

// Calculate vector norm using AVX
float calculate_norm_avx(const float* vec, int dim) {
    __m256 sum_sq = _mm256_setzero_ps();
    
    // Process 8 elements at a time
    int i = 0;
    for (; i <= dim - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(vec + i);
        sum_sq = _mm256_add_ps(sum_sq, _mm256_mul_ps(v, v));
    }
    
    // Horizontal sum of 8 elements
    __m256 temp1 = _mm256_hadd_ps(sum_sq, sum_sq);
    __m256 temp2 = _mm256_hadd_ps(temp1, temp1);
    // Extract the lower 128 bits and add them to the upper 128 bits
    __m128 sum_low = _mm256_extractf128_ps(temp2, 0);
    __m128 sum_high = _mm256_extractf128_ps(temp2, 1);
    __m128 final_sum = _mm_add_ps(sum_low, sum_high);
    float result = _mm_cvtss_f32(final_sum);
    
    // Handle remaining elements
    for (; i < dim; ++i) {
        result += vec[i] * vec[i];
    }
    
    return std::sqrt(result);
}

// Normalize a vector in-place using AVX
void normalize_vector_avx(float* vec, int dim) {
    float norm = calculate_norm_avx(vec, dim);
    __m256 norm_vec = _mm256_set1_ps(norm);
    
    // Process 8 elements at a time
    int i = 0;
    for (; i <= dim - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(vec + i);
        v = _mm256_div_ps(v, norm_vec);
        _mm256_storeu_ps(vec + i, v);
    }
    
    // Handle remaining elements
    for (; i < dim; ++i) {
        vec[i] /= norm;
    }
}

// Dot product using AVX (for normalized vectors = cosine similarity)
float dot_product_avx(const float* vec1, const float* vec2, int dim) {
    __m256 sum = _mm256_setzero_ps();
    
    // Process 8 elements at a time
    int i = 0;
    for (; i <= dim - 8; i += 8) {
        __m256 v1 = _mm256_loadu_ps(vec1 + i);
        __m256 v2 = _mm256_loadu_ps(vec2 + i);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));
    }
    
    // Horizontal sum of 8 elements
    __m256 temp1 = _mm256_hadd_ps(sum, sum);
    __m256 temp2 = _mm256_hadd_ps(temp1, temp1);
    // Extract the lower 128 bits and add them to the upper 128 bits
    __m128 sum_low = _mm256_extractf128_ps(temp2, 0);
    __m128 sum_high = _mm256_extractf128_ps(temp2, 1);
    __m128 final_sum = _mm_add_ps(sum_low, sum_high);
    float result = _mm_cvtss_f32(final_sum);
    
    // Handle remaining elements
    for (; i < dim; ++i) {
        result += vec1[i] * vec2[i];
    }
    
    return result;
}

struct SimilarityResult {
    int i;
    int j;
    float similarity;
};

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
    
    // Use a more efficient parallel approach with chunk partitioning
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<SimilarityResult>> thread_results(num_threads);
    
    for (int t = 0; t < num_threads; ++t) {
        thread_results[t].reserve(max_results / num_threads);
    }
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int chunk_size = (A_size + num_threads - 1) / num_threads;
        int start = thread_id * chunk_size;
        int end = std::min(start + chunk_size, A_size);
        
        for (int i = start; i < end; ++i) {
            float* vec_a = A.get_vector(i);
            
            for (int j = 0; j < B_size; ++j) {
                float* vec_b = B.get_vector(j);
                float similarity = dot_product_avx(vec_a, vec_b, embedding_dim);
                
                thread_results[thread_id].push_back({i, j, similarity});
            }
        }
    }
    
    // Merge results from all threads
    for (int t = 0; t < num_threads; ++t) {
        results.insert(results.end(), thread_results[t].begin(), thread_results[t].end());
    }
    
    // Stop timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
    
    std::cout << "Computation time: " << duration << " seconds" << std::endl;
    
    // Print a few results as example
    std::cout << "Number of results: " << results.size() << std::endl;
    
    return 0;
}