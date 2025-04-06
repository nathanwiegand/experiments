#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <random>
#include <omp.h>
#include <cblas.h>


// Structure to hold ID and embedding
struct EmbeddingItem {
    std::string id;
    std::vector<float> embedding;
};

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

// Normalize vector using BLAS
void normalize_vector_blas(float* vec, int dim) {
    // Calculate the norm using BLAS
    float norm = cblas_snrm2(dim, vec, 1);
    
    // Scale the vector (1/norm) using BLAS
    float scale = 1.0f / norm;
    cblas_sscal(dim, scale, vec, 1);
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
        normalize_vector_blas(A.get_vector(i), embedding_dim);
    }
    
    #pragma omp parallel for
    for (int i = 0; i < B_size; ++i) {
        normalize_vector_blas(B.get_vector(i), embedding_dim);
    }
    
    // Pre-allocate results array
    const int max_results = A_size * B_size;
    std::vector<SimilarityResult> results;
    results.reserve(max_results);
    
    // Use a more efficient parallel approach with chunk partitioning
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<SimilarityResult>> thread_results(num_threads);
    
    for (int t = 0; t < num_threads; ++t) {
        thread_results[t].reserve(max_results / num_threads);
    }
    
    // Option 1: Block-based approach similar to before but using BLAS
    const int block_size = 32;
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int blocks_A = (A_size + block_size - 1) / block_size;
        int blocks_per_thread = (blocks_A + num_threads - 1) / num_threads;
        int start_block = thread_id * blocks_per_thread;
        int end_block = std::min(start_block + blocks_per_thread, blocks_A);
        
        for (int block_i = start_block; block_i < end_block; block_i++) {
            int start_i = block_i * block_size;
            int end_i = std::min(start_i + block_size, A_size);
            
            for (int block_j = 0; block_j < (B_size + block_size - 1) / block_size; block_j++) {
                int start_j = block_j * block_size;
                int end_j = std::min(start_j + block_size, B_size);
                
                for (int i = start_i; i < end_i; i++) {
                    float* vec_a = A.get_vector(i);
                    
                    for (int j = start_j; j < end_j; j++) {
                        float* vec_b = B.get_vector(j);
                        
                        // Use BLAS dot product
                        float similarity = cblas_sdot(embedding_dim, vec_a, 1, vec_b, 1);
                        
                        thread_results[thread_id].push_back({i, j, similarity});
                    }
                }
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
    std::cout << "Example results (first 5):" << std::endl;
    std::cout << "Number of results: " << results.size() << std::endl;
    
    return 0;
}
