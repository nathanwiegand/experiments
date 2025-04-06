#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <random>
#include <omp.h>  // For OpenMP parallelization

// Structure to hold ID and embedding
struct EmbeddingItem {
    std::string id;
    std::vector<float> embedding;
};

// Create test data with random embeddings
std::vector<EmbeddingItem> create_test_data(int n_vectors, int dim) {
    std::vector<EmbeddingItem> result;
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    // Generate random embeddings
    for (int i = 0; i < n_vectors; ++i) {
        EmbeddingItem item;
        item.id = "id_" + std::to_string(i);
        item.embedding.resize(dim);
        
        for (int j = 0; j < dim; ++j) {
            item.embedding[j] = dist(gen);
        }
        
        result.push_back(item);
    }
    
    return result;
}

// Calculate vector norm
float calculate_norm(const std::vector<float>& vec) {
    float sum_squares = 0.0f;
    for (const auto& val : vec) {
        sum_squares += val * val;
    }
    return std::sqrt(sum_squares);
}

// Normalize a vector (make it unit length)
void normalize_vector_inplace(std::vector<float>& vec) {
    float norm = calculate_norm(vec);
    for (auto& val : vec) {
        val /= norm;
    }
}

// Dot product (for normalized vectors = cosine similarity)
float dot_product(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    float result = 0.0f;
    for (size_t i = 0; i < vec1.size(); ++i) {
        result += vec1[i] * vec2[i];
    }
    return result;
}

int main() {
    // Test parameters
    int A_size = 10'000;
    int B_size = 10'000;
    int embedding_dim = 1'000;
    
    // Create test data
    std::cout << "Creating test data..." << std::endl;
    auto A = create_test_data(A_size, embedding_dim);
    auto B = create_test_data(B_size, embedding_dim);
    
    // Start timing
    std::cout << "Calculating similarities..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Normalize all vectors in A and B (in place)
    #pragma omp parallel for
    for (int i = 0; i < A_size; ++i) {
        normalize_vector_inplace(A[i].embedding);
    }
    
    #pragma omp parallel for
    for (int i = 0; i < B_size; ++i) {
        normalize_vector_inplace(B[i].embedding);
    }
    
    // Calculate similarities and format results
    std::vector<std::pair<std::pair<int, int>, float>> results;
    results.reserve(A_size * B_size);  // Pre-allocate to avoid reallocation
    
    // Use OpenMP to parallelize the outer loop
    #pragma omp parallel
    {
        // Local results for each thread
        std::vector<std::pair<std::pair<int, int>, float>> local_results;
        local_results.reserve(A_size * B_size / omp_get_num_threads());
        
        #pragma omp for
        for (int i = 0; i < A_size; ++i) {
            for (int j = 0; j < B_size; ++j) {
                float similarity = dot_product(A[i].embedding, B[j].embedding);
                local_results.push_back({{i, j}, similarity});
            }
        }
        
        // Merge results
        #pragma omp critical
        {
            results.insert(results.end(), local_results.begin(), local_results.end());
        }
    }
    
    // Stop timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
    
    std::cout << "Computation time: " << duration << " seconds" << std::endl;
    
    std::cout << "Results size: " << results.size() << std::endl;
    
    return 0;
}
