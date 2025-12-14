/**
 * -----------------------------------------------------------------------------
 * Project: Fossil Logic
 *
 * This file is part of the Fossil Logic project, which aims to develop
 * high-performance, cross-platform applications and libraries. The code
 * contained herein is licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License. You may obtain
 * a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * Author: Michael Gene Brockus (Dreamer)
 * Date: 04/05/2014
 *
 * Copyright (C) 2014-2025 Fossil Logic. All rights reserved.
 * -----------------------------------------------------------------------------
 */
#ifndef FOSSIL_ALGORITHM_ML_H
#define FOSSIL_ALGORITHM_ML_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ======================================================
// Fossil Algorithm ML — Core Types
// ======================================================

/**
 * @brief Opaque ML model handle.
 *
 * Concrete implementation may include:
 * - Linear models, decision trees, random forests
 * - Neural networks / Grok-inspired memory
 * - Persistent state and parameters
 */
typedef struct fossil_ml_model fossil_ml_model_t;

/**
 * @brief Opaque dataset handle.
 *
 * Backend-defined; could be memory, storage, or tensor-backed.
 */
typedef struct fossil_ml_dataset fossil_ml_dataset_t;

/**
 * @brief ML execution phase.
 */
typedef enum fossil_ml_phase {
    FOSSIL_ML_PHASE_TRAIN,
    FOSSIL_ML_PHASE_EVAL,
    FOSSIL_ML_PHASE_PREDICT
} fossil_ml_phase_t;

/**
 * @brief Metric / callback function.
 *
 * @param metric_id Metric name (e.g., "loss", "accuracy")
 * @param value Metric value
 * @param step Training/evaluation step
 * @param user User context
 * @return true to continue, false to abort
 */
typedef bool (*fossil_ml_metric_fn)(
    const char *metric_id,
    double value,
    size_t step,
    void *user
);

// ======================================================
// ML Exec Interface
// ======================================================

/**
 * @brief Executes an ML algorithm.
 *
 * Algorithm IDs (string-based):
 *   - Supervised: "linear-regression", "logistic-regression", "svm", "knn"
 *   - Unsupervised: "kmeans", "dbscan", "pca"
 *   - Tree-based: "decision-tree", "random-forest"
 *   - Neural: "mlp", "cnn", "rnn"
 *
 * Returns:
 *   >=0 : algorithm-specific result or success
 *   -1  : failure (training or prediction)
 *   -2  : invalid input / dataset
 *   -3  : unsupported algorithm
 *   -4  : unsupported phase or configuration
 */
int fossil_algorithm_ml_exec(
    fossil_ml_model_t *model,
    fossil_ml_dataset_t *dataset,
    const char *algorithm_id,
    fossil_ml_phase_t phase,
    fossil_ml_metric_fn metric,
    void *user
);

// ======================================================
// Model Lifecycle
// ======================================================

/**
 * @brief Creates a model for a given algorithm.
 */
fossil_ml_model_t *fossil_algorithm_ml_model_create(
    const char *algorithm_id
);

/**
 * @brief Destroys a model and releases resources.
 */
void fossil_algorithm_ml_model_destroy(
    fossil_ml_model_t *model
);

// ======================================================
// Dataset Lifecycle
// ======================================================

/**
 * @brief Creates a dataset (memory-backed / tensor-backed).
 */
fossil_ml_dataset_t *fossil_algorithm_ml_dataset_create(
    void *data,
    size_t rows,
    size_t cols,
    bool labels_present
);

/**
 * @brief Destroys a dataset.
 */
void fossil_algorithm_ml_dataset_destroy(
    fossil_ml_dataset_t *dataset
);

// ======================================================
// Utility API
// ======================================================

/**
 * @brief Checks if an ML algorithm is supported.
 */
bool fossil_algorithm_ml_supported(const char *algorithm_id);

/**
 * @brief Checks if an ML algorithm requires labels.
 */
bool fossil_algorithm_ml_requires_labels(const char *algorithm_id);

#ifdef __cplusplus
}

#include <string>

namespace fossil {

    namespace algorithm {
    
        /**
         * @brief RAII wrapper for Fossil ML algorithms.
         */
        class ML
        {
        public:
            static fossil_ml_model_t *create(const std::string &algorithm_id) {
                return fossil_algorithm_ml_model_create(algorithm_id.c_str());
            }
        
            static void destroy(fossil_ml_model_t *model) {
                fossil_algorithm_ml_model_destroy(model);
            }
        
            static int exec(
                fossil_ml_model_t *model,
                fossil_ml_dataset_t *dataset,
                const std::string &algorithm_id,
                fossil_ml_phase_t phase,
                fossil_ml_metric_fn metric = nullptr,
                void *user = nullptr
            ) {
                return fossil_algorithm_ml_exec(
                    model,
                    dataset,
                    algorithm_id.c_str(),
                    phase,
                    metric,
                    user
                );
            }
        
            static bool supported(const std::string &algorithm_id) {
                return fossil_algorithm_ml_supported(algorithm_id.c_str());
            }
        
            static bool requires_labels(const std::string &algorithm_id) {
                return fossil_algorithm_ml_requires_labels(algorithm_id.c_str());
            }
        };
    
    } // namespace algorithm

} // namespace fossil

#endif /* __cplusplus */

#endif /* FOSSIL_ALGORITHM_ML_H */
