import numpy as np
from numba import njit, float32

# This is the function that caused the error.
# The traceback indicates that the calculated `combined_scores` is of type float64,
# which conflicts with the function's signature specifying a float32[:] return type.
@njit(float32[:](float32[:], float32[:, :], float32[:], float32[:], float32, float32))
def calculate_similarity_njit(query_embedding, memory_embeddings, rewards, recency_scores, alpha_recency, alpha_relevance):
    """
    Calculates combined similarity scores using Numba.
    """
    # Relevance score based on cosine similarity (dot product of normalized vectors)
    dot_product = np.dot(memory_embeddings, query_embedding)
    relevance_scores = dot_product

    # Weighted combination of relevance and recency
    relevance_component = alpha_relevance * relevance_scores
    recency_component = alpha_recency * recency_scores
    
    # Combine with rewards
    combined_scores = relevance_component + recency_component + rewards

    # FIX: The TypingError occurs because intermediate calculations can sometimes
    # promote the dtype to float64. We explicitly cast the result back to float32
    # to match the return type specified in the @njit signature.
    return combined_scores.astype(np.float32)

@njit(float32[:](float32[:]))
def softmax_njit(scores):
    """
    Computes softmax for a 1D array using Numba, ensuring float32 output.
    """
    if scores.size == 0:
        return np.empty(0, dtype=np.float32)
    # Subtract max for numerical stability
    max_score = np.max(scores)
    exp_scores = np.exp(scores - max_score)
    sum_exp_scores = np.sum(exp_scores)
    
    # Ensure the final output is float32
    return (exp_scores / sum_exp_scores).astype(np.float32)

class LocalSimulator:
    """
    A simulator that manages a local memory buffer and uses Numba-optimized
    functions for retrieval.
    """
    def __init__(self, num_memories, embedding_dim, alpha_recency=1.0, alpha_relevance=1.0):
        self.num_memories = num_memories
        self.embedding_dim = embedding_dim
        self.alpha_recency = np.float32(alpha_recency)
        self.alpha_relevance = np.float32(alpha_relevance)

        # Initialize memory structures with float32 dtype
        self.memory_embeddings = np.zeros((num_memories, embedding_dim), dtype=np.float32)
        self.rewards = np.zeros(num_memories, dtype=np.float32)
        self.recency_scores = np.zeros(num_memories, dtype=np.float32)
        self.memory_counter = 0

    def add_memory(self, embedding, reward):
        """
        Adds a new memory to the buffer, overwriting the oldest if full.
        """
        idx = self.memory_counter % self.num_memories
        self.memory_embeddings[idx] = embedding.astype(np.float32)
        self.rewards[idx] = np.float32(reward)
        
        # Update recency scores: decay all, then set the new one to 1.0
        self.recency_scores *= np.float32(0.99)
        self.recency_scores[idx] = np.float32(1.0)
        
        self.memory_counter += 1

    def retrieve_memory_probabilities(self, query_embedding):
        """
        Calculates the probability distribution over memories for a given query.
        """
        if self.memory_counter == 0:
            return np.array([], dtype=np.float32)

        num_active_memories = min(self.memory_counter, self.num_memories)
        
        # Slicing can create non-contiguous views. Using np.ascontiguousarray
        # resolves the NumbaPerformanceWarning and can improve performance.
        active_embeddings = np.ascontiguousarray(self.memory_embeddings[:num_active_memories])
        active_rewards = self.rewards[:num_active_memories]
        active_recency = self.recency_scores[:num_active_memories]

        # Ensure query_embedding is also a contiguous float32 array
        query_embedding_f32 = np.ascontiguousarray(query_embedding, dtype=np.float32)

        combined_scores = calculate_similarity_njit(
            query_embedding_f32,
            active_embeddings,
            active_rewards,
            active_recency,
            self.alpha_recency,
            self.alpha_relevance
        )

        if combined_scores.size == 0:
            return np.array([], dtype=np.float32)

        probabilities = softmax_njit(combined_scores)
        return probabilities

# Example usage (for verification)
if __name__ == '__main__':
    NUM_MEMORIES = 100
    EMBEDDING_DIM = 64

    simulator = LocalSimulator(NUM_MEMORIES, EMBEDDING_DIM)

    # Add some memories
    for i in range(50):
        embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        embedding /= np.linalg.norm(embedding)  # Normalize
        reward = np.random.rand()
        simulator.add_memory(embedding, reward)

    # Create a query
    query = np.random.rand(EMBEDDING_DIM).astype(np.float32)
    query /= np.linalg.norm(query) # Normalize

    # Retrieve probabilities
    probs = simulator.retrieve_memory_probabilities(query)

    print("Simulator initialized and ran successfully.")
    print(f"Number of active memories: {min(simulator.memory_counter, simulator.num_memories)}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Probabilities sum: {np.sum(probs)}")
    print(f"Probabilities dtype: {probs.dtype}")
    assert probs.dtype == np.float32
    assert np.isclose(np.sum(probs), 1.0)