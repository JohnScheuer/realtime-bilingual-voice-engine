#ifndef SPSC_RING_BUFFER_HPP
#define SPSC_RING_BUFFER_HPP

#include <atomic>
#include <vector>
#include <cstddef>
#include <algorithm>
#include <new>

// Cache line size on modern x86_64 architectures
constexpr size_t HARDWARE_DESTRUCTIVE_INTERFERENCE_SIZE = 64;

template <typename T>
class SPSCRingBuffer {
public:
    explicit SPSCRingBuffer(size_t capacity)
        : capacity_(capacity), buffer_(capacity + 1) {
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
    }

    ~SPSCRingBuffer() = default;

    SPSCRingBuffer(const SPSCRingBuffer&) = delete;
    SPSCRingBuffer& operator=(const SPSCRingBuffer&) = delete;

    inline bool push(const T& item) {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t next_tail = (current_tail + 1) % buffer_.size();

        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false; // Buffer full
        }

        buffer_[current_tail] = item;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    inline bool pop(T& item) {
        const size_t current_head = head_.load(std::memory_order_relaxed);

        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false; // Buffer empty
        }

        item = buffer_[current_head];
        head_.store((current_head + 1) % buffer_.size(), std::memory_order_release);
        return true;
    }

    inline size_t push_slice(const T* data, size_t count) {
        size_t pushed = 0;
        while (pushed < count) {
            const size_t current_tail = tail_.load(std::memory_order_relaxed);
            const size_t current_head = head_.load(std::memory_order_acquire);

            size_t free_space = 0;
            if (current_tail >= current_head) {
                free_space = buffer_.size() - current_tail + current_head - 1;
            } else {
                free_space = current_head - current_tail - 1;
            }

            if (free_space == 0) break;

            size_t chunk = std::min(count - pushed, free_space);
            size_t contiguous = std::min(chunk, buffer_.size() - current_tail);

            std::copy(data + pushed, data + pushed + contiguous, buffer_.begin() + current_tail);

            size_t next_tail = (current_tail + contiguous) % buffer_.size();
            tail_.store(next_tail, std::memory_order_release);
            pushed += contiguous;
        }
        return pushed;
    }

    inline size_t pop_slice(T* destination, size_t max_count) {
        size_t popped = 0;
        while (popped < max_count) {
            const size_t current_head = head_.load(std::memory_order_relaxed);
            const size_t current_tail = tail_.load(std::memory_order_acquire);

            if (current_head == current_tail) break;

            size_t available = 0;
            if (current_tail > current_head) {
                available = current_tail - current_head;
            } else {
                available = buffer_.size() - current_head;
            }

            size_t chunk = std::min(max_count - popped, available);
            std::copy(buffer_.begin() + current_head, buffer_.begin() + current_head + chunk, destination + popped);

            size_t next_head = (current_head + chunk) % buffer_.size();
            head_.store(next_head, std::memory_order_release);
            popped += chunk;
        }
        return popped;
    }

    [[nodiscard]] inline size_t capacity() const noexcept { return capacity_; }

    [[nodiscard]] inline size_t size() const noexcept {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail >= head) return tail - head;
        return buffer_.size() - head + tail;
    }

    [[nodiscard]] inline bool empty() const noexcept {
        return head_.load(std::memory_order_relaxed) == tail_.load(std::memory_order_relaxed);
    }

private:
    const size_t capacity_;
    std::vector<T> buffer_;

    alignas(HARDWARE_DESTRUCTIVE_INTERFERENCE_SIZE) std::atomic<size_t> head_{0};
    alignas(HARDWARE_DESTRUCTIVE_INTERFERENCE_SIZE) std::atomic<size_t> tail_{0};
};

#endif // SPSC_RING_BUFFER_HPP
