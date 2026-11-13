#pragma once
#include <cstddef>
#include <functional>
#include <string>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

// Runs task(0) ... task(num_tasks - 1) and returns when all of them are done.
class IExecutor {
public:
    virtual ~IExecutor() = default;
    virtual std::string name() const = 0;
    virtual int concurrency() const = 0;
    virtual void run(std::size_t num_tasks, const std::function<void(std::size_t)>& task) const = 0;
};

class SequentialExecutor final : public IExecutor {
public:
    std::string name() const override { return "seq"; }
    int concurrency() const override { return 1; }

    void run(std::size_t num_tasks, const std::function<void(std::size_t)>& task) const override {
        for (std::size_t i = 0; i < num_tasks; ++i) task(i);
    }
};

// Each executor has its own task_arena, i.e. its own thread limit.
class TbbExecutor final : public IExecutor {
public:
    explicit TbbExecutor(int num_threads = tbb::task_arena::automatic) : arena_(num_threads) {}

    std::string name() const override { return "tbb" + std::to_string(concurrency()); }
    int concurrency() const override { return arena_.max_concurrency(); }

    void run(std::size_t num_tasks, const std::function<void(std::size_t)>& task) const override {
        arena_.execute([&] {
            tbb::parallel_for(tbb::blocked_range<std::size_t>(0, num_tasks),
                              [&](const tbb::blocked_range<std::size_t>& r) {
                                  for (std::size_t i = r.begin(); i != r.end(); ++i) task(i);
                              });
        });
    }

private:
    mutable tbb::task_arena arena_;  // execute() is non-const
};
