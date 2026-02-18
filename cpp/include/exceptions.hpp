#pragma once
#include <stdexcept>
#include <string>

/// Base exception for inference-related errors
class InferenceError : public std::runtime_error {
public:
    explicit InferenceError(const std::string& msg) : std::runtime_error(msg) {}
};

/// Exception raised when inference submission fails
class InferenceSubmitError : public InferenceError {
public:
    explicit InferenceSubmitError(const std::string& msg) : InferenceError(msg) {}
};

/// Exception raised when inference operation times out
class InferenceTimeoutError : public InferenceError {
public:
    explicit InferenceTimeoutError(const std::string& msg) : InferenceError(msg) {}
};

/// Exception raised when waiting for inference results fails
class InferenceWaitError : public InferenceError {
public:
    explicit InferenceWaitError(const std::string& msg) : InferenceError(msg) {}
};

/// Exception raised during synchronous inference pipeline execution
class InferencePipelineError : public InferenceError {
public:
    explicit InferencePipelineError(const std::string& msg) : InferenceError(msg) {}
};
