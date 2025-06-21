/**
 * Performance Monitoring Utilities
 * Tracks performance metrics for optimization analysis
 */

interface PerformanceMetrics {
  textProcessingTime: number;
  suggestionGenerationTime: number;
  renderCount: number;
  cacheHitRate: number;
  memoryUsage: number;
}

class PerformanceMonitor {
  private metrics: PerformanceMetrics = {
    textProcessingTime: 0,
    suggestionGenerationTime: 0,
    renderCount: 0,
    cacheHitRate: 0,
    memoryUsage: 0,
  };

  private timers = new Map<string, number>();

  /**
   * Start timing an operation
   */
  startTimer(operation: string): void {
    this.timers.set(operation, performance.now());
  }

  /**
   * End timing an operation and record the duration
   */
  endTimer(operation: string): number {
    const startTime = this.timers.get(operation);
    if (!startTime) {
      console.warn(`Timer for operation "${operation}" was not started`);
      return 0;
    }

    const duration = performance.now() - startTime;
    this.timers.delete(operation);

    // Record the duration based on operation type
    switch (operation) {
      case "textProcessing":
        this.metrics.textProcessingTime = duration;
        break;
      case "suggestionGeneration":
        this.metrics.suggestionGenerationTime = duration;
        break;
    }

    return duration;
  }

  /**
   * Increment render count
   */
  incrementRenderCount(): void {
    this.metrics.renderCount++;
  }

  /**
   * Update cache hit rate
   */
  updateCacheHitRate(hitRate: number): void {
    this.metrics.cacheHitRate = hitRate;
  }

  /**
   * Get current performance metrics
   */
  getMetrics(): PerformanceMetrics {
    return { ...this.metrics };
  }

  /**
   * Reset all metrics
   */
  reset(): void {
    this.metrics = {
      textProcessingTime: 0,
      suggestionGenerationTime: 0,
      renderCount: 0,
      cacheHitRate: 0,
      memoryUsage: 0,
    };
    this.timers.clear();
  }

  /**
   * Get memory usage if available
   */
  getMemoryUsage(): number {
    if ("memory" in performance) {
      // @ts-expect-error - memory is not in standard types but exists in Chrome
      return performance.memory?.usedJSHeapSize || 0;
    }
    return 0;
  }

  /**
   * Log performance summary
   */
  logSummary(): void {
    const metrics = this.getMetrics();
    console.group("🚀 Performance Metrics");
    console.log(
      `📝 Text Processing: ${metrics.textProcessingTime.toFixed(2)}ms`
    );
    console.log(
      `🤖 AI Suggestions: ${metrics.suggestionGenerationTime.toFixed(2)}ms`
    );
    console.log(`🔄 Render Count: ${metrics.renderCount}`);
    console.log(`💾 Cache Hit Rate: ${metrics.cacheHitRate.toFixed(1)}%`);
    console.log(
      `🧠 Memory Usage: ${(this.getMemoryUsage() / 1024 / 1024).toFixed(2)}MB`
    );
    console.groupEnd();
  }
}

// Create singleton instance
export const performanceMonitor = new PerformanceMonitor();

// Export for testing
export { PerformanceMonitor };
