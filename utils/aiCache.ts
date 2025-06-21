/**
 * AI Response Cache
 * Implements LRU (Least Recently Used) caching for AI suggestions
 * Reduces API calls and improves response time for repeated patterns
 */

interface CacheEntry {
  response: string;
  timestamp: number;
  hitCount: number;
}

interface CacheStats {
  hits: number;
  misses: number;
  totalRequests: number;
  hitRate: number;
}

class AICache {
  private cache = new Map<string, CacheEntry>();
  private maxSize: number;
  private ttl: number; // Time to live in milliseconds
  private stats: CacheStats = {
    hits: 0,
    misses: 0,
    totalRequests: 0,
    hitRate: 0,
  };

  constructor(maxSize = 100, ttlMinutes = 30) {
    this.maxSize = maxSize;
    this.ttl = ttlMinutes * 60 * 1000; // Convert to milliseconds
  }

  /**
   * Generate cache key from text and conversation history
   */
  private generateKey(text: string, conversationHistory: string[]): string {
    const historyHash = conversationHistory.join("|");
    return `${text.toLowerCase().trim()}:${this.hashString(historyHash)}`;
  }

  /**
   * Simple hash function for conversation history
   */
  private hashString(str: string): string {
    let hash = 0;
    if (str.length === 0) return hash.toString();

    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32-bit integer
    }

    return Math.abs(hash).toString(36);
  }

  /**
   * Check if cache entry is expired
   */
  private isExpired(entry: CacheEntry): boolean {
    return Date.now() - entry.timestamp > this.ttl;
  }

  /**
   * Remove expired entries
   */
  private cleanExpired(): void {
    const now = Date.now();
    for (const [key, entry] of this.cache.entries()) {
      if (now - entry.timestamp > this.ttl) {
        this.cache.delete(key);
      }
    }
  }

  /**
   * Remove least recently used entry when cache is full
   */
  private evictLRU(): void {
    if (this.cache.size === 0) return;

    let oldestKey = "";
    let oldestTime = Date.now();

    for (const [key, entry] of this.cache.entries()) {
      if (entry.timestamp < oldestTime) {
        oldestTime = entry.timestamp;
        oldestKey = key;
      }
    }

    if (oldestKey) {
      this.cache.delete(oldestKey);
    }
  }

  /**
   * Get cached response if available and not expired
   */
  get(text: string, conversationHistory: string[]): string | null {
    this.stats.totalRequests++;

    const key = this.generateKey(text, conversationHistory);
    const entry = this.cache.get(key);

    if (!entry || this.isExpired(entry)) {
      this.stats.misses++;
      if (entry) {
        this.cache.delete(key); // Remove expired entry
      }
      this.updateStats();
      return null;
    }

    // Update access time and hit count
    entry.timestamp = Date.now();
    entry.hitCount++;
    this.stats.hits++;
    this.updateStats();

    return entry.response;
  }

  /**
   * Store response in cache
   */
  set(text: string, conversationHistory: string[], response: string): void {
    if (!response || response.trim().length === 0) {
      return; // Don't cache empty responses
    }

    const key = this.generateKey(text, conversationHistory);

    // Clean expired entries periodically
    if (this.cache.size > this.maxSize * 0.8) {
      this.cleanExpired();
    }

    // Evict LRU if at capacity
    if (this.cache.size >= this.maxSize) {
      this.evictLRU();
    }

    const entry: CacheEntry = {
      response: response.trim(),
      timestamp: Date.now(),
      hitCount: 0,
    };

    this.cache.set(key, entry);
  }

  /**
   * Update cache statistics
   */
  private updateStats(): void {
    this.stats.hitRate =
      this.stats.totalRequests > 0
        ? (this.stats.hits / this.stats.totalRequests) * 100
        : 0;
  }

  /**
   * Get cache statistics
   */
  getStats(): CacheStats {
    return { ...this.stats };
  }

  /**
   * Clear all cache entries
   */
  clear(): void {
    this.cache.clear();
    this.stats = {
      hits: 0,
      misses: 0,
      totalRequests: 0,
      hitRate: 0,
    };
  }

  /**
   * Get cache size
   */
  size(): number {
    return this.cache.size;
  }

  /**
   * Check if cache has space
   */
  hasSpace(): boolean {
    return this.cache.size < this.maxSize;
  }

  /**
   * Get cache entries for debugging
   */
  getEntries(): Array<{ key: string; entry: CacheEntry }> {
    return Array.from(this.cache.entries()).map(([key, entry]) => ({
      key,
      entry: { ...entry },
    }));
  }
}

// Create singleton instance
export const aiCache = new AICache(100, 30); // 100 entries, 30 minutes TTL

// Export for testing
export { AICache };
