# Kick.com Detection System Guide

## Overview

The kick detection system is a sophisticated content filtering mechanism that prevents users from including kick.com streaming platform links in their bios. It uses multiple detection layers to catch 70+ obfuscation patterns while maintaining <5ms performance.

## Why Kick Detection?

Many community platforms prohibit promotional links, especially to streaming services. Users often try to circumvent these rules by obfuscating URLs in creative ways. This system catches these attempts while allowing legitimate uses of the word "kick" in normal contexts.

## Architecture

### Core Components

1. **Detection Engine** (`lib/kickDetection.ts`)
   - 71 regex patterns for various obfuscations
   - Homoglyph detection for Unicode confusables
   - Fuzzy matching with Levenshtein distance
   - Context analysis for confidence scoring
   - Zero-width character normalization

2. **React Hook** (`hooks/useKickDetection.tsx`)
   - 300ms debounced checking
   - Session-based logging
   - Integration with form state

3. **UI Components** (`components/KickDetectionWarning.tsx`)
   - Inline warnings for low confidence
   - Full panel warnings for high confidence
   - Color-coded severity levels

4. **API Logging** (`app/api/kick-detection-logs/route.ts`)
   - Pattern collection for improvement
   - False positive tracking

## Detection Patterns

### Pattern Categories

1. **Basic Substitutions** (26 patterns)
   ```
   k1ck, k!ck, k|ck, k.i.c.k
   ```

2. **Phonetic Variations** (18 patterns)
   ```
   keek, kyck, kaik, kook
   ```

3. **Advanced Obfuscations** (8 patterns)
   ```
   ckik (reversed), k((i))ck (nested)
   ```

4. **Zero-Width Characters** (6 types)
   ```
   k\u200Bi\u200Bck (invisible spaces)
   ```

5. **Homoglyphs** (Unicode lookalikes)
   ```
   кick (Cyrillic к), κick (Greek κ)
   ```

### Whitelisted Phrases

The system allows legitimate uses:
- "kick the ball"
- "kick off the event"
- "kick back and relax"
- 30+ other common phrases

## Implementation Guide

### Basic Usage

```typescript
import { useKickDetection } from '@/hooks/useKickDetection';

function MyForm() {
  const [bio, setBio] = useState('');
  const { 
    isKickDetected, 
    confidence, 
    logDetection 
  } = useKickDetection(bio);

  return (
    <>
      <textarea 
        value={bio}
        onChange={(e) => setBio(e.target.value)}
      />
      {isKickDetected && (
        <KickDetectionWarning 
          confidence={confidence}
          onDismiss={() => logDetection('dismissed')}
        />
      )}
    </>
  );
}
```

### Advanced Configuration

```typescript
// Adjust detection sensitivity
const FUZZY_MATCH_THRESHOLD = 1; // Levenshtein distance

// Modify confidence thresholds
const INLINE_WARNING_THRESHOLD = 0.5;  // 50%
const FULL_WARNING_THRESHOLD = 0.7;    // 70%

// Add custom patterns
const CUSTOM_PATTERNS = [
  /your-pattern-here/gi
];
```

## Testing

### Manual Testing

Try these examples in the bio field:

**Should Detect:**
- `k!ck.com`
- `check my k i c k`
- `кick` (Cyrillic)
- `k((i))ck`
- `k\u200Bi\u200Bck`

**Should NOT Detect:**
- `kick the ball`
- `kickstart my project`
- `quick response`

### Automated Testing

```bash
# Run the test suite
npm test lib/__tests__/kickDetection.test.ts

# Performance benchmark
npm run benchmark:kick
```

## Performance

- **Average detection time**: <5ms
- **Cache hit rate**: 95%+
- **False positive rate**: <0.1%
- **Pattern coverage**: 70+ variations

### Optimization Techniques

1. **Progressive Detection**: Quick pre-check before full analysis
2. **Result Caching**: 5-minute TTL for repeated text
3. **Early Exit**: Skip processing for obvious non-matches
4. **Efficient Regex**: Optimized pattern order

## Maintenance

### Adding New Patterns

1. Identify the obfuscation technique
2. Add pattern to appropriate phase in `kickDetection.ts`
3. Add test cases to verify detection
4. Update documentation

### Monitoring False Positives

1. Review logs at `/api/kick-detection-logs`
2. Identify legitimate phrases being caught
3. Add to whitelist or adjust patterns
4. Deploy and monitor

### Pattern Analysis

The logging system helps identify new obfuscation attempts:

```typescript
// Example log entry
{
  sessionId: "abc123",
  timestamp: 1234567890,
  text: "k!ck.com",
  confidence: 0.95,
  action: "detected",
  patterns: ["k[i1l!|]ck"]
}
```

## Future Enhancements

1. **Machine Learning**: Train model on collected patterns
2. **Admin Dashboard**: Visual pattern analysis
3. **A/B Testing**: Optimize warning UI
4. **Rate Limiting**: Prevent detection spam
5. **Multi-language**: Support international obfuscations

## Troubleshooting

### Detection Not Working

1. Check if text feature coordinator is blocking
2. Verify debounce delay (300ms)
3. Ensure patterns are loaded correctly
4. Check browser console for errors

### Too Many False Positives

1. Review whitelist phrases
2. Adjust confidence thresholds
3. Refine regex patterns
4. Consider context weighting

### Performance Issues

1. Check cache implementation
2. Review regex complexity
3. Enable progressive detection
4. Profile with browser DevTools

## Best Practices

1. **User Experience First**: Don't block typing
2. **Explain Detections**: Clear messaging
3. **Allow Dismissal**: Users can override
4. **Log Everything**: Learn from usage
5. **Regular Updates**: Evolve with new patterns

---

*For implementation details, see the archived documentation in `Docs/archive/`*