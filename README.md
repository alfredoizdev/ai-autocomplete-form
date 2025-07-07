# AI Autocomplete & Spell Check

A sophisticated AI-powered text assistant built with Next.js 15 and React 19 that combines intelligent autocomplete, advanced spell checking, and smart feature coordination. Uses Ollama's Gemma 3 12B model for contextual text suggestions and typo-js with custom dictionaries for professional-grade spell checking.

## Features

### 🤖 Advanced AI Autocomplete
- **Context-aware suggestions** using Ollama Gemma 3 12B model
- **Smart triggering** - activates after 5+ words with 500ms debouncing
- **Inline suggestion display** with layered textarea approach
- **Tab key acceptance** with intelligent spacing
- **Vector database integration** for enhanced contextual understanding

### ✍️ Professional Spell Checking
- **typo-js integration** with English Hunspell dictionaries
- **Custom dictionary system** with localStorage persistence
- **Word mapping functionality** - learns user corrections
- **Contraction handling** - automatically suggests 35+ common contractions
- **Click-to-correct interface** with intelligent popup positioning
- **Performance optimized** with 800ms debouncing and suggestion caching

### 🎯 Intelligent Feature Coordination
- **Text Feature Coordinator** prevents conflicts between autocomplete and spellcheck
- **Adaptive timing system** with feature-specific lock durations
- **Seamless multi-feature operation** - autocomplete and spellcheck work together
- **Memory management** with proper cleanup and state handling

### 🚀 Modern Architecture
- **Next.js 15** with App Router and Server Actions
- **React 19** with advanced hooks and TypeScript
- **Tailwind CSS v4** with responsive design
- **Mobile-optimized UX** with 16px fonts to prevent zoom
- **Dynamic textarea resizing** with smooth 300ms transitions
- **Performance optimized** with progressive debouncing and memoization

## Prerequisites

Before you begin, ensure you have the following installed:

- **Node.js** (version 18 or higher)
- **npm** or **yarn**
- **Ollama** (for running the Gemma 3 12B model locally)

## Ollama Setup

1. **Install Ollama** on your system:

   - Visit [Ollama's official website](https://ollama.ai) and download the installer for your operating system
   - Follow the installation instructions

2. **Pull the Gemma 3 12B model**:

   ```bash
   ollama pull gemma3:12b
   ```

3. **Start Ollama service**:

   ```bash
   ollama serve
   ```

4. **Verify the installation**:
   ```bash
   ollama list
   ```
   You should see `gemma3:12b` in the list of available models.

## Installation

1. **Clone the repository**:

   ```bash
   git clone <your-repository-url>
   cd ai-autocomplete-spellcheck
   ```

2. **Install dependencies**:

   ```bash
   npm install
   # or
   yarn install
   ```

3. **Set up environment variables**:
   Create a `.env.local` file in the root directory:
   ```env
   OLLAMA_PATH_API=http://127.0.0.1:11434/api
   ```

4. **Set up spell check dictionaries**:
   The application includes English dictionaries in the `public/dictionaries/en_US/` folder:
   - `en_US.aff` - Affix rules file
   - `en_US.dic` - Dictionary words file

## Usage

1. **Start the development server**:

   ```bash
   npm run dev
   # or
   yarn dev
   ```

2. **Open your browser** and navigate to `http://localhost:3000`

3. **Start typing** in the bio description field to experience:
   - **AI Autocomplete**: After 5+ complete words, AI suggestions appear as gray inline text
   - **Spell Check**: Misspelled words show red dotted underlines with click-to-correct
   - **Contraction Help**: Type "dont" and see automatic "don't" suggestions
   - **Custom Dictionary**: Add frequently used words to your personal dictionary
   - **Auto-capitalization**: Smart sentence formatting applied in real-time

4. **Advanced interactions**:
   - Press **Tab** to accept AI suggestions with proper spacing
   - **Click** misspelled words for instant popup with corrections
   - **Right-click** words to add them to your custom dictionary
   - **Type naturally** - the Text Feature Coordinator prevents interference
   - **Mobile-friendly** - all features work seamlessly on touch devices

## How It Works

### Sophisticated Hook Architecture
The application uses a **4-hook system** for optimal performance and feature coordination:

1. **`useFormAutocomplete`** - Main form logic with AI integration
2. **`useSpellCheck`** - Core spell checking with typo-js and custom dictionaries
3. **`useDebouncedSpellCheck`** - Performance-optimized wrapper with caching
4. **`useTextFeatureCoordinator`** - Prevents conflicts between features

### Text Feature Coordination System
The **Text Feature Coordinator** manages three text features:
- **AUTOCOMPLETE** - AI-powered text suggestions
- **SPELLCHECK** - Real-time spell checking
- **CAPITALIZATION** - Smart sentence formatting

**Key Coordination Features:**
- **Feature locking** with adaptive durations (200ms for autocomplete, configurable for others)
- **Coexistence support** - autocomplete and spellcheck can run simultaneously
- **Memory management** - automatic cleanup of timeouts and state
- **Conflict prevention** - ensures features don't interfere with each other

### AI Autocomplete System
- **Layered textarea approach** for inline suggestion display
- **Word completion detection** - waits for complete words before suggesting
- **Progressive debouncing** - 500ms delay with smart triggering after 5+ words
- **Vector database integration** with Weaviate for enhanced context
- **Server Actions** communicate with Ollama Gemma 3 12B model
- **Intelligent spacing** - proper handling of tab acceptance and word boundaries

### Advanced Spell Check System
- **typo-js integration** with English Hunspell dictionaries (`en_US.aff`, `en_US.dic`)
- **Custom dictionary service** with localStorage persistence
  - Add/remove custom words
  - Word mapping system for learning corrections
  - Persistent storage across browser sessions
- **Contraction handling** - automatic suggestions for 35+ common contractions:
  - `dont` → `don't`, `do not`
  - `youre` → `you're`, `you are`
  - `wont` → `won't`, `will not`
  - And many more...
- **Click-to-correct interface** with intelligent popup positioning
- **Performance optimizations:**
  - 800ms debounce delay
  - Suggestion caching to prevent repeated lookups
  - Memoization to prevent unnecessary re-renders
  - Progressive debouncing based on text length

### Smart Auto-Capitalization
- **Sentence boundary detection** with proper punctuation handling
- **Real-time processing** without disrupting typing flow
- **Context-aware rules:**
  - First letter of sentences
  - After periods, exclamation marks, question marks
  - Pronoun "I" capitalization
- **Seamless integration** with autocomplete and spell check features

## Project Structure

```
ai-autocomplete-spellcheck/
├── actions/
│   └── ai-text.ts                    # Server actions for AI integration
├── app/
│   ├── layout.tsx                    # Root layout
│   ├── page.tsx                      # Main page component
│   ├── image-analyzer/               # Image analysis feature
│   └── globals.css                   # Global styles with Tailwind v4
├── components/
│   ├── Form.tsx                      # Main form with layered textarea
│   ├── SpellCheckPopup.tsx           # Interactive spell suggestion popup
│   └── Navbar.tsx                    # Navigation component
├── hooks/                            # Sophisticated 4-hook architecture
│   ├── useFormAutocomplete.tsx       # Main form logic with AI integration
│   ├── useSpellCheck.tsx             # Core spell check with contractions
│   ├── useDebouncedSpellCheck.tsx    # Performance-optimized wrapper
│   └── useTextFeatureCoordinator.tsx # Feature conflict prevention
├── lib/
│   ├── customDictionary.ts           # Custom dictionary service
│   ├── openai.ts                     # OpenAI integration
│   └── utils.ts                      # Utility functions
├── data/
│   └── bios.json                     # Bio examples for AI training
├── public/
│   ├── dictionaries/
│   │   └── en_US/                    # Hunspell dictionaries
│   │       ├── en_US.aff            # Affix rules
│   │       └── en_US.dic            # Dictionary words
│   └── images/                       # Static images
├── types/                            # TypeScript type definitions
└── [Configuration Files]
    ├── next.config.js                # Next.js configuration
    ├── tailwind.config.js            # Tailwind CSS v4 config
    └── tsconfig.json                 # TypeScript configuration
```

## Available Scripts

- `npm run dev` - Start development server with Turbopack
- `npm run build` - Build the application for production
- `npm run start` - Start the production server
- `npm run lint` - Run ESLint for code quality

## Configuration

### Ollama Configuration

The application expects Ollama to be running on `http://127.0.0.1:11434` by default. You can modify this in your `.env.local` file:

```env
OLLAMA_PATH_API=http://your-ollama-host:port/api
```

### Spell Check Configuration

Spell checking features advanced configuration options:

**Core Settings:**
- **Debounce delay**: 800ms for optimal performance
- **Dictionary caching**: Suggestions cached to prevent repeated lookups
- **Custom dictionary**: Persistent localStorage-based word storage
- **Word mapping**: Learn and remember user corrections

**Contraction Handling:**
The system automatically handles 35+ common contractions:
```javascript
// Examples of automatic contraction suggestions
dont     → don't, do not
youre    → you're, you are
wont     → won't, will not
havent   → haven't, have not
its      → it's, it is, it has
```

**Custom Dictionary Features:**
- **Add words**: Right-click misspelled words to add to dictionary
- **Word mappings**: System learns your preferred corrections
- **Persistent storage**: Dictionary survives browser restarts
- **Import/Export**: Backup and restore custom words

### AI Model Configuration

**Default Setup:**
- **Primary Model**: `gemma3:12b` via Ollama (local inference)
- **Vector Database**: Weaviate for contextual enhancement (optional)
- **OpenAI Integration**: Available as fallback option

**Configuration Options:**
```javascript
// In actions/ai-text.ts - modify model settings
const model = "gemma3:12b";  // Change model here
const maxTokens = 50;        // Adjust response length
const temperature = 0.7;     // Control creativity
```

**Performance Tuning:**
- **Progressive debouncing**: Adapts timing based on text length
- **Context window**: Optimized for bio completion tasks
- **Feature coordination**: Automatic AI pause during spell check operations

## Troubleshooting

### Common Issues

1. **Ollama not responding**:

   - Ensure Ollama service is running: `ollama serve`
   - Check if the model is pulled: `ollama list`
   - Verify the API endpoint in `.env.local`

2. **Model not found**:

   - Pull the model: `ollama pull gemma3:12b`
   - Restart Ollama service

3. **Slow responses**:
   - This is normal for local AI models
   - Consider using a more powerful machine or adjusting the debounce delay

4. **Spell check not working**:
   - Ensure dictionary files are present in `public/dictionaries/en_US/`
   - Check browser console for any dictionary loading errors
   - Try refreshing the page to reinitialize the spell checker
   - Clear localStorage if custom dictionary is corrupted: `localStorage.clear()`

5. **Performance issues while typing**:
   - The app uses sophisticated debouncing to prevent lag
   - Text Feature Coordinator manages feature conflicts automatically
   - If typing feels slow, check if other applications are using high CPU
   - Progressive debouncing adapts timing based on text length

6. **Features conflicting with each other**:
   - The Text Feature Coordinator should prevent this automatically
   - If issues persist, try refreshing the page to reset coordinator state
   - Check browser console for any coordination errors

## Dependencies

### Main Dependencies

- **Next.js 15.3.3** - React framework with App Router and Server Actions
- **React 19** - Latest UI library with advanced hooks
- **React Hook Form 7.58.0** - Sophisticated form management
- **typo-js 1.2.5** - Spell checking with Hunspell dictionary support
- **use-debounce 10.0.5** - Performance optimization utility
- **weaviate-client 3.6.2** - Vector database for contextual AI
- **openai 5.5.0** - OpenAI API integration (optional)

### Development Dependencies

- **TypeScript 5** - Comprehensive type safety
- **Tailwind CSS 4** - Latest version with PostCSS
- **ESLint 9** - Advanced code linting with Next.js config

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Key Features In Detail

### Smart Spell Checking
- **Professional-grade accuracy** using typo-js with Hunspell dictionaries
- **Click-to-correct interface** - no right-click needed, just click misspelled words
- **Custom dictionary system** with persistent localStorage storage
- **Word mapping functionality** - learns and remembers your corrections
- **35+ contraction suggestions** - automatic handling of missing apostrophes
- **Performance optimized** with 800ms debouncing and intelligent caching
- **Non-intrusive visual indicators** - dotted red underlines without typing disruption

### AI-Powered Autocomplete  
- **Context-aware suggestions** using Gemma 3 12B model for relevance
- **Layered textarea approach** for seamless inline suggestion display
- **Smart triggering system** - activates after 5+ complete words
- **Vector database integration** with Weaviate for enhanced understanding
- **Intelligent spacing logic** - proper tab acceptance and word boundaries
- **Progressive debouncing** - 500ms delay with adaptive timing

### Text Feature Coordination
- **Conflict prevention system** - prevents autocomplete and spellcheck interference
- **Adaptive feature locking** with customizable durations per feature type
- **Memory management** - automatic cleanup of timeouts and state
- **Coexistence support** - allows multiple features to work simultaneously
- **Performance monitoring** - tracks and optimizes feature interactions

### Mobile-Optimized Experience
- **16px font sizes** prevent unwanted mobile browser zoom
- **Dynamic textarea resizing** with smooth 300ms transitions
- **Touch-friendly interface** with properly sized click targets
- **Responsive design** adapts to all screen sizes
- **Optimized debouncing** for mobile keyboard behavior
- **Battery-conscious processing** with intelligent feature management

## Acknowledgments

- [Ollama](https://ollama.ai) for providing local AI model hosting
- [Google Gemma](https://ai.google.dev/gemma) for the powerful language model
- [typo-js](https://github.com/cfinke/Typo.js) for excellent spell checking capabilities
- [Hunspell](http://hunspell.github.io/) for comprehensive dictionary support
- [Next.js](https://nextjs.org) team for the excellent framework
