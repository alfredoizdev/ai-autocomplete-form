# AI Autocomplete & Spell Check

An intelligent text autocomplete and spell checking application built with Next.js that uses Ollama's Gemma 3 12B model for AI-powered text suggestions and typo-js for accurate spell checking with a custom click-to-correct interface.

## Features

- 🤖 AI-powered text autocomplete using Ollama model Gemma 3 12B
- ✍️ Advanced spell checking with typo-js and Hunspell dictionaries
- 🎯 Click-to-correct spell check interface with popup suggestions
- ⚡ Real-time suggestions as you type with optimized performance
- 🔤 Automatic capitalization for proper sentence structure
- 🚀 Built with Next.js 15 and React 19
- 📱 Responsive design with Tailwind CSS v4
- 🔧 TypeScript support with comprehensive type safety
- ⚡ Performance optimized with debouncing and memoization

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
   cd ai-app-oucomplete
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
   - **AI Autocomplete**: When you have 3+ words in a sentence, AI suggestions appear in gray text
   - **Spell Check**: Misspelled words appear with red dotted underlines
   - **Auto-capitalization**: Proper sentence capitalization is applied automatically

4. **Interact with features**:
   - Press **Tab** to accept AI suggestions
   - **Click** misspelled words to see correction suggestions in a popup
   - Type normally - the app won't interfere with your writing flow

## How It Works

### AI Autocomplete System
- **React Hook Form** for form management
- **Custom debounced hook** (`useFormAutocomplete`) to prevent excessive API calls
- **Server Actions** ([`askOllamaCompletationAction`](actions/ai-text.ts)) to communicate with Ollama
- **Gemma 3 12B model** for generating contextual text completions
- AI triggers after 3+ words in a sentence with 2-second debounce

### Spell Check System
- **typo-js library** with Hunspell English dictionaries for accurate spell checking
- **Custom debounced spell check hook** (`useDebouncedSpellCheck`) with 800ms delay
- **Performance optimizations** including memoization and suggestion caching
- **Click-to-correct interface** with popup suggestions positioned intelligently
- **Overlay system** for visual spell check indicators without disrupting typing

### Auto-capitalization
- **Smart sentence detection** with proper punctuation handling
- **Real-time capitalization** for first letters and after sentence endings
- **Context-aware processing** that works seamlessly with autocomplete and spell check

## Project Structure

```
ai-autocomplete-spellcheck/
├── actions/
│   └── ai-text.ts         # Server actions for AI integration
├── app/
│   ├── layout.tsx         # Root layout
│   ├── page.tsx           # Main page component
│   └── globals.css        # Global styles
├── components/
│   ├── Form.tsx           # Main form component with spell check overlay
│   └── SpellCheckPopup.tsx # Popup component for spell suggestions
├── hooks/
│   ├── useFormAutocomplete.tsx    # Custom autocomplete hook
│   ├── useSpellCheck.tsx          # Core spell checking functionality
│   └── useDebouncedSpellCheck.tsx # Performance-optimized spell check
├── lib/                   # Utility libraries
├── public/
│   └── dictionaries/
│       └── en_US/         # English spell check dictionaries
└── type/                  # TypeScript type definitions
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

Spell checking is automatically enabled with English dictionaries. The system includes:
- **Debounce delay**: 800ms to optimize performance
- **Dictionary caching**: Suggestions are cached to avoid repeated lookups
- **Performance optimization**: Memoization prevents unnecessary re-renders

### Model Configuration

The application is configured to use the `gemma3:12b` model. You can change this in the [`askOllamaCompletationAction`](actions/ai-text.ts) function if needed.

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

5. **Performance issues while typing**:
   - The app uses debouncing to prevent lag
   - If typing feels slow, check if other applications are using high CPU
   - Spell check delay is optimized at 800ms for best performance

## Dependencies

### Main Dependencies

- **Next.js 15.3.3** - React framework
- **React 19** - UI library
- **React Hook Form 7.58.0** - Form management
- **typo-js** - Spell checking library with Hunspell support
- **use-debounce 10.0.5** - Debouncing utility for performance optimization

### Development Dependencies

- **TypeScript 5** - Type safety
- **Tailwind CSS 4** - Styling framework
- **ESLint 9** - Code linting

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
- **Accurate Detection**: Uses typo-js with Hunspell dictionaries for professional-grade spell checking
- **Click-to-Correct**: No right-click needed - simply click misspelled words for suggestions
- **Performance Optimized**: 800ms debounce prevents lag while typing
- **Non-Intrusive**: Dotted red underlines that don't interfere with the writing experience

### AI-Powered Autocomplete  
- **Context Aware**: Understands sentence structure and provides relevant continuations
- **Smart Triggering**: Only activates after 3+ words to avoid premature suggestions
- **Seamless Integration**: Works harmoniously with spell check and auto-capitalization

### Auto-Capitalization
- **Intelligent Rules**: Proper sentence beginnings, "I" pronouns, and post-punctuation capitalization
- **Real-Time Processing**: Applies corrections as you type without disrupting flow
- **Context Sensitive**: Understands when to apply different capitalization rules

## Acknowledgments

- [Ollama](https://ollama.ai) for providing local AI model hosting
- [Google Gemma](https://ai.google.dev/gemma) for the powerful language model
- [typo-js](https://github.com/cfinke/Typo.js) for excellent spell checking capabilities
- [Hunspell](http://hunspell.github.io/) for comprehensive dictionary support
- [Next.js](https://nextjs.org) team for the excellent framework
