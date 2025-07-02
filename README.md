# AI App Autocomplete

An intelligent text autocomplete application built with Next.js that uses Ollama's Gemma 3 12B model to provide AI-powered text suggestions for personal bio completion.

## Features

- 🤖 AI-powered text autocomplete using Ollama model Gemma 3 12B
- ⚡ Real-time suggestions as you type
- 🎯 Specialized for personal bio completion
- 🚀 Built with Next.js 15 and React 19
- 📱 Responsive design with Tailwind CSS
- 🔧 TypeScript support

## Prerequisites

Before you begin, ensure you have the following installed:

- **Node.js** (version 18 or higher)
- **npm** or **yarn**
- **Ollama** (for running the LLaMA 3.2 model locally)

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
   OLLAMA_PATH_API=http://localhost:11434/api
   ```

## Usage

1. **Start the development server**:

   ```bash
   npm run dev
   # or
   yarn dev
   ```

2. **Open your browser** and navigate to `http://localhost:3000`

3. **Start typing** in the text input field to begin your personal bio

4. **Watch the magic happen** as the AI suggests continuations for your text in real-time

## How It Works

The application uses:

- **React Hook Form** for form management
- **Custom debounced hook** (`useFormAutocomplete`) to prevent excessive API calls
- **Server Actions** ([`askOllamaCompletationAction`](actions/ai-text.ts)) to communicate with Ollama
- **Gemma 3 12B model** for generating contextual text completions

The AI is specifically prompted to continue personal bios with short, relevant sentences without quotes or explanations.

## Project Structure

```
ai-app-oucomplete/
├── actions/
│   └── ai.ts              # Server actions for AI integration
├── app/
│   ├── layout.tsx         # Root layout
│   ├── page.tsx           # Main page component
│   └── globals.css        # Global styles
├── components/
│   └── Form.tsx           # Main form component
├── hooks/
│   └── useFormAutocomplete.tsx # Custom autocomplete hook
├── lib/                   # Utility libraries
└── public/               # Static assets
```

## Available Scripts

- `npm run dev` - Start development server with Turbopack
- `npm run build` - Build the application for production
- `npm run start` - Start the production server
- `npm run lint` - Run ESLint for code quality

## Configuration

### Ollama Configuration

The application expects Ollama to be running on `http://localhost:11434` by default. You can modify this in your `.env.local` file:

```env
OLLAMA_PATH_API=http://your-ollama-host:port/api
```

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

## Dependencies

### Main Dependencies

- **Next.js 15.3.3** - React framework
- **React 19** - UI library
- **OpenAI 5.5.0** - AI integration utilities
- **React Hook Form 7.58.0** - Form management
- **use-debounce 10.0.5** - Debouncing utility

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

## Acknowledgments

- [Ollama](https://ollama.ai) for providing local AI model hosting
- [Google Gemma](https://ai.google.dev/gemma) for the powerful language model
- [Next.js](https://nextjs.org) team for the excellent framework
