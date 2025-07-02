# CLAUDE.md

## Standard Workflow

1. First think through the problem, read the codebase for relevant files, and write a plan to tasks/todo.md.
2. The plan should have a list of todo items that you can check off as you complete them
3. Before you begin working, check in with me and I will verify the plan.
4. Then, begin working on the todo items, marking them as complete as you go.
5. Please every step of the way just give me a high level explanation of what changes you made
6. Make every task and code change you do as simple as possible. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity.
7. Finally, add a review section to the todo.md file with a summary of the changes you made and any other relevant information.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Next.js 15 application with React 19 that provides AI-powered text autocomplete functionality for personal bio completion. The app integrates with Ollama (local AI model using Gemma 3 12B) and includes support for Weaviate vector database.

## Key Commands

### Development

- `npm run dev` - Start development server with Turbopack (runs on http://localhost:3000)
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint checks

### External Services

- **Ollama**: Must be running locally on port 11434
  - Start: `ollama serve`
  - Pull model: `ollama pull gemma3:12b`
  - Verify: `ollama list`
- **Docker services** (optional): `docker-compose up -d` (Weaviate + transformers)

## Architecture Overview

### Core Technologies

- **Next.js 15.3.3** with App Router
- **React 19** with TypeScript
- **Tailwind CSS v4** for styling
- **React Hook Form** for form management
- **Server Actions** for AI integration

### Key Files and Patterns

**AI Integration** (`actions/ai-text.ts`):

- Server actions communicate with Ollama API
- `askOllamaCompletationAction` - Main AI completion function
- Uses streaming responses for real-time suggestions

**Form Management** (`components/Form.tsx`):

- Main form component using React Hook Form
- Integrates with custom autocomplete hook
- Handles user input and displays AI suggestions

**Custom Hook** (`hooks/useFormAutocomplete.tsx`):

- Debounced autocomplete functionality
- Prevents excessive API calls
- Manages suggestion state

**Type Definitions** (`type/`):

- Contains TypeScript interfaces and types
- Ensures type safety across the application

### Environment Configuration

The app requires `.env.local` with:

```
OLLAMA_PATH_API=http://127.0.0.1:11434/api
```

### Project Structure

```
actions/        # Server actions for AI integration
app/           # Next.js app router pages and layouts
components/    # React components
data/          # Static data files (bios)
hooks/         # Custom React hooks
lib/           # Utility functions and configurations
type/          # TypeScript type definitions
```

## Important Notes

- No test framework is configured - consider adding tests before major changes
- ESLint is configured with Next.js recommended rules
- The app uses the new Tailwind CSS v4 with PostCSS
- Weaviate integration exists but appears optional (via Docker)
- Main functionality depends on Ollama running locally
