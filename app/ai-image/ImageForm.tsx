"use client";

import { analyzeImage } from "@/actions/ai-vision";
import { useState } from "react";
import ImageResultCard from "./ImageResultCard";
import Image from "next/image";

interface ImageAnalysis {
  file: File;
  result: { naked: boolean; kids: boolean } | string;
}

export default function ImageForm() {
  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [results, setResults] = useState<ImageAnalysis[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [currentProcessing, setCurrentProcessing] = useState<string>("");

  async function handleSubmit() {
    setLoading(true);
    setProgress(0);
    setCurrentProcessing("");

    const analysisResults: ImageAnalysis[] = [];
    const totalFiles = imageFiles.length;

    for (let i = 0; i < imageFiles.length; i++) {
      const file = imageFiles[i];
      setCurrentProcessing(file.name);

      const formData = new FormData();
      formData.append("image", file);

      const response = await analyzeImage(formData);
      const result = response?.result || "Error procesando imagen.";

      analysisResults.push({ file, result });

      // Update progress
      const progressPercentage = ((i + 1) / totalFiles) * 100;
      setProgress(progressPercentage);
    }

    setResults(analysisResults);
    setLoading(false);
    setProgress(0);
    setCurrentProcessing("");
  }

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      const maxSize = 1 * 1024 * 1024; // 1MB

      const validFiles = files.filter((file) => {
        if (file.size > maxSize) {
          setError(
            `The File ${file.name} is more them 1mb please select other image.`
          );
          return false;
        }
        return true;
      });

      setImageFiles(validFiles);
      setResults([]);
      setError(null); // Clear any previous errors when new files are selected
    }
  }

  const removeImage = (indexToRemove: number) => {
    const updatedFiles = imageFiles.filter(
      (_, index) => index !== indexToRemove
    );
    setImageFiles(updatedFiles);
    setResults([]);
  };

  return (
    <>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleSubmit();
        }}
        className="flex flex-col gap-4 w-full max-w-[500px] py-6 px-10 bg-white border border-gray-200 rounded-[10px] mx-auto"
      >
        {/* Logo */}
        <div className="flex justify-center mb-1">
          <Image
            src="/images/logo-swing.svg"
            alt="Swing Logo"
            width={120}
            height={40}
            className="h-20 w-auto"
          />
        </div>

        {/* Header and description */}
        <div className="text-center mb-3">
          <h2 className="text-2xl font-semibold text-gray-900 mb-3">
            Let AI Analyze Your Images
          </h2>
          <p className="text-gray-500 text-sm">
            Let AI help you determine if images contain nudity or children.
          </p>
        </div>

        {/* Custom File Upload Area */}
        <div className="relative">
          <input
            type="file"
            name="image"
            accept="image/*"
            multiple
            required
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
            onChange={handleChange}
            title="Upload images to analyze less than 1MB each"
          />
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-gray-400 transition-colors duration-200 bg-gray-50 hover:bg-gray-100">
            {/* Cloud Upload Icon */}
            <div className="flex justify-center mb-4">
              <svg
                className="w-12 h-12 text-gray-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
            </div>

            <div className="text-gray-600">
              <p className="text-lg font-medium mb-1">Upload a File</p>
              <p className="text-sm text-gray-500">Drag and drop files here</p>
              <p className="text-xs text-gray-400 mt-2">
                Maximum file size: 1MB each
              </p>
            </div>
          </div>
        </div>

        {/* Image Preview Section */}
        {imageFiles.length > 0 && (
          <div className="mt-4">
            <h3 className="text-sm font-medium text-gray-700 mb-3">
              Selected Images ({imageFiles.length})
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
              {imageFiles.map((file, index) => (
                <div key={index} className="relative group">
                  <div className="aspect-square relative overflow-hidden rounded-lg border border-gray-200 bg-gray-50">
                    <Image
                      src={URL.createObjectURL(file)}
                      alt={`Preview ${index + 1}`}
                      fill
                      className="object-cover"
                    />
                    {/* Remove button */}
                    <button
                      type="button"
                      onClick={() => removeImage(index)}
                      className="absolute top-1 right-1 bg-red-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs hover:bg-red-600 transition-colors opacity-0 group-hover:opacity-100"
                      title="Remove image"
                    >
                      ×
                    </button>
                  </div>
                  <p className="text-xs text-gray-500 mt-1 truncate">
                    {file.name}
                  </p>
                  <p className="text-xs text-gray-400">
                    {(file.size / 1024).toFixed(1)} KB
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        <button
          type="submit"
          disabled={loading || imageFiles.length === 0 || error !== null}
          className="cursor-pointer w-full bg-gray-900 text-white font-semibold py-2 px-4 rounded hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-opacity-50 "
        >
          Submit
        </button>

        {/* Progress Bar */}
        {loading && (
          <div className="mt-4">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-gray-700">
                Processing images...
              </span>
              <span className="text-sm text-gray-500">
                {Math.round(progress)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300 ease-out"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
            {currentProcessing && (
              <p className="text-xs text-gray-500 mt-2">
                Currently processing: {currentProcessing}
              </p>
            )}
          </div>
        )}

        {error && <span className="text-red-500"> {error}</span>}

        {results.length === 0 && !loading && (
          <span className="text-gray-500 text-center">
            No results yet. Please upload images to analyze.
          </span>
        )}

        <div className="text-center text-sm text-gray-500 mt-4">
          Brought to you by{" "}
          <a
            href="https://swing.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-700 font-semibold hover:text-yellow-500"
          >
            Swing.com
          </a>
        </div>
      </form>

      {results.map(({ file, result }, idx) => (
        <ImageResultCard
          key={idx}
          imageUrl={URL.createObjectURL(file)}
          result={result}
        />
      ))}
    </>
  );
}
