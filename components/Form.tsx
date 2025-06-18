"use client";

import Image from "next/image";
import useFormAutocomplete from "@/hooks/useFormAutocomplete";

const Form = () => {
  const {
    register,
    handleSubmit,
    errors,
    onSubmit,
    textareaRef,
    measureRef,
    suggestion,
    isPending,
    handleKeyDown,
    textareaHeight,
    promptValue,
  } = useFormAutocomplete();

  console.log("sugestion", suggestion);

  return (
    <form
      onSubmit={handleSubmit(onSubmit)}
      className="flex flex-col gap-4 w-full max-w-[500px] p-8 bg-white shadow-md rounded-[8px] mx-auto "
    >
      {/* Logo */}
      <div className="flex justify-center mb-6">
        <Image
          src="/images/logo-swing.svg"
          alt="Swing Logo"
          width={120}
          height={40}
          className="h-20 w-auto"
        />
      </div>

      {/* Header and description */}
      <div className="text-center mb-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-2">
          Enter you Bio
        </h2>
        <p className="text-gray-500">Sign in to your account</p>
      </div>

      <div className="mb-3">
        <label
          htmlFor="name"
          className="block text-sm font-semibold text-gray-700"
        >
          Name:
        </label>
        <input
          {...register("name", {
            required: "Name is required",
            maxLength: {
              value: 50,
              message: "Name cannot exceed 50 characters",
            },
          })}
          type="text"
          id="name"
          className="mt-1 text-gray-900 placeholder:text-gray-400 block w-full h-[48px] p-2 border border-gray-200 rounded-[6px] focus:ring-gray-500 focus:border-gray-500"
          placeholder="Enter your name"
        />
        {errors.name && (
          <p className="text-red-500 text-sm mt-1">{errors.name.message}</p>
        )}
      </div>
      <div className="relative w-full">
        <label
          htmlFor="prompt"
          className="block text-sm font-semibold text-gray-700 mb-1"
        >
          Bio Description:
        </label>

        {/* Hidden textarea for height measurement */}
        <textarea
          ref={measureRef}
          className="absolute opacity-0 pointer-events-none -z-10 w-full p-2 border border-gray-200 rounded-[6px] resize-none whitespace-pre-wrap"
          style={{
            position: "absolute",
            left: "-9999px",
            top: "-9999px",
          }}
          tabIndex={-1}
        />

        {/* Container for textarea with overlay */}
        <div className="relative">
          {/* Background div that shows user text + suggestion */}
          <div
            className="absolute inset-0 w-full p-2 border border-gray-200 rounded resize-none whitespace-pre-wrap pointer-events-none transition-all duration-300 ease-out"
            style={{
              height: textareaHeight,
              minHeight: "96px",
              fontSize: "14px",
              lineHeight: "1.5",
              fontFamily:
                "ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif",
              color: "transparent",
              zIndex: 1,
            }}
          >
            <span style={{ color: "transparent" }}>{promptValue}</span>
            {suggestion && (
              <span style={{ color: "#9CA3AF" }}>
                {promptValue && !promptValue.endsWith(" ") ? " " : ""}
                {suggestion}
              </span>
            )}
          </div>

          {/* Actual input textarea */}
          <textarea
            id="prompt"
            {...register("prompt", {
              required: "Description is required",
              maxLength: {
                value: 200,
                message: "Description cannot exceed 200 characters",
              },
            })}
            ref={(e) => {
              register("prompt").ref(e);
              textareaRef.current = e;
            }}
            onKeyDown={handleKeyDown}
            className="relative bg-transparent text-black placeholder:text-gray-400 block w-full p-2 border border-gray-200 rounded focus:ring-gray-500 focus:border-gray-500 resize-none transition-all duration-300 ease-out"
            placeholder="Write a brief description about yourself..."
            style={{
              height: textareaHeight,
              minHeight: "96px",
              fontSize: "14px",
              lineHeight: "1.5",
              fontFamily:
                "ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif",
              zIndex: 2,
            }}
          />
        </div>

        {/* Instructional message */}
        <div className="mt-3 text-xs text-gray-500 min-h-[16px]">
          {suggestion ? (
            <span>
              💡 Press{" "}
              <kbd className="px-1 py-0.5 bg-gray-100 border border-gray-300 rounded text-xs">
                Tab
              </kbd>{" "}
              to accept suggestion
            </span>
          ) : isPending ? (
            <span className="animate-pulse">🤔 Thinking of suggestions...</span>
          ) : null}
        </div>

        {errors.prompt && (
          <p className="text-red-500 text-sm mt-1">{errors.prompt.message}</p>
        )}
      </div>
      <button
        type="submit"
        disabled={isPending}
        className="cursor-pointer w-full bg-gray-900 text-white font-semibold py-2 px-4 rounded hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-opacity-50"
      >
        Submit
      </button>
      <div className="text-center text-sm text-gray-500 mt-4">
        Brought to you by{" "}
        <a
          href="https://swing.com"
          target="_blank"
          rel="noopener noreferrer"
          className="text-gray-700 hover:text-gray-900 underline"
        >
          Swing.com
        </a>
      </div>
    </form>
  );
};

export default Form;
