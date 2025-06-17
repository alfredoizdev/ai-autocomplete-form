"use client";

import useFormAutocomplete from "@/hooks/useFormAutocomplete";

const Form = () => {
  const {
    register,
    handleSubmit,
    errors,
    onSubmit,
    textareaRef,
    suggestion,
    isPending,
    handleKeyDown,
    promptValue,
  } = useFormAutocomplete();

  console.log("sugestion", suggestion);

  return (
    <form
      onSubmit={handleSubmit(onSubmit)}
      className="flex flex-col gap-4 w-full max-w-md p-6 bg-white shadow-md rounded-none mx-auto"
    >
      <div className="mb-3">
        <label
          htmlFor="name"
          className="block text-sm font-medium text-gray-700"
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
          className="mt-1 text-gray-900 placeholder:text-gray-400 block w-full p-2 border border-gray-300 rounded-none shadow-sm focus:ring-gray-500 focus:border-gray-500"
          placeholder="Enter your name"
        />
        {errors.name && (
          <p className="text-red-500 text-sm mt-1">{errors.name.message}</p>
        )}
      </div>
      <div className="relative w-full">
        <label
          htmlFor="prompt"
          className="block text-sm font-medium text-gray-700 mb-1"
        >
          Bio Description:
        </label>

        {/* Container for textarea with overlay */}
        <div className="relative">
          {/* Background textarea for suggestion text */}
          <textarea
            value={promptValue + (suggestion ? " " + suggestion : "")}
            readOnly
            rows={4}
            className="absolute inset-0 w-full p-2 border border-gray-300 rounded-none shadow-sm resize-none text-gray-400 pointer-events-none whitespace-pre-wrap"
            style={{ zIndex: 1 }}
          />

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
            rows={4}
            className="relative bg-transparent text-black placeholder:text-gray-400 block w-full p-2 border border-gray-300 rounded-none shadow-sm focus:ring-gray-500 focus:border-gray-500 resize-none"
            placeholder="Write a brief description about yourself..."
            style={{ zIndex: 2 }}
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
        className="cursor-pointer w-full bg-gray-900 text-white font-semibold py-2 px-4 rounded-none hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-opacity-50"
      >
        Submit
      </button>
    </form>
  );
};

export default Form;
