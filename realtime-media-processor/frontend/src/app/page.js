"use client";

import Image from "next/image";
import { useState, useRef } from "react";
import { useMutation } from "@tanstack/react-query";
import useAppStore from "@/stores/appStore";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export default function Home() {
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

  const { isLoading, message, predictionResult, setLoading, setMessage, setPredictionResult, clearState } = useAppStore();

  const predictImageMutation = useMutation({
    mutationFn: async (formData) => {
      setLoading(true);
      setMessage("Processing image...");
      const response = await fetch(`${API_BASE_URL}/predict/image`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to process image.");
      }
      return response.json();
    },
    onSuccess: (data) => {
      setMessage("Image processed successfully!");
      setPredictionResult(data.prediction);
      setLoading(false);
    },
    onError: (error) => {
      setMessage(`Error: ${error.message}`);
      setLoading(false);
      setPredictionResult(null);
    },
  });

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setImagePreview(URL.createObjectURL(file));
      clearState();
    } else {
      setSelectedFile(null);
      setImagePreview(null);
      clearState();
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      setMessage("Please select an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    predictImageMutation.mutate(formData);
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
      <h1 className="text-4xl font-bold text-gray-800 mb-6">Realtime Media Processor</h1>

      <div className="bg-white p-8 rounded-lg shadow-md w-full max-w-2xl">
        <h2 className="text-2xl font-semibold text-gray-700 mb-4">Image Processing</h2>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="file-upload" className="block text-sm font-medium text-gray-700">
              Upload Image
            </label>
            <input
              id="file-upload"
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              ref={fileInputRef}
              className="mt-1 block w-full text-sm text-gray-500
              file:mr-4 file:py-2 file:px-4
              file:rounded-full file:border-0
              file:text-sm file:font-semibold
              file:bg-blue-50 file:text-blue-700
              hover:file:bg-blue-100"
            />
          </div>

          {imagePreview && (
            <div className="mt-4">
              <h3 className="text-lg font-medium text-gray-700">Image Preview:</h3>
              <Image src={imagePreview} alt="Image Preview" width={300} height={300} className="mt-2 rounded-md object-contain max-h-80" />
            </div>
          )}

          <button
            type="submit"
            disabled={!selectedFile || isLoading}
            className={`w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-white font-semibold ${
              (!selectedFile || isLoading) ? "bg-gray-400 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            }`}
          >
            {isLoading ? "Processing..." : "Process Image"}
          </button>
        </form>

        {message && (
          <p className={`mt-4 text-center ${predictImageMutation.isError ? "text-red-600" : "text-gray-600"}`}>
            {message}
          </p>
        )}

        {predictionResult && (
          <div className="mt-6 p-4 border border-gray-200 rounded-md bg-gray-50">
            <h3 className="text-lg font-medium text-gray-700 mb-2">Prediction Results:</h3>
            <pre className="bg-gray-100 p-3 rounded-md text-sm text-gray-800 overflow-x-auto">
              {JSON.stringify(predictionResult, null, 2)}
            </pre>
          </div>
        )}
      </div>

      <p className="mt-8 text-gray-500 text-sm">
        Built with Next.js, React-Query, Zustand, FastAPI.
      </p>
    </div>
  );
}