import { useState } from "react";
import axios from "axios";
import { useDropzone } from "react-dropzone";
import { FaSun, FaMoon } from "react-icons/fa";

const App = () => {
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [message, setMessage] = useState("");
  const [dark, setDark] = useState(true);

  const onDrop = (acceptedFiles) => {
    setFiles(acceptedFiles);
    const previewUrls = acceptedFiles.map((file) =>
      URL.createObjectURL(file)
    );
    setPreviews(previewUrls);
  };

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: { "image/*": [] },
    multiple: true,
  });

  const uploadImage = async () => {
    if (files.length === 0) {
      setMessage("⚠ Please upload images first");
      return;
    }

    try {
      for (const file of files) {
        const formData = new FormData();
        formData.append("file", file);

        await axios.post("http://127.0.0.1:8000/upload", formData);
      }

      setMessage("✅ All images uploaded successfully");
    } catch (error) {
      console.error(error);
      setMessage("❌ Upload failed");
    }
  };

  return (
    <div className={dark ? "bg-slate-900 text-slate-200 min-h-screen" : "bg-white text-slate-900 min-h-screen"}>
      
      {/* HEADER */}
      <div className="flex justify-center items-center px-10 py-5 bg-slate-700 text-white relative">
        
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-full bg-indigo-500 flex items-center justify-center font-bold text-sm">
            AI
          </div>
          <h1 className="text-3xl tracking-widest font-bold">VisionAI</h1>
        </div>

        <div
          className="absolute right-10 cursor-pointer text-lg opacity-80 hover:opacity-100"
          onClick={() => setDark(!dark)}
        >
          {dark ? <FaSun /> : <FaMoon />}
        </div>
      </div>

      {/* HERO */}
      <div className="mt-10 text-center">
        <h2 className="text-2xl font-semibold">
          AI Generated Image Quality Analyzer
        </h2>
        <p className="mt-2 text-slate-500 dark:text-slate-400">
          Upload AI generated images and detect visual defects using deep learning models.
        </p>
      </div>

      {/* CARD */}
      <div className={`w-[520px] mx-auto mt-12 p-10 rounded-2xl shadow-xl transition 
        ${dark ? "bg-slate-800 shadow-black/40" : "bg-white shadow-black/10 border"}`}>

        <h3 className="text-lg font-semibold">Upload Images</h3>

        {/* DROPZONE */}
        <div
          {...getRootProps()}
          className={`mt-5 p-10 border-2 border-dashed rounded-xl cursor-pointer transition transform hover:scale-[1.02]
          ${dark ? "bg-black border-indigo-400 text-slate-200" : "bg-indigo-100 border-indigo-500 text-slate-900"}`}
        >
          <input {...getInputProps()} />
          <p className="font-semibold">📂 Drag & Drop Images Here</p>
          <span className="text-sm opacity-70">or click to browse</span>
        </div>

        {/* PREVIEW */}
        <div className="flex flex-wrap gap-3 mt-6 justify-center">
          {previews.map((src, index) => (
            <img
              key={index}
              src={src}
              alt="preview"
              className="w-52 rounded-xl shadow-lg"
            />
          ))}
        </div>

        {/* BUTTON */}
        <button
          onClick={uploadImage}
          className="mt-8 px-6 py-3 bg-indigo-500 text-white rounded-lg hover:translate-y-[-2px] hover:shadow-lg hover:shadow-indigo-500/40 transition"
        >
          Analyze Images
        </button>

        {/* MESSAGE */}
        {message && (
          <div
            className={`mt-6 p-3 rounded-lg font-semibold
            ${dark ? "bg-green-900 text-green-200" : "bg-green-100 text-green-700"}`}
          >
            {message}
          </div>
        )}
      </div>
    </div>
  );
};

export default App;