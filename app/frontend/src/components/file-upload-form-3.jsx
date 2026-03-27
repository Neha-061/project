import { Upload, X } from "lucide-react";
import { useState,useEffect } from "react";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  FileUpload,
  FileUploadDropzone,
  FileUploadItem,
  FileUploadItemDelete,
  FileUploadItemMetadata,
  FileUploadItemPreview,
  FileUploadList,
  FileUploadTrigger,
} from "@/components/ui/file-upload";

export default function FileUploadForm({ setLoading, setData }) {
  const [files, setFiles] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!error) return;

    const timer = setTimeout(() => {
      setError(null);
    }, 2000);

    return () => clearTimeout(timer);
  }, [error]);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (files.length === 0) {
      setError("Please upload at least one file");
      return;
    }

    setLoading(true);
    setError(null);

     try {
      for (const file of files) {
        const formData = new FormData();
        formData.append("file", file);

        const response = await axios.post("http://127.0.0.1:8000/api/ask", formData);
        
        if(response.status === 200){
          console.log(response);
          setData({
            title: response.data.image_name,
            conclusion: response.data.conclusion,
            image: URL.createObjectURL(file),
            score: response.data.score,
            key_observations: response.data.key_observations,
            summery: response.data.summery
          });
          setLoading(false);
        }
        
        
      }

    } catch (error) {
      console.error(error);
    }



  };

  const handleFilesChange = (newFiles) => {
    setFiles(newFiles);
    if (newFiles.length > 0) {
      setError(null);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-md space-y-4">
      <div className="space-y-2">
        <Label>
          Upload Image <span className="text-destructive">*</span>
        </Label>

        <FileUpload
          value={files}
          onValueChange={handleFilesChange}
          maxFiles={1}
          maxSize={10 * 1024 * 1024}
          accept=".jpg, .jpeg, .png, .webp"
          // required
        >
          <FileUploadDropzone
            className={`${error ? "border-destructive" : "border-accent"}`}
          >
            <div className="flex flex-col items-center gap-1 text-center">
              <div className="flex items-center justify-center rounded-full border p-2.5">
                <Upload className="size-6 text-muted-foreground" />
              </div>

              <p className="text-sm font-medium">Drop Image Here</p>

              <p className="text-xs text-muted-foreground">
                JPG, PNG, JPEG, WEBP
              </p>
            </div>

            <FileUploadTrigger asChild>
              <Button variant="outline" size="sm" className="mt-2">
                Select from Device
              </Button>
            </FileUploadTrigger>
          </FileUploadDropzone>

          <FileUploadList>
            {files.map((file, index) => (
              <FileUploadItem key={index} value={file}>
                <FileUploadItemPreview />
                <FileUploadItemMetadata />

                <FileUploadItemDelete asChild>
                  <Button variant="ghost" size="icon" className="size-7">
                    <X className="size-4" />
                  </Button>
                </FileUploadItemDelete>
              </FileUploadItem>
            ))}
          </FileUploadList>
        </FileUpload>

        {error && <p className="text-sm text-destructive">{error}</p>}
      </div>

      <Button type="submit" size="lg" className="w-full h-12 md:h-10">
        Generate Score
      </Button>
    </form>
  );
}
