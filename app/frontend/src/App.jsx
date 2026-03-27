import { ModeToggle } from "@/components/mode-toggle";
import FileUploadForm from "@/components/file-upload-form-3";
import { SkeletonLoading } from "@/components/skeleton-loading";
import { useState } from "react";
import { OutputCard } from "@/components/output-card";

function App() {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  return (
    <>
      <div className="min-h-screen px-4 md:px-100 bg-background text-foreground">
        <div className="min-h-full flex flex-col items-center pt-35 md:pt-55">
          <Status
            loading={loading}
            data={data}
            error={error}
            setData={setData}
            setLoading={setLoading}
          />
          {/* <OutputCard/> */}

        </div>
      </div>
      <div className="fixed top-4 right-4 z-50">
        <ModeToggle />
      </div>
    </>
  );
}

function Status({ loading, data, error, setData, setLoading }) {
  if (data) {
    return <OutputCard data={data} />;
  }
  if (loading) {
    return <SkeletonLoading />;
  }
  // if (error) {
  //   return <div>Error</div>;
  // }
  return <FileUploadForm setLoading={setLoading} setData={setData}/>;
}

export default App;
