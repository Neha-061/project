import { Badge } from "@/components/ui/badge";
import { Spinner } from "@/components/ui/spinner";
import { Skeleton } from "@/components/ui/skeleton";

export function SkeletonLoading() {
  return (
    <div className="flex w-xl flex-col gap-3 mt-6">
      <div className="flex w-full gap-3">
        <div className="w-2/5">
          <Skeleton className="aspect-video h-full w-full" />
        </div>
        <div className="flex w-3/5 flex-col gap-2">
          <div className="flex flex-col gap-3">
            <Skeleton className="h-4 w-20" />
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-4 w-full" />
          </div>
          <div className="flex flex-col gap-3">
            <Skeleton className="h-4 w-20" />
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-4 w-full" />
          </div>
          <div className="flex flex-col gap-3">
            <Skeleton className="h-8 w-full" />
          </div>
        </div>
      </div>
      <div>
        <Badge variant="secondary" className={"p-3 rounded-md"}>
          <Spinner data-icon="inline-start" />
          Generating Summary
        </Badge>
      </div>
    </div>
  );
}
