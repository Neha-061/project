import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardAction,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export function OutputCard({data}) {
  return (
    <div className="flex flex-col gap-5 items-center">
      <Card className=" w-4xl overflow-hidden grid md:grid-cols-2 p-0 -mt-15">
        {/* LEFT: Image */}
        <div className="relative">
          <img
            src={data.image}
            alt="Event cover"
            className="h-100 w-[90%] object-cover"
          />
        </div>
        {/* RIGHT: Content */}
        <div className="flex flex-col justify-between p-4 max-h-100 overflow-y-scroll">
          <CardHeader className="p-0">
            <CardTitle>{data.title}</CardTitle>
            <br />

            <CardDescription className="col-span-2 text-lg">
              <div className="font-bold text-xl">Summery:</div>
              {data.summery}
              <br />
              <br />
              <div className="font-bold text-xl">Key Observations:</div>
              {data.key_observations.map((observation, index) => (
                <>
                  <div key={index}>&nbsp;&nbsp;- {observation}</div>
                </>
              ))}
              <br />
              <div className="font-bold text-xl">Conclusion:</div>
              {data.conclusion}
              <br />
            </CardDescription>
          </CardHeader>
        </div>
      </Card>
      <Button variant="secondary" className="text-3xl w-fit px-8 py-5">
        Socre - {data.score}
      </Button>
    </div>
  );
}
