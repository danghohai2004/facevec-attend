"use client";

import * as React from "react";
import Webcam from "react-webcam";
import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";
import { checkIn, checkOut } from "@/lib/api";
import { useShiftSettings } from "@/lib/shift-settings";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

type Detection = {
  id: string;
  name: string;
  empId: string;
  status: "checkin" | "checkout";
  box: { x: number; y: number; width: number; height: number };
};

const videoConstraints = {
  width: 1280,
  height: 720,
  facingMode: "user",
};

export function AttendanceClient() {
  const { settings } = useShiftSettings();

  const [isStreaming, setIsStreaming] = React.useState(false);
  const [detection] = React.useState<Detection | null>(null);
  const [action, setAction] = React.useState<"checkin" | "checkout">("checkin");
  const [manualEmpId, setManualEmpId] = React.useState("");

  const checkInMutation = useMutation({
    mutationFn: checkIn,
    onSuccess: () => {
      toast.success("Check-in recorded successfully.");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Unable to record check-in.");
    },
  });

  const checkOutMutation = useMutation({
    mutationFn: checkOut,
    onSuccess: () => {
      toast.success("Check-out recorded successfully.");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Unable to record check-out.");
    },
  });

  const handleSubmitAttendance = () => {
    const targetEmpId = manualEmpId || detection?.empId;
    if (!targetEmpId) {
      toast.error("Provide an employee ID or wait for detection.");
      return;
    }

    if (action === "checkin") {
      checkInMutation.mutate({ empId: targetEmpId });
    } else {
      checkOutMutation.mutate({ empId: targetEmpId });
    }
  };

  return (
    <Tabs defaultValue="realtime" className="space-y-6">
      <TabsList>
        <TabsTrigger value="realtime">Real-time Mode</TabsTrigger>
      </TabsList>

      <TabsContent value="realtime" className="space-y-6">
        <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle>Live Camera Feed</CardTitle>
              <Badge variant={isStreaming ? "default" : "secondary"}>
                {isStreaming ? "Streaming" : "Paused"}
              </Badge>
            </CardHeader>
            <CardContent>
              <div className="relative overflow-hidden rounded-xl border bg-muted/40">
                <Webcam
                  audio={false}
                  screenshotFormat="image/jpeg"
                  videoConstraints={videoConstraints}
                  className="h-[360px] w-full object-cover"
                />
                {detection && (
                  <div
                    className="absolute border-2 border-primary"
                    style={{
                      left: `${detection.box.x}%`,
                      top: `${detection.box.y}%`,
                      width: `${detection.box.width}%`,
                      height: `${detection.box.height}%`,
                    }}
                  />
                )}
              </div>
              <div className="mt-4 flex flex-wrap items-center gap-3">
                <Button onClick={() => setIsStreaming((prev) => !prev)}>
                  {isStreaming ? "Stop Capture" : "Start Capture"}
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Recognition Status</CardTitle>
            </CardHeader>
            <CardContent className="space-y-5">
              <div className="space-y-2">
                <Label>Attendance Action</Label>
                <Select
                  value={action}
                  onValueChange={(value) =>
                    setAction(value as "checkin" | "checkout")
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="checkin">Check-in</SelectItem>
                    <SelectItem value="checkout">Check-out</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Employee ID (manual override)</Label>
                <Input
                  value={manualEmpId}
                  onChange={(event) => setManualEmpId(event.target.value)}
                  placeholder="EMP-1024 or internal ID"
                />
              </div>

              <Separator />

              <div className="space-y-3">
                <p className="text-sm font-medium">Detected Employee</p>
                {detection ? (
                  <div className="rounded-lg border bg-muted/40 p-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-semibold">{detection.name}</p>
                        <p className="text-xs text-muted-foreground">
                          {detection.empId}
                        </p>
                      </div>
                      <Badge>
                        {detection.status === "checkin"
                          ? "Check-in ready"
                          : "Check-out ready"}
                      </Badge>
                    </div>
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    Waiting for a face to be detected...
                  </p>
                )}
              </div>

              <div className="space-y-2 rounded-lg border bg-muted/30 p-3">
                <p className="text-xs font-medium text-muted-foreground">
                  Active Shift Window
                </p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <span>Check-in: {settings.checkInStart}</span>
                  <span>Check-in End: {settings.checkInEnd}</span>
                  <span>Check-out: {settings.checkOutStart}</span>
                  <span>Check-out End: {settings.checkOutEnd}</span>
                </div>
              </div>

              <Button
                onClick={handleSubmitAttendance}
                disabled={checkInMutation.isPending || checkOutMutation.isPending}
              >
                Submit {action === "checkin" ? "Check-in" : "Check-out"}
              </Button>
            </CardContent>
          </Card>
        </div>
      </TabsContent>
    </Tabs>
  );
}
