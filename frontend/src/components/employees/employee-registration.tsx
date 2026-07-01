"use client";

import * as React from "react";
import Webcam from "react-webcam";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm, useWatch } from "react-hook-form";
import { z } from "zod";
import { toast } from "sonner";
import { createEmployee } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

const registrationSchema = z.object({
  name: z.string().min(2, "Name is required."),
  empCode: z.string().min(2, "Employee code is required."),
  files: z.array(z.instanceof(File)).min(1, "Capture at least one image."),
});

type RegistrationValues = z.infer<typeof registrationSchema>;

const videoConstraints = {
  width: 1280,
  height: 720,
  facingMode: "user",
};

const AUTO_CAPTURE_INTERVAL_MS = 1500;
const MAX_CAPTURED_FRAMES = 20;

function dataUrlToFile(dataUrl: string, filename: string) {
  const [header, base64] = dataUrl.split(",");
  const mime = header.match(/:(.*?);/)?.[1] ?? "image/jpeg";
  const binary = atob(base64);
  const array = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    array[i] = binary.charCodeAt(i);
  }
  return new File([array], filename, { type: mime });
}

function CapturedFramePreview({
  file,
  index,
}: {
  file: File;
  index: number;
}) {
  const imageRef = React.useRef<HTMLImageElement>(null);

  React.useEffect(() => {
    const url = URL.createObjectURL(file);
    const image = imageRef.current;
    if (image) {
      image.src = url;
    }
    return () => {
      image?.removeAttribute("src");
      URL.revokeObjectURL(url);
    };
  }, [file]);

  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      ref={imageRef}
      alt={`capture-${index}`}
      className="h-20 w-full object-cover"
    />
  );
}

export function EmployeeRegistration() {
  const [step, setStep] = React.useState<1 | 2>(1);
  const [autoCaptureEnabled, setAutoCaptureEnabled] = React.useState(true);
  const limitToastShownRef = React.useRef(false);
  const webcamRef = React.useRef<Webcam>(null);
  const queryClient = useQueryClient();

  const form = useForm<RegistrationValues>({
    resolver: zodResolver(registrationSchema),
    defaultValues: {
      name: "",
      empCode: "",
      files: [],
    },
  });

  const files = useWatch({ control: form.control, name: "files" }) ?? [];

  const createMutation = useMutation({
    mutationFn: createEmployee,
    onSuccess: () => {
      toast.success("Employee registered successfully.");
      form.reset();
      setStep(1);
      setAutoCaptureEnabled(true);
      queryClient.invalidateQueries({ queryKey: ["employees"] });
    },
    onError: (error: Error) => {
      toast.error(error.message || "Unable to register employee.");
    },
  });

  const handleNext = async () => {
    const valid = await form.trigger(["name", "empCode"]);
    if (valid) {
      setStep(2);
    }
  };

  const handleCapture = React.useCallback((options?: {
    silent?: boolean;
    stopAutoOnLimit?: boolean;
  }) => {
    const currentFiles = form.getValues("files") ?? [];
    if (currentFiles.length >= MAX_CAPTURED_FRAMES) {
      if (!limitToastShownRef.current) {
        toast.error(`Maximum ${MAX_CAPTURED_FRAMES} frames reached.`);
        limitToastShownRef.current = true;
      }
      if (options?.stopAutoOnLimit) {
        setAutoCaptureEnabled(false);
      }
      return;
    }
    const screenshot = webcamRef.current?.getScreenshot();
    if (!screenshot) {
      if (!options?.silent) {
        toast.error("Unable to capture a frame.");
      }
      return;
    }
    const file = dataUrlToFile(screenshot, `capture-${Date.now()}.jpg`);
    const nextFiles = [...currentFiles, file];
    form.setValue("files", nextFiles, { shouldValidate: true });
  }, [form]);

  React.useEffect(() => {
    if (step !== 2 || !autoCaptureEnabled || createMutation.isPending) {
      return;
    }

    const interval = window.setInterval(() => {
      handleCapture({ silent: true, stopAutoOnLimit: true });
    }, AUTO_CAPTURE_INTERVAL_MS);

    return () => window.clearInterval(interval);
  }, [step, autoCaptureEnabled, createMutation.isPending, handleCapture]);

  React.useEffect(() => {
    if (files.length < MAX_CAPTURED_FRAMES) {
      limitToastShownRef.current = false;
    }
  }, [files.length]);

  const removeFile = (index: number) => {
    const nextFiles = files.filter((_, idx) => idx !== index);
    form.setValue("files", nextFiles, { shouldValidate: true });
  };

  const onSubmit = (values: RegistrationValues) => {
    createMutation.mutate({
      name: values.name,
      empCode: values.empCode,
      files: values.files,
    });
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Register New Employee</CardTitle>
        <Badge variant="outline">Step {step} of 2</Badge>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
          <div className="space-y-5">
            <div className="space-y-2">
              <Label>Name</Label>
              <Input placeholder="Full name" {...form.register("name")} />
              {form.formState.errors.name && (
                <p className="text-xs text-destructive">
                  {form.formState.errors.name.message}
                </p>
              )}
            </div>
            <div className="space-y-2">
              <Label>Employee Code</Label>
              <Input placeholder="EMP-1004" {...form.register("empCode")} />
              {form.formState.errors.empCode && (
                <p className="text-xs text-destructive">
                  {form.formState.errors.empCode.message}
                </p>
              )}
            </div>

            {step === 1 ? (
              <div className="flex items-center gap-3">
                <Button type="button" onClick={handleNext}>
                  Continue to face capture
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => form.reset()}
                >
                  Reset
                </Button>
              </div>
            ) : (
              <div className="flex items-center gap-3">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => {
                    setStep(1);
                    setAutoCaptureEnabled(true);
                  }}
                >
                  Back
                </Button>
                <Button
                  type="button"
                  onClick={form.handleSubmit(onSubmit)}
                  disabled={createMutation.isPending}
                >
                  Submit Registration
                </Button>
              </div>
            )}
          </div>

          <div className="space-y-4">
            <div className="space-y-3">
              <div className="overflow-hidden rounded-xl border bg-muted/40">
                <Webcam
                  ref={webcamRef}
                  audio={false}
                  screenshotFormat="image/jpeg"
                  videoConstraints={videoConstraints}
                  className="h-[220px] w-full object-cover"
                />
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setAutoCaptureEnabled((prev) => !prev)}
                >
                  {autoCaptureEnabled ? "Pause Auto Capture" : "Resume Auto Capture"}
                </Button>
                <Button type="button" onClick={() => handleCapture()}>
                  Capture Frame
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                Auto capture saves a frame every {AUTO_CAPTURE_INTERVAL_MS / 1000}s (max{" "}
                {MAX_CAPTURED_FRAMES} frames).
              </p>
            </div>

            <div className="space-y-2">
              <Label>Captured Frames</Label>
              {files.length === 0 ? (
                <p className="text-xs text-muted-foreground">
                  No images captured yet.
                </p>
              ) : (
                <div className="grid grid-cols-3 gap-2">
                  {files.map((file, index) => (
                    <div
                      key={`${file.name}-${index}`}
                      className="group relative overflow-hidden rounded-lg border bg-muted/40"
                    >
                      <CapturedFramePreview file={file} index={index} />
                      <button
                        type="button"
                        onClick={() => removeFile(index)}
                        className="absolute right-1 top-1 rounded-full bg-background/80 px-2 py-0.5 text-[10px] opacity-0 transition group-hover:opacity-100"
                      >
                        Remove
                      </button>
                    </div>
                  ))}
                </div>
              )}
              {form.formState.errors.files && (
                <p className="text-xs text-destructive">
                  {form.formState.errors.files.message}
                </p>
              )}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
