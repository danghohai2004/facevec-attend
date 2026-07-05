"use client";

import * as React from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { toast } from "sonner";
import { ShiftSettingsSchema, useShiftSettings } from "@/lib/shift-settings";
import type { ShiftSettings } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export function ShiftSettingsForm() {
  const { settings, updateSettings } = useShiftSettings();
  const form = useForm<ShiftSettings>({
    resolver: zodResolver(ShiftSettingsSchema),
    defaultValues: settings,
  });

  React.useEffect(() => {
    form.reset(settings);
  }, [form, settings]);

  const onSubmit = async (values: ShiftSettings) => {
    const saved = await updateSettings(values);
    if (saved) {
      toast.success("Shift settings saved.");
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Attendance Time Windows</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Check-in Start</Label>
              <Input type="time" {...form.register("checkInStart")} />
              {form.formState.errors.checkInStart && (
                <p className="text-xs text-destructive">
                  {form.formState.errors.checkInStart.message}
                </p>
              )}
            </div>
            <div className="space-y-2">
              <Label>Check-in End</Label>
              <Input type="time" {...form.register("checkInEnd")} />
              {form.formState.errors.checkInEnd && (
                <p className="text-xs text-destructive">
                  {form.formState.errors.checkInEnd.message}
                </p>
              )}
            </div>
            <div className="space-y-2">
              <Label>Check-out Start</Label>
              <Input type="time" {...form.register("checkOutStart")} />
              {form.formState.errors.checkOutStart && (
                <p className="text-xs text-destructive">
                  {form.formState.errors.checkOutStart.message}
                </p>
              )}
            </div>
            <div className="space-y-2">
              <Label>Check-out End</Label>
              <Input type="time" {...form.register("checkOutEnd")} />
              {form.formState.errors.checkOutEnd && (
                <p className="text-xs text-destructive">
                  {form.formState.errors.checkOutEnd.message}
                </p>
              )}
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Button type="submit">Save Settings</Button>
            <Button
              type="button"
              variant="outline"
              onClick={() => form.reset(settings)}
            >
              Reset
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
